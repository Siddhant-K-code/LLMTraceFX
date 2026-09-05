from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llmtracefx.cache_audit.adapters.reference import (
    ReferenceCacheAdapter,
    SyntheticCacheEngine,
    _EngineEntry,
)
from llmtracefx.cache_audit.api import sanitize_audit_bundle
from llmtracefx.cache_audit.bundle import (
    PUBLIC_REDACTED_FACT_SCOPE,
    PUBLIC_REDACTED_TIMING_EXCLUSIONS,
    PUBLIC_REDACTED_TIMING_SCOPE,
    PUBLIC_REDACTED_TIMING_UNIT,
    CacheAuditBundleError,
    _verify_manifest_chronology,
    read_bundle,
    verify_bundle,
    write_bundle,
)
from llmtracefx.cache_audit.expected import MLXCacheOracle
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import (
    AuditManifest,
    CacheConfig,
    CostEvidence,
    EligibilityStatus,
    MemoryEvidence,
    OutputEvidence,
    PublicationMode,
    RequestSpec,
    ReuseEvidence,
    ScenarioKind,
    Verdict,
)
from llmtracefx.cache_audit.verdicts import classify_request
from llmtracefx.cache_audit.workloads import (
    adversarial_requests,
    eviction_requests,
    gated_extension_requests,
)
from llmtracefx.optimizer.schema import Measurement, MetricProvenance


def test_reference_adapter_exercises_truth_states() -> None:
    records = ReferenceCacheAdapter().run(adversarial_requests())
    verdicts = {record.spec.request_id: record.verdict for record in records}
    assert verdicts["cold"] is Verdict.VERIFIED_MISS
    assert verdicts["identical"] is Verdict.VERIFIED_HIT
    assert verdicts["first-token-mutation"] is Verdict.VERIFIED_MISS
    assert verdicts["within-block-mutation"] is Verdict.PARTIAL_REUSE
    assert verdicts["namespace-isolation"] is Verdict.VERIFIED_MISS


def test_bundle_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    requests = adversarial_requests()
    run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=requests,
        cache_config=CacheConfig(
            namespace_id="synthetic-tenants",
            cache_type="token_trie",
            max_entries=32,
        ),
        output_dir=output,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        publication_mode=PublicationMode.PUBLIC_SYNTHETIC,
        seed=7,
        created_at="2026-01-01T00:00:00Z",
    )
    verified = verify_bundle(output)
    assert verified["request_count"] == len(requests)
    assert verified["token_identity_reproducible"] is True

    summary = output / "summary.json"
    value = json.loads(summary.read_text())
    value["request_count"] += 1
    summary.write_text(json.dumps(value))
    with pytest.raises(CacheAuditBundleError, match="checksum mismatch"):
        verify_bundle(output)


def test_bundle_rejects_symlinked_artifact(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    requests = adversarial_requests()
    run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=requests,
        cache_config=CacheConfig(
            namespace_id="synthetic-tenants",
            cache_type="token_trie",
        ),
        output_dir=output,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    summary = output / "summary.json"
    target = tmp_path / "summary.json"
    target.write_bytes(summary.read_bytes())
    summary.unlink()
    summary.symlink_to(target)
    with pytest.raises(CacheAuditBundleError, match="non-symlink"):
        verify_bundle(output)


def test_controlled_eviction_and_gated_extensions() -> None:
    eviction = ReferenceCacheAdapter(max_entries=1).run(eviction_requests())
    assert eviction[-1].verdict is Verdict.EVICTED
    byte_requests = list(eviction_requests())
    byte_requests[-1] = replace(byte_requests[-1], scenario=ScenarioKind.EVICTION_BYTES)
    byte_eviction = ReferenceCacheAdapter(max_bytes=500).run(byte_requests)
    assert byte_eviction[-1].verdict is Verdict.EVICTED

    gated = ReferenceCacheAdapter().run(gated_extension_requests())
    assert {record.verdict for record in gated} == {Verdict.UNSUPPORTED}


def test_independent_engine_attestation_mutation_does_not_change_oracle() -> None:
    records = ReferenceCacheAdapter(cached_token_offsets={"identical": -1}).run(
        adversarial_requests()[:2]
    )
    warm = records[1]
    assert (
        warm.reuse.policy_reusable_tokens.value
        == len(warm.spec.input_token_ids or ()) - 1
    )
    assert warm.verdict is Verdict.INVALID
    assert "engine_attestation_mismatch" in warm.verdict_reasons


def test_independent_prompt_operation_off_by_one_detects_recomputation() -> None:
    records = ReferenceCacheAdapter(prompt_operation_offsets={"identical": 1}).run(
        adversarial_requests()[:2]
    )
    assert records[1].verdict is Verdict.RECOMPUTED
    assert records[1].reuse.policy_reusable_tokens.value == (
        records[1].reuse.engine_cached_tokens.value
    )


def test_independent_baseline_failure_only_gates_output() -> None:
    records = ReferenceCacheAdapter(corrupt_baseline_requests=("identical",)).run(
        adversarial_requests()[:2]
    )
    warm = records[1]
    assert warm.verdict is Verdict.VERIFIED_HIT
    assert warm.eligibility.output_equivalence is EligibilityStatus.INELIGIBLE
    assert warm.output.output_token_ids != warm.output.baseline_token_ids


def test_synthetic_engine_prefix_compaction_regression() -> None:
    engine = SyntheticCacheEngine(
        max_entries=32,
        max_bytes=1 << 30,
        bytes_per_token=64,
    )
    oracle = MLXCacheOracle(max_entries=32, max_bytes=1 << 30)
    original = RequestSpec(
        request_id="original",
        scenario=ScenarioKind.COLD,
        order=0,
        input_token_ids=(1, 2),
        input_token_count=2,
        output_tokens=1,
    )
    first = engine.execute(original)
    first_sequence = original.input_token_ids + first.output_token_ids
    oracle.insert(
        entry_id=original.request_id,
        model_key="synthetic-tiny-model",
        namespace_id=original.namespace_id,
        tokens=first_sequence,
        nbytes=len(first_sequence) * 64,
    )
    extension_tokens = first_sequence + (9,)
    extension = RequestSpec(
        request_id="extension",
        scenario=ScenarioKind.SUFFIX_CHANGE,
        order=1,
        input_token_ids=extension_tokens,
        input_token_count=len(extension_tokens),
        output_tokens=1,
    )
    expected_extension = oracle.lookup(
        "synthetic-tiny-model", extension.namespace_id, extension_tokens
    )
    observed_extension = engine.execute(extension)
    assert observed_extension.cached_tokens == expected_extension.policy_reusable_tokens
    extension_sequence = extension_tokens + observed_extension.output_token_ids
    oracle.insert(
        entry_id=extension.request_id,
        model_key="synthetic-tiny-model",
        namespace_id=extension.namespace_id,
        tokens=extension_sequence,
        nbytes=len(extension_sequence) * 64,
    )
    expected_original = oracle.lookup(
        "synthetic-tiny-model", original.namespace_id, original.input_token_ids
    )
    observed_original = engine.execute(replace(original, request_id="original-again"))
    assert observed_original.cached_tokens == expected_original.policy_reusable_tokens
    assert observed_original.cached_tokens == 1


def test_synthetic_engine_randomized_differential_state_machine() -> None:
    randomizer = random.Random(1729)
    engine = SyntheticCacheEngine(
        max_entries=256,
        max_bytes=1 << 30,
        bytes_per_token=64,
    )
    oracle = MLXCacheOracle(max_entries=256, max_bytes=1 << 30)
    for order in range(200):
        length = randomizer.randint(1, 12)
        tokens = tuple(randomizer.randint(0, 15) for _ in range(length))
        namespace = f"namespace-{randomizer.randint(0, 2)}"
        request = RequestSpec(
            request_id=f"random-{order}",
            scenario=ScenarioKind.DUPLICATE,
            order=order,
            input_token_ids=tokens,
            input_token_count=length,
            output_tokens=randomizer.randint(1, 3),
            namespace_id=namespace,
        )
        expected = oracle.lookup("synthetic-tiny-model", namespace, tokens)
        observed = engine.execute(request)
        assert observed.cached_tokens == expected.policy_reusable_tokens
        sequence = tokens + observed.output_token_ids
        oracle.insert(
            entry_id=request.request_id,
            model_key="synthetic-tiny-model",
            namespace_id=namespace,
            tokens=sequence,
            nbytes=len(sequence) * 64,
        )


@pytest.mark.parametrize(
    ("resident", "request_tokens", "expected_kind"),
    [
        ((1, 2, 3), (1, 2, 3), "exact"),
        ((1, 2), (1, 2, 3, 4), "shorter"),
        ((1, 2, 3, 4), (1, 2, 3), "longer_trimmed"),
        ((1, 2, 9, 9, 9), (1, 2, 3, 4), "longer_trimmed"),
    ],
)
def test_capacity_two_lookup_matches_pinned_mlx_policy_without_recency_touch(
    resident: tuple[int, ...],
    request_tokens: tuple[int, ...],
    expected_kind: str,
) -> None:
    engine = SyntheticCacheEngine(
        max_entries=2,
        max_bytes=1 << 30,
        bytes_per_token=64,
    )
    oracle = MLXCacheOracle(max_entries=2, max_bytes=1 << 30)
    for entry_id, tokens in (("oldest", resident), ("newer", (8, 8, 8))):
        engine._insert(_EngineEntry(entry_id, "default", tokens, len(tokens) * 64))
        oracle.insert(
            entry_id=entry_id,
            model_key="synthetic-tiny-model",
            namespace_id="default",
            tokens=tokens,
            nbytes=len(tokens) * 64,
        )

    expected = oracle.lookup("synthetic-tiny-model", "default", request_tokens)
    observed, _ = engine._lookup("default", request_tokens)
    assert expected.match_kind == expected_kind
    assert observed == expected.policy_reusable_tokens

    engine_evicted = engine._insert(_EngineEntry("third", "default", (7, 7, 7), 3 * 64))
    oracle_evicted = oracle.insert(
        entry_id="third",
        model_key="synthetic-tiny-model",
        namespace_id="default",
        tokens=(7, 7, 7),
        nbytes=3 * 64,
    )
    assert engine_evicted == set(oracle_evicted) == {"oldest"}


def test_synthetic_engine_randomized_low_capacity_eviction_differential() -> None:
    randomizer = random.Random(31337)
    engine = SyntheticCacheEngine(
        max_entries=2,
        max_bytes=7 * 64,
        bytes_per_token=64,
    )
    oracle = MLXCacheOracle(max_entries=2, max_bytes=7 * 64)
    for order in range(500):
        length = randomizer.randint(1, 6)
        tokens = tuple(randomizer.randint(0, 5) for _ in range(length))
        namespace = f"tenant-{randomizer.randint(0, 1)}"
        request = RequestSpec(
            request_id=f"low-capacity-{order}",
            scenario=ScenarioKind.DUPLICATE,
            order=order,
            input_token_ids=tokens,
            input_token_count=length,
            output_tokens=1,
            namespace_id=namespace,
        )
        expected = oracle.lookup("synthetic-tiny-model", namespace, tokens)
        observed = engine.execute(request)
        assert observed.cached_tokens == expected.policy_reusable_tokens
        sequence = tokens + observed.output_token_ids
        oracle.insert(
            entry_id=request.request_id,
            model_key="synthetic-tiny-model",
            namespace_id=namespace,
            tokens=sequence,
            nbytes=len(sequence) * 64,
        )
        assert engine.entry_ids == oracle.entry_ids


def test_writer_rejects_workload_digest_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=source,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(CacheAuditBundleError, match="specifications"):
        write_bundle(
            tmp_path / "invalid",
            replace(manifest, workload_digest="sha256:" + "0" * 64),
            records,
        )


def test_runner_rejects_caller_backend_and_cache_metadata_mismatch(
    tmp_path: Path,
) -> None:
    request = adversarial_requests()[:1]
    with pytest.raises(ValueError, match="backend version"):
        run_audit(
            adapter=ReferenceCacheAdapter(),
            requests=request,
            cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
            output_dir=tmp_path / "wrong-version",
            backend_version="caller-claim",
            model_id="synthetic-tiny-model",
            tokenizer_id="integer-tokenizer-v1",
        )
    with pytest.raises(ValueError, match="max_entries"):
        run_audit(
            adapter=ReferenceCacheAdapter(max_entries=4),
            requests=request,
            cache_config=CacheConfig(
                namespace_id="synthetic",
                cache_type="token_trie",
                max_entries=5,
            ),
            output_dir=tmp_path / "wrong-limit",
            backend_version="1",
            model_id="synthetic-tiny-model",
            tokenizer_id="integer-tokenizer-v1",
        )


def test_runner_persists_adapter_owned_runtime_and_limits(tmp_path: Path) -> None:
    manifest, _ = run_audit(
        adapter=ReferenceCacheAdapter(max_entries=7, max_bytes=8192),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=tmp_path / "bundle",
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    assert manifest.runtime_identity["synthetic_engine"] == (
        "independent-state-machine-v2"
    )
    assert manifest.cache_config.max_entries == 7
    assert manifest.cache_config.max_bytes == 8192


def test_writer_rejects_mlx_manifest_identity_mismatch(tmp_path: Path) -> None:
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=tmp_path / "source",
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(CacheAuditBundleError, match="MLX manifest"):
        write_bundle(
            tmp_path / "invalid-mlx",
            replace(
                manifest,
                backend="mlx_lm_local",
                backend_version="0.0.0",
                cache_config=replace(
                    manifest.cache_config,
                    cache_type="mlx_lru_prompt_cache",
                ),
                model_artifact_digest="sha256:" + "1" * 64,
                runtime_identity={"mlx": "0.0.0", "mlx_lm": "0.0.0"},
            ),
            records,
        )


def test_public_synthetic_rejects_non_reference_backend(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=source,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    with pytest.raises(CacheAuditBundleError, match="adapter-owned identity"):
        write_bundle(
            tmp_path / "invalid",
            replace(
                manifest,
                backend="mlx_lm_local",
                publication_mode=PublicationMode.PUBLIC_SYNTHETIC,
            ),
            records,
        )


def test_public_synthetic_rejects_unapproved_token_workload(tmp_path: Path) -> None:
    request = replace(
        adversarial_requests()[0],
        request_id="declared-synthetic-but-unapproved",
        input_token_ids=(91, 92, 93),
        input_token_count=3,
    )
    with pytest.raises(CacheAuditBundleError, match="approved built-in"):
        run_audit(
            adapter=ReferenceCacheAdapter(),
            requests=(request,),
            cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
            output_dir=tmp_path / "invalid",
            backend_version="1",
            model_id="synthetic-tiny-model",
            tokenizer_id="integer-tokenizer-v1",
            publication_mode=PublicationMode.PUBLIC_SYNTHETIC,
            created_at="2026-01-01T00:00:00Z",
        )


def test_verifier_recomputes_output_token_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=source,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    record = records[0]
    assert record.output.output_token_ids is not None
    contradictory = replace(
        record,
        output=replace(record.output, baseline_token_ids=(999,)),
    )
    output = tmp_path / "contradictory"
    write_bundle(output, manifest, [contradictory])
    with pytest.raises(CacheAuditBundleError, match="baseline control mismatch"):
        verify_bundle(output)


def test_public_redaction_removes_correlating_identifiers_and_downgrades_verdicts(
    tmp_path: Path,
) -> None:
    base = adversarial_requests()[:2]
    requests = (
        replace(
            base[0],
            request_id="customer-cold",
            pair_id="customer-pair",
            namespace_id="tenant-acme",
        ),
        replace(
            base[1],
            request_id="customer-warm",
            pair_id="customer-pair",
            namespace_id="tenant-acme",
            expected_predecessors=("customer-cold",),
        ),
    )
    private = tmp_path / "private"
    run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=requests,
        cache_config=CacheConfig(namespace_id="tenant-acme", cache_type="token_trie"),
        output_dir=private,
        backend_version="1",
        model_id="/srv/private/customer-model",
        tokenizer_id="customer-tokenizer",
        created_at="2026-01-01T00:00:00Z",
    )

    public = tmp_path / "public"
    result = sanitize_audit_bundle(private, public)
    assert result["token_identity_reproducible"] is False
    manifest, records = read_bundle(public)
    assert manifest.model_id == "redacted-model"
    assert [record.verdict for record in records] == [
        Verdict.UNSUPPORTED,
        Verdict.ATTESTED_ONLY,
    ]
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in public.iterdir()
    )
    for private_value in (
        "customer-cold",
        "customer-warm",
        "customer-pair",
        "tenant-acme",
        "/srv/private/customer-model",
        "customer-tokenizer",
    ):
        assert private_value not in public_text


def test_public_redaction_replaces_all_fact_and_timing_scopes(
    tmp_path: Path,
) -> None:
    secret = "CONFIDENTIAL_SCOPE_NEVER_PUBLIC"
    source = tmp_path / "source"
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="private", cache_type="token_trie"),
        output_dir=source,
        backend_version="1",
        model_id="private-model",
        tokenizer_id="private-tokenizer",
        created_at="2026-01-01T00:00:00Z",
    )
    record = records[0]

    def private_fact(fact):
        return replace(fact, scope=secret, limitations=(secret,))

    poisoned = replace(
        record,
        reuse=ReuseEvidence(
            **{
                name: private_fact(getattr(record.reuse, name))
                for name in record.reuse.__dataclass_fields__
            }
        ),
        timing=replace(
            record.timing,
            total=Measurement(
                value=1.0,
                provenance=MetricProvenance.MEASURED_WALL_CLOCK,
                unit=secret,
            ),
            scope=secret,
            exclusions=(secret,),
        ),
        memory=MemoryEvidence(
            **{
                name: (
                    replace(
                        private_fact(getattr(record.memory, name)),
                        value=secret,
                    )
                    if name == "logical_cache_bytes"
                    else private_fact(getattr(record.memory, name))
                )
                for name in record.memory.__dataclass_fields__
            }
        ),
        output=OutputEvidence(
            output_token_ids=record.output.output_token_ids,
            baseline_token_ids=record.output.baseline_token_ids,
            token_identity=private_fact(record.output.token_identity),
            correctness=private_fact(record.output.correctness),
            finish_reason=record.output.finish_reason,
        ),
        cost=CostEvidence(
            billed=private_fact(record.cost.billed),
            estimated=private_fact(record.cost.estimated),
            currency=record.cost.currency,
        ),
        cache_before=(
            None
            if record.cache_before is None
            else replace(
                record.cache_before,
                **{
                    name: private_fact(getattr(record.cache_before, name))
                    for name in (
                        "entry_count",
                        "logical_bytes",
                        "valid_token_offsets",
                        "cache_classes",
                    )
                },
            )
        ),
        cache_after=(
            None
            if record.cache_after is None
            else replace(
                record.cache_after,
                **{
                    name: private_fact(getattr(record.cache_after, name))
                    for name in (
                        "entry_count",
                        "logical_bytes",
                        "valid_token_offsets",
                        "cache_classes",
                    )
                },
            )
        ),
    )
    private = tmp_path / "private"
    write_bundle(private, manifest, [poisoned])
    public = tmp_path / "public"
    sanitize_audit_bundle(private, public)
    _, sanitized = read_bundle(public)
    serialized = sanitized[0].to_dict()

    def values_for_key(value, key):
        if isinstance(value, dict):
            for item_key, item in value.items():
                if item_key == key:
                    yield item
                yield from values_for_key(item, key)
        elif isinstance(value, list):
            for item in value:
                yield from values_for_key(item, key)

    assert set(values_for_key(serialized, "scope")) == {
        PUBLIC_REDACTED_FACT_SCOPE,
        PUBLIC_REDACTED_TIMING_SCOPE,
    }
    assert serialized["timing"]["exclusions"] == list(PUBLIC_REDACTED_TIMING_EXCLUSIONS)
    assert serialized["timing"]["total"] is None
    assert serialized["memory"]["logical_cache_bytes"]["basis"] == "unavailable"
    assert all(
        measurement is None or measurement["unit"] == PUBLIC_REDACTED_TIMING_UNIT
        for name, measurement in serialized["timing"].items()
        if name not in {"scope", "exclusions"}
    )
    public_text = "\n".join(
        path.read_text(encoding="utf-8") for path in public.iterdir()
    )
    assert secret not in public_text
    assert PUBLIC_REDACTED_FACT_SCOPE in public_text
    assert PUBLIC_REDACTED_TIMING_SCOPE in public_text


def test_eviction_proof_rejects_cross_namespace_and_unrelated_predecessors() -> None:
    cross_namespace = list(eviction_requests())
    cross_namespace[-1] = replace(
        cross_namespace[-1],
        namespace_id="tenant-b",
    )
    cross_record = ReferenceCacheAdapter(max_entries=1).run(cross_namespace)[-1]
    assert cross_record.verdict is Verdict.INVALID
    assert "eviction_predecessor_namespace_id_mismatch" in (
        cross_record.verdict_reasons
    )

    unrelated = list(eviction_requests())
    unrelated[-1] = replace(
        unrelated[-1],
        input_token_ids=(90, 91, 92, 93),
    )
    unrelated_record = ReferenceCacheAdapter(max_entries=1).run(unrelated)[-1]
    assert unrelated_record.verdict is Verdict.INVALID
    assert "eviction_predecessor_not_reuse_producing" in (
        unrelated_record.verdict_reasons
    )


@pytest.mark.parametrize("field", ["model_id", "cache_config_digest"])
def test_eviction_proof_rejects_predecessor_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    _, records = run_audit(
        adapter=ReferenceCacheAdapter(max_entries=1),
        requests=eviction_requests(),
        cache_config=CacheConfig(
            namespace_id="synthetic",
            cache_type="token_trie",
            max_entries=1,
        ),
        output_dir=tmp_path / field,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    record = records[-1]
    proof = record.eviction_predecessor
    assert proof is not None
    changed = replace(
        proof.predecessor,
        **{field: "other-model" if field == "model_id" else "sha256:" + "0" * 64},
    )
    classified = classify_request(
        replace(
            record,
            eviction_predecessor=replace(proof, predecessor=changed),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.verdict is Verdict.INVALID
    assert any(field in reason for reason in classified.verdict_reasons)


def test_bundle_rejects_eviction_identity_forged_away_from_manifest(
    tmp_path: Path,
) -> None:
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(max_entries=1),
        requests=eviction_requests(),
        cache_config=CacheConfig(
            namespace_id="synthetic",
            cache_type="token_trie",
            max_entries=1,
        ),
        output_dir=tmp_path / "valid",
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    proof = records[-1].eviction_predecessor
    assert proof is not None
    forged_identity = replace(proof.predecessor, model_id="forged-model")
    forged_proof = replace(
        proof,
        predecessor=forged_identity,
        current=replace(proof.current, model_id="forged-model"),
    )
    forged = list(records)
    forged[-1] = replace(forged[-1], eviction_predecessor=forged_proof)
    output = tmp_path / "forged"
    write_bundle(output, manifest, forged)
    with pytest.raises(CacheAuditBundleError, match="identity mismatch"):
        verify_bundle(output)


def test_bundle_rejects_generation_before_capture_and_commit(tmp_path: Path) -> None:
    request = adversarial_requests()[:1]
    common = {
        "adapter": ReferenceCacheAdapter(),
        "requests": request,
        "cache_config": CacheConfig(
            namespace_id="synthetic",
            cache_type="token_trie",
        ),
        "backend_version": "1",
        "model_id": "synthetic-tiny-model",
        "tokenizer_id": "integer-tokenizer-v1",
    }
    with pytest.raises(CacheAuditBundleError, match="predates evidence capture"):
        run_audit(
            **common,
            output_dir=tmp_path / "capture",
            created_at="2026-01-02T00:00:00Z",
            generated_at="2026-01-01T00:00:00Z",
        )
    with pytest.raises(CacheAuditBundleError, match="predates generator commit"):
        run_audit(
            **common,
            output_dir=tmp_path / "commit",
            created_at="2000-01-01T00:00:00Z",
            generated_at="2000-01-02T00:00:00Z",
        )


def _committed_cache_manifest() -> AuditManifest:
    return AuditManifest.from_dict(
        json.loads(
            Path(
                "examples/cache-audit/reference-positive-control/audit-manifest.json"
            ).read_text(encoding="utf-8")
        )
    )


def test_repository_chronology_corroborates_available_commit() -> None:
    assert (
        _verify_manifest_chronology(
            _committed_cache_manifest(),
            repository=Path.cwd(),
        )
        == "verified"
    )


def test_repository_chronology_rejects_wrong_available_timestamp() -> None:
    manifest = _committed_cache_manifest()
    with pytest.raises(
        CacheAuditBundleError,
        match="commit timestamp does not match git",
    ):
        _verify_manifest_chronology(
            replace(manifest, generator_commit_at="2026-09-05T16:11:24Z"),
            repository=Path.cwd(),
        )


def test_repository_chronology_rejects_wrong_available_tree() -> None:
    manifest = _committed_cache_manifest()
    wrong_commit = "b84ec2d71ec9e9939be5194e5adf86e24428699c"
    timestamp = subprocess.run(
        ["git", "show", "-s", "--format=%cI", wrong_commit],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(
        CacheAuditBundleError,
        match="package tree does not match package digest",
    ):
        _verify_manifest_chronology(
            replace(
                manifest,
                generator_commit=wrong_commit,
                generator_commit_at=timestamp,
            ),
            repository=Path.cwd(),
        )


def test_repository_chronology_rejects_available_non_commit_object(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    object_id = subprocess.run(
        ["git", "-C", str(repository), "hash-object", "-w", "--stdin"],
        input="not a commit",
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    manifest = _committed_cache_manifest()
    with pytest.raises(CacheAuditBundleError, match="object is not a commit"):
        _verify_manifest_chronology(
            replace(manifest, generator_commit=object_id),
            repository=repository,
        )


def test_repository_chronology_is_unavailable_in_shallow_checkout(
    tmp_path: Path,
) -> None:
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            Path.cwd().resolve().as_uri(),
            str(shallow),
        ],
        check=True,
    )
    assert (
        _verify_manifest_chronology(
            _committed_cache_manifest(),
            repository=shallow,
        )
        == "unavailable"
    )


def test_repository_chronology_is_unavailable_without_git(tmp_path: Path) -> None:
    assert (
        _verify_manifest_chronology(
            _committed_cache_manifest(),
            repository=tmp_path,
        )
        == "unavailable"
    )


def test_portable_verifier_accepts_shallow_checkout_without_generator_object(
    tmp_path: Path,
) -> None:
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            Path.cwd().resolve().as_uri(),
            str(shallow),
        ],
        check=True,
    )
    bundle = Path("examples/cache-audit/reference-positive-control").resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(bundle / "evidence_bundle.py"),
            "verify",
            "--public-dir",
            str(bundle),
            "--package-root",
            str(shallow),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["repository_chronology_corroboration"] == (
        "unavailable"
    )


def test_portable_verifier_accepts_matching_package_without_git(
    tmp_path: Path,
) -> None:
    installed = tmp_path / "installed"
    shutil.copytree("llmtracefx", installed / "llmtracefx")
    bundle = Path("examples/cache-audit/reference-positive-control").resolve()
    result = subprocess.run(
        [
            sys.executable,
            str(bundle / "evidence_bundle.py"),
            "verify",
            "--public-dir",
            str(bundle),
            "--package-root",
            str(installed),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["repository_chronology_corroboration"] == (
        "unavailable"
    )


def test_portable_verifier_refuses_unrelated_installed_package(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=bundle,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    fake_root = tmp_path / "fake-package"
    fake_module = fake_root / "llmtracefx" / "cache_audit"
    fake_module.mkdir(parents=True)
    sentinel = tmp_path / "attacker-executed"
    (fake_root / "llmtracefx" / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    (fake_module / "bundle.py").write_text("# unrelated package\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(bundle / "evidence_bundle.py"),
            "verify",
            "--public-dir",
            str(bundle),
            "--package-root",
            str(fake_root),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "matching llmtracefx source not found" in result.stderr
    assert not sentinel.exists()


def test_portable_verifier_rejects_manifest_retarget_before_import(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:1],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=bundle,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    manifest_path = bundle / "audit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generator_package_digest"] = "sha256:" + "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    checksums = (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    updated = []
    for line in checksums:
        _, name = line.split("  ")
        digest = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        updated.append(f"{digest}  {name}")
    (bundle / "SHA256SUMS").write_text("\n".join(updated) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(bundle / "evidence_bundle.py"),
            "verify",
            "--public-dir",
            str(bundle),
            "--package-root",
            str(Path.cwd()),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "embedded trust anchor" in result.stderr


def test_documented_commands_work_from_clean_checkout() -> None:
    text = Path("docs/cache-audit.md").read_text(encoding="utf-8")
    assert "uv run llmtracefx-cache-audit compile" in text
    assert "uv run llmtracefx-cache-audit run" in text
    assert "\nllmtracefx-cache-audit " not in text
