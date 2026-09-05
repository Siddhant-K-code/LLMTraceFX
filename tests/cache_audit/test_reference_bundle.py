from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llmtracefx.cache_audit.adapters.reference import (
    ReferenceCacheAdapter,
    SyntheticCacheEngine,
)
from llmtracefx.cache_audit.api import sanitize_audit_bundle
from llmtracefx.cache_audit.bundle import (
    CacheAuditBundleError,
    read_bundle,
    verify_bundle,
    write_bundle,
)
from llmtracefx.cache_audit.expected import MLXCacheOracle
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import (
    CacheConfig,
    EligibilityStatus,
    PublicationMode,
    RequestSpec,
    ScenarioKind,
    Verdict,
)
from llmtracefx.cache_audit.workloads import (
    adversarial_requests,
    eviction_requests,
    gated_extension_requests,
)


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
