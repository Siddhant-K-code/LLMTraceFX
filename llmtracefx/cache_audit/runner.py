"""High-level cache-audit execution API."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from .adapters.base import CacheAuditAdapter
from .bundle import package_source_digest, write_bundle
from .expected import longest_common_prefix
from .schema import (
    AuditManifest,
    CacheConfig,
    EvictionPredecessorProof,
    PublicationMode,
    RequestCacheIdentity,
    RequestEvidence,
    RequestSpec,
)
from .verdicts import classify_request
from .workloads import workload_digest

ADAPTER_VERSION = "2"


def _run_id(backend: str, requests: Sequence[RequestSpec], seed: int) -> str:
    material = f"{backend}:{seed}:{workload_digest(requests)}".encode("ascii")
    return "cache-audit-" + hashlib.sha256(material).hexdigest()[:16]


def source_commit() -> tuple[str | None, str | None]:
    """Return the repository commit and its timestamp, when available."""

    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository), "show", "-s", "--format=%H%n%cI", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2 or len(lines[0]) != 40:
        return None, None
    return lines[0], lines[1]


def _cache_config_digest(config: CacheConfig) -> str:
    encoded = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _request_cache_identity(
    *,
    spec: RequestSpec,
    backend: str,
    model_id: str,
    tokenizer_id: str,
    model_artifact_digest: str | None,
    cache_config_digest: str,
) -> RequestCacheIdentity | None:
    if spec.input_token_ids is None:
        return None
    return RequestCacheIdentity(
        backend=backend,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model_artifact_digest=model_artifact_digest,
        cache_config_digest=cache_config_digest,
        namespace_id=spec.namespace_id,
        input_token_ids=spec.input_token_ids,
    )


def _bind_eviction_predecessors(
    records: Sequence[RequestEvidence],
    *,
    backend: str,
    model_id: str,
    tokenizer_id: str,
    model_artifact_digest: str | None,
    cache_config: CacheConfig,
) -> list[RequestEvidence]:
    cache_digest = _cache_config_digest(cache_config)
    prior: dict[str, RequestEvidence] = {}
    bound: list[RequestEvidence] = []
    for record in records:
        proof = None
        if record.reuse.eviction_observed.value is True:
            predecessors = [
                prior[request_id]
                for request_id in record.spec.expected_predecessors
                if request_id in prior
            ]
            current_identity = _request_cache_identity(
                spec=record.spec,
                backend=backend,
                model_id=model_id,
                tokenizer_id=tokenizer_id,
                model_artifact_digest=model_artifact_digest,
                cache_config_digest=cache_digest,
            )
            if len(predecessors) == 1 and current_identity is not None:
                predecessor = predecessors[0]
                predecessor_identity = _request_cache_identity(
                    spec=predecessor.spec,
                    backend=backend,
                    model_id=model_id,
                    tokenizer_id=tokenizer_id,
                    model_artifact_digest=model_artifact_digest,
                    cache_config_digest=cache_digest,
                )
                if predecessor_identity is not None:
                    reusable = min(
                        max(0, len(current_identity.input_token_ids) - 1),
                        longest_common_prefix(
                            predecessor_identity.input_token_ids,
                            current_identity.input_token_ids,
                        ),
                    )
                    proof = EvictionPredecessorProof(
                        predecessor_request_id=predecessor.spec.request_id,
                        predecessor=predecessor_identity,
                        current=current_identity,
                        reusable_prefix_tokens=reusable,
                    )
        rebound = classify_request(
            replace(
                record,
                eviction_predecessor=proof,
                verdict=None,
                verdict_reasons=(),
            )
        )
        bound.append(rebound)
        prior[record.spec.request_id] = rebound
    return bound


def run_audit(
    *,
    adapter: CacheAuditAdapter,
    requests: Sequence[RequestSpec],
    cache_config: CacheConfig,
    output_dir: Path,
    backend_version: str,
    model_id: str,
    tokenizer_id: str,
    model_artifact_digest: str | None = None,
    publication_mode: PublicationMode = PublicationMode.PRIVATE,
    seed: int = 0,
    created_at: str | None = None,
    generated_at: str | None = None,
) -> tuple[AuditManifest, list[RequestEvidence]]:
    """Execute an adapter and persist a complete deterministic evidence bundle."""

    capability = adapter.capabilities()
    if not capability.supported:
        raise RuntimeError(
            f"{adapter.backend} cannot run: {', '.join(capability.reasons)}"
        )
    if publication_mode is PublicationMode.PUBLIC_REDACTED:
        raise ValueError(
            "direct public_redacted runs are refused; create a private bundle and "
            "sanitize it after verification"
        )
    identity = adapter.audit_identity()
    if backend_version != identity.backend_version:
        raise ValueError(
            "caller backend version does not match adapter: "
            f"{backend_version!r} != {identity.backend_version!r}"
        )
    if model_artifact_digest != identity.model_artifact_digest:
        raise ValueError(
            "caller model artifact digest does not match adapter: "
            f"{model_artifact_digest!r} != {identity.model_artifact_digest!r}"
        )
    authoritative_cache_config = identity.authoritative_cache_config(cache_config)
    ordered = tuple(sorted(requests, key=lambda request: request.order))
    if tuple(request.order for request in ordered) != tuple(range(len(ordered))):
        raise ValueError("request order must be contiguous and zero-based")
    records = adapter.run(ordered)
    if tuple(record.spec for record in records) != ordered:
        raise RuntimeError(
            "adapter returned request specifications that differ from input"
        )
    records = _bind_eviction_predecessors(
        records,
        backend=adapter.backend,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model_artifact_digest=identity.model_artifact_digest,
        cache_config=authoritative_cache_config,
    )
    now = (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    generator_commit, generator_commit_at = source_commit()
    manifest = AuditManifest(
        run_id=_run_id(adapter.backend, ordered, seed),
        created_at=created_at or now,
        generated_at=generated_at or now,
        backend=adapter.backend,
        backend_version=identity.backend_version,
        adapter_version=ADAPTER_VERSION,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model_artifact_digest=identity.model_artifact_digest,
        runtime_identity=identity.runtime_identity,
        cache_config=authoritative_cache_config,
        publication_mode=publication_mode,
        request_order=tuple(request.request_id for request in ordered),
        workload_digest=workload_digest(ordered),
        seed=seed,
        generator_commit=generator_commit,
        generator_commit_at=generator_commit_at,
        generator_package_digest=package_source_digest(),
    )
    write_bundle(output_dir, manifest, records)
    return manifest, records
