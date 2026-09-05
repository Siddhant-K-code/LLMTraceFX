"""High-level cache-audit execution API."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .adapters.base import CacheAuditAdapter
from .bundle import package_source_digest, write_bundle
from .schema import (
    AuditManifest,
    CacheConfig,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
)
from .workloads import workload_digest

ADAPTER_VERSION = "2"


def _run_id(backend: str, requests: Sequence[RequestSpec], seed: int) -> str:
    material = f"{backend}:{seed}:{workload_digest(requests)}".encode("ascii")
    return "cache-audit-" + hashlib.sha256(material).hexdigest()[:16]


def source_commit() -> str | None:
    """Return the repository commit containing this package, when available."""

    repository = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


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
    manifest = AuditManifest(
        run_id=_run_id(adapter.backend, ordered, seed),
        created_at=created_at
        or datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
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
        generator_commit=source_commit(),
        generator_package_digest=package_source_digest(),
    )
    write_bundle(output_dir, manifest, records)
    return manifest, records
