"""High-level cache-audit execution API."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .adapters.base import CacheAuditAdapter
from .bundle import write_bundle
from .schema import (
    AuditManifest,
    CacheConfig,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
)
from .workloads import workload_digest

ADAPTER_VERSION = "1"


def _run_id(backend: str, requests: Sequence[RequestSpec], seed: int) -> str:
    material = f"{backend}:{seed}:{workload_digest(requests)}".encode("ascii")
    return "cache-audit-" + hashlib.sha256(material).hexdigest()[:16]


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
    if adapter.backend == "mlx_lm_local" and model_artifact_digest is None:
        raise ValueError("MLX audits require a model artifact digest")
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
        backend_version=backend_version,
        adapter_version=ADAPTER_VERSION,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        model_artifact_digest=model_artifact_digest,
        runtime_identity={
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
        },
        cache_config=cache_config,
        publication_mode=publication_mode,
        request_order=tuple(request.request_id for request in ordered),
        workload_digest=workload_digest(ordered),
        seed=seed,
    )
    write_bundle(output_dir, manifest, records)
    return manifest, records
