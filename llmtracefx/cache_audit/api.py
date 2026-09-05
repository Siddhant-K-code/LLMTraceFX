"""Stable Python API for cache-audit workflows."""

from __future__ import annotations

from pathlib import Path

from .bundle import (
    read_bundle,
    sanitize_bundle_records,
    verify_bundle,
    write_bundle,
)
from .report_html import render_html
from .runner import run_audit
from .schema import AuditManifest, RequestEvidence, RequestSpec
from .workloads import adversarial_requests


def compile_audit(*, block_size: int = 4) -> tuple[RequestSpec, ...]:
    """Compile the built-in deterministic adversarial workload."""

    return adversarial_requests(block_size=block_size)


def verify_audit_bundle(bundle_dir: Path) -> dict[str, object]:
    """Verify one evidence bundle without network or model loading."""

    return verify_bundle(bundle_dir)


def render_audit_report(manifest: AuditManifest, records: list[RequestEvidence]) -> str:
    """Render deterministic, self-contained HTML from validated evidence."""

    return render_html(manifest, records)


def sanitize_audit_bundle(source: Path, destination: Path) -> dict[str, object]:
    """Create and verify a public-redacted copy of a private/synthetic bundle."""

    manifest, records = read_bundle(source)
    public_manifest, public_records = sanitize_bundle_records(manifest, records)
    write_bundle(destination, public_manifest, public_records)
    return verify_bundle(destination)


__all__ = [
    "compile_audit",
    "render_audit_report",
    "run_audit",
    "sanitize_audit_bundle",
    "verify_audit_bundle",
]
