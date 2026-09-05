"""Evidence-first KV and prompt-cache auditing."""

from .api import (
    compile_audit,
    render_audit_report,
    run_audit,
    sanitize_audit_bundle,
    verify_audit_bundle,
)
from .expected import (
    MLXCacheOracle,
    ReuseExpectation,
    VLLMReuseConfig,
    expected_vllm_reuse,
    longest_common_prefix,
)
from .schema import (
    CACHE_AUDIT_SCHEMA_VERSION,
    AuditManifest,
    CacheConfig,
    CacheEventRecord,
    CacheStateSnapshot,
    CostEvidence,
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
    ScenarioKind,
    Verdict,
)
from .verdicts import classify_request

__all__ = [
    "CACHE_AUDIT_SCHEMA_VERSION",
    "AuditManifest",
    "CacheEventRecord",
    "CacheConfig",
    "CacheStateSnapshot",
    "CostEvidence",
    "EvidenceBasis",
    "EvidenceFact",
    "Limitation",
    "MLXCacheOracle",
    "PublicationMode",
    "RequestEvidence",
    "RequestSpec",
    "ReuseExpectation",
    "ScenarioKind",
    "VLLMReuseConfig",
    "Verdict",
    "classify_request",
    "compile_audit",
    "expected_vllm_reuse",
    "longest_common_prefix",
    "render_audit_report",
    "run_audit",
    "sanitize_audit_bundle",
    "verify_audit_bundle",
]
