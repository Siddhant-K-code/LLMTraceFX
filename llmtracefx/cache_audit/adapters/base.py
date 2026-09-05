"""Shared adapter boundary for cache-audit runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..schema import CacheConfig, RequestEvidence, RequestSpec


@dataclass(frozen=True)
class CacheAuditCapability:
    backend: str
    supported: bool
    reasons: tuple[str, ...] = ()
    observable_facts: tuple[str, ...] = ()
    unavailable_facts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "supported": self.supported,
            "reasons": list(self.reasons),
            "observable_facts": list(self.observable_facts),
            "unavailable_facts": list(self.unavailable_facts),
        }


@dataclass(frozen=True)
class AdapterAuditIdentity:
    """Adapter-owned manifest identity and cache semantics."""

    backend_version: str
    runtime_identity: dict[str, str]
    model_artifact_digest: str | None
    cache_type: str
    max_entries: int | None
    max_bytes: int | None
    hash_algorithm: str | None = None
    hash_block_size: int | None = None
    physical_block_sizes: tuple[int, ...] = ()
    fine_grained_hits: bool = False

    def authoritative_cache_config(self, requested: CacheConfig) -> CacheConfig:
        expected = {
            "cache_type": self.cache_type,
            "hash_algorithm": self.hash_algorithm,
            "hash_block_size": self.hash_block_size,
            "physical_block_sizes": self.physical_block_sizes,
            "fine_grained_hits": self.fine_grained_hits,
        }
        for field, value in expected.items():
            if getattr(requested, field) != value:
                raise ValueError(
                    f"caller cache configuration {field} does not match adapter: "
                    f"{getattr(requested, field)!r} != {value!r}"
                )
        for field, value in (
            ("max_entries", self.max_entries),
            ("max_bytes", self.max_bytes),
        ):
            supplied = getattr(requested, field)
            if supplied is not None and supplied != value:
                raise ValueError(
                    f"caller cache limit {field} does not match adapter: "
                    f"{supplied!r} != {value!r}"
                )
        return CacheConfig(
            namespace_id=requested.namespace_id,
            cache_type=self.cache_type,
            max_entries=self.max_entries,
            max_bytes=self.max_bytes,
            hash_algorithm=self.hash_algorithm,
            hash_block_size=self.hash_block_size,
            physical_block_sizes=self.physical_block_sizes,
            fine_grained_hits=self.fine_grained_hits,
            cache_salt_relationship=requested.cache_salt_relationship,
        )


class CacheAuditAdapter(Protocol):
    @property
    def backend(self) -> str: ...

    def capabilities(self) -> CacheAuditCapability: ...

    def audit_identity(self) -> AdapterAuditIdentity: ...

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]: ...
