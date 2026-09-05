"""Shared adapter boundary for cache-audit runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..schema import RequestEvidence, RequestSpec


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


class CacheAuditAdapter(Protocol):
    @property
    def backend(self) -> str: ...

    def capabilities(self) -> CacheAuditCapability: ...

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]: ...
