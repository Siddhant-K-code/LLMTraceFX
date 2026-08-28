"""Comparable-group and candidate-identity keys for the tuner.

Two runs are only ever ranked against each other if they share a
``GroupKey``: the same workload/version, context tier, model id/family,
accelerator, runtime/backend, and workload prompt hash. Everything else
(quantization, speculative settings, model/tokenizer revisions, seed, the
collector's own config hash) distinguishes different *candidates* within
one group -- see ``CandidateKey``. Grouping and identity are computed only
from already-validated ``RowVerification``/``ExperimentRecord`` pairs (see
``loader.py``); nothing here re-derives or guesses a value that was not
actually recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schema import ExperimentRecord
from ..workloads.verify import RowVerification


@dataclass(frozen=True)
class GroupKey:
    """Identifies one set of directly comparable candidate configurations."""

    workload_id: str
    workload_version: str
    context_tier: str
    model_id: str
    model_family: str | None
    accelerator: str | None
    runtime_name: str
    runtime_backend: str | None
    workload_prompt_hash: str

    def label(self) -> str:
        family = f"/{self.model_family}" if self.model_family else ""
        return (
            f"{self.workload_id}@v{self.workload_version} [{self.context_tier}] "
            f"model={self.model_id}{family} "
            f"accelerator={self.accelerator or 'unknown'} "
            f"runtime={self.runtime_name}/{self.runtime_backend or 'unknown'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "context_tier": self.context_tier,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "accelerator": self.accelerator,
            "runtime_name": self.runtime_name,
            "runtime_backend": self.runtime_backend,
            "workload_prompt_hash": self.workload_prompt_hash,
        }

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.workload_id,
            self.workload_version,
            self.context_tier,
            self.model_id,
            self.model_family or "",
            self.accelerator or "",
            self.runtime_name,
            self.runtime_backend or "",
            self.workload_prompt_hash,
        )


@dataclass(frozen=True)
class CandidateKey:
    """Identifies one distinct inference configuration within a group.

    Deliberately conservative: two runs are only ever treated as the same
    candidate (i.e. their measurements are averaged together) when every
    one of these fields matches exactly. Seed is always included (never
    only "when relevant") so this module never silently merges evidence
    collected under different seeds.
    """

    decode_mode: str
    runtime_version: str | None
    quantization: str | None
    model_revision: str | None
    tokenizer_revision: str | None
    speculative_enabled: bool
    speculative_method: str | None
    speculative_configured_depth: int | None
    seed: int | None
    config_hash: str | None

    def label(self) -> str:
        spec = (
            f"{self.speculative_method or 'none'}"
            f"(depth={self.speculative_configured_depth})"
            if self.speculative_enabled
            else "none"
        )
        return (
            f"{self.decode_mode} quant={self.quantization or 'unspecified'} "
            f"speculative={spec} seed={self.seed if self.seed is not None else 'unspecified'} "
            f"model_rev={self.model_revision or 'unspecified'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decode_mode": self.decode_mode,
            "runtime_version": self.runtime_version,
            "quantization": self.quantization,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "speculative_enabled": self.speculative_enabled,
            "speculative_method": self.speculative_method,
            "speculative_configured_depth": self.speculative_configured_depth,
            "seed": self.seed,
            "config_hash": self.config_hash,
        }

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.decode_mode,
            self.runtime_version or "",
            self.quantization or "",
            self.model_revision or "",
            self.tokenizer_revision or "",
            self.speculative_enabled,
            self.speculative_method or "",
            (
                self.speculative_configured_depth
                if self.speculative_configured_depth is not None
                else -1
            ),
            self.seed if self.seed is not None else -1,
            self.config_hash or "",
        )


def group_key_for(verification: RowVerification, record: ExperimentRecord) -> GroupKey:
    """Build the comparable-group key for one verified run.

    ``verification.verified_prompt_hash`` (the hash the verify pipeline
    actually measured from the executed prompt file, not merely the hash
    the matrix expected) is used for ``workload_prompt_hash``; callers must
    have already confirmed this matches ``record.command.workload_hash``
    (see ``loader.py``) before trusting this key.
    """
    if verification.verified_prompt_hash is None:
        raise ValueError(
            "cannot build a GroupKey without a verified_prompt_hash "
            f"(run_id={verification.run_id!r})"
        )
    return GroupKey(
        workload_id=verification.workload_id,
        workload_version=verification.workload_version,
        context_tier=verification.context_tier,
        model_id=record.model.model_id,
        model_family=record.model.model_family,
        accelerator=record.platform.accelerator,
        runtime_name=record.runtime.name,
        runtime_backend=record.runtime.backend,
        workload_prompt_hash=verification.verified_prompt_hash,
    )


def candidate_key_for(
    verification: RowVerification, record: ExperimentRecord
) -> CandidateKey:
    """Build the within-group candidate identity key for one verified run."""
    return CandidateKey(
        decode_mode=verification.decode_mode,
        runtime_version=record.runtime.version,
        quantization=record.model.quantization,
        model_revision=record.model.model_revision,
        tokenizer_revision=record.model.tokenizer_revision,
        speculative_enabled=record.speculative.enabled,
        speculative_method=record.speculative.method,
        speculative_configured_depth=record.speculative.configured_depth,
        seed=record.repetition.seed,
        config_hash=record.command.config_hash,
    )
