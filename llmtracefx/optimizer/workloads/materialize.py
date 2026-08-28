"""Deterministic prompt materialization: reaching a target context tier.

This module defines, explicitly and reproducibly, how a workload's base
prompt is padded toward a target context length. It never silently
invents or truncates task content: the base prompt is always preserved
verbatim, and any padding added is deterministic filler text whose exact
composition is recorded alongside a hash of the fully materialized
prompt.

Token counts here are a documented **planning approximation**
(``APPROX_CHARS_PER_TOKEN``), not a measurement. The actual token count
for a real run is always measured by the runtime's own tokenizer at
collection time and recorded in ``ExperimentRecord.tokens`` -- this
module's ``target_context_tokens``/``approx_chars_per_token`` fields
exist so consumers can tell planned intent apart from measured fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..collectors._shared import sha256_text
from .catalog import CONTEXT_FILLER_CORPUS
from .schema import CONTEXT_TIER_TARGET_TOKENS, ContextTier, Workload

#: Conservative, fixed approximation used only to decide how much filler
#: to add when materializing a prompt for planning/matrix-generation
#: purposes. English technical text commonly tokenizes at roughly 3-4
#: characters per token across common BPE tokenizers; 4 is intentionally
#: conservative (under-pads rather than over-pads relative to most
#: tokenizers, so materialized prompts are unlikely to *exceed* the
#: target once actually tokenized).
APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class MaterializedPrompt:
    """The fully materialized prompt text plus its planning metadata."""

    workload_id: str
    workload_version: str
    context_tier: ContextTier
    target_context_tokens: int
    approx_chars_per_token: int
    filler_segments_used: int
    text: str
    prompt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "context_tier": self.context_tier.value,
            "target_context_tokens": self.target_context_tokens,
            "approx_chars_per_token": self.approx_chars_per_token,
            "filler_segments_used": self.filler_segments_used,
            "prompt_hash": self.prompt_hash,
            "prompt_char_count": len(self.text),
        }


def materialize_prompt(workload: Workload, tier: ContextTier) -> MaterializedPrompt:
    """Deterministically pad ``workload``'s base prompt toward ``tier``.

    The base prompt is always kept verbatim and never truncated -- if it
    already meets or exceeds the approximate target size, no filler is
    added. Padding, when added, is deterministic filler text appended
    after a blank-line separator, numbered from zero, so the exact
    output is reproducible from ``(workload_id, version, tier)`` alone.
    """
    target_tokens = CONTEXT_TIER_TARGET_TOKENS[tier]
    target_chars = target_tokens * APPROX_CHARS_PER_TOKEN
    base = workload.base_prompt

    if len(base) >= target_chars:
        text = base
        filler_segments = 0
    else:
        remaining = target_chars - len(base)
        filler_parts: list[str] = []
        index = 0
        while remaining > 0:
            segment = CONTEXT_FILLER_CORPUS.format(index=index) + "\n"
            filler_parts.append(segment)
            remaining -= len(segment)
            index += 1
        text = base + "\n\n" + "".join(filler_parts)
        filler_segments = index

    return MaterializedPrompt(
        workload_id=workload.workload_id,
        workload_version=workload.version,
        context_tier=tier,
        target_context_tokens=target_tokens,
        approx_chars_per_token=APPROX_CHARS_PER_TOKEN,
        filler_segments_used=filler_segments,
        text=text,
        prompt_hash=sha256_text(text),
    )
