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
from .schema import (
    CONTEXT_TIER_TARGET_TOKENS,
    ContextTier,
    PaddingPlacement,
    Workload,
)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterializedPrompt:
        """Reconstruct planning metadata from a persisted matrix manifest.

        ``to_dict`` deliberately never persists ``text`` (the prompt body
        lives in its own ``prompts/<id>-<tier>.txt`` file so it is not
        duplicated into ``manifest.json``), so ``text`` is always empty
        here. This is intentional: callers loading a manifest back from
        disk must read the prompt file at the owning entry's
        ``prompt_path`` and verify its hash against ``prompt_hash``
        rather than trusting an in-memory copy, which is exactly the
        prompt-integrity check the verification pipeline performs before
        executing any row.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"MaterializedPrompt must be an object, got {type(data).__name__}"
            )
        try:
            return cls(
                workload_id=data["workload_id"],
                workload_version=data["workload_version"],
                context_tier=ContextTier(data["context_tier"]),
                target_context_tokens=int(data["target_context_tokens"]),
                approx_chars_per_token=int(data["approx_chars_per_token"]),
                filler_segments_used=int(data["filler_segments_used"]),
                text="",
                prompt_hash=data["prompt_hash"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid MaterializedPrompt: {exc}") from exc


def _build_filler(remaining_chars: int) -> tuple[str, int]:
    """Deterministically build filler text totalling >= ``remaining_chars``.

    Returns the filler text and the number of segments used, numbered
    from zero so the output is reproducible.
    """
    filler_parts: list[str] = []
    index = 0
    remaining = remaining_chars
    while remaining > 0:
        segment = CONTEXT_FILLER_CORPUS.format(index=index) + "\n"
        filler_parts.append(segment)
        remaining -= len(segment)
        index += 1
    return "".join(filler_parts), index


def materialize_prompt(workload: Workload, tier: ContextTier) -> MaterializedPrompt:
    """Deterministically pad ``workload``'s base prompt toward ``tier``.

    The base prompt is always kept verbatim and never truncated -- if it
    already meets or exceeds the approximate target size, no filler is
    added. Padding, when added, is deterministic filler text numbered
    from zero, so the exact output is reproducible from
    ``(workload_id, version, tier)`` alone.

    Placement depends on ``workload.padding_placement``:

    * ``APPEND_AFTER_BASE_PROMPT``: filler is appended after the full
      base prompt. Correct for workloads whose base prompt is already a
      complete, self-contained instruction (structured JSON, prose
      reasoning).
    * ``BEFORE_CONTINUATION_STUB``: filler is inserted between the base
      prompt's instructions and its fixed ``continuation_stub``, so the
      stub -- e.g. an open function signature/docstring for code
      completion -- remains the exact trailing content the model must
      continue from. Appending filler after the stub would instead
      leave irrelevant filler as the last thing the model sees,
      corrupting the completion point.
    """
    target_tokens = CONTEXT_TIER_TARGET_TOKENS[tier]
    target_chars = target_tokens * APPROX_CHARS_PER_TOKEN
    base = workload.base_prompt

    if len(base) >= target_chars:
        text = base
        filler_segments = 0
    elif workload.padding_placement is PaddingPlacement.BEFORE_CONTINUATION_STUB:
        stub = workload.continuation_stub
        prefix = base[: -len(stub)]
        filler_text, filler_segments = _build_filler(target_chars - len(base))
        text = f"{prefix}{filler_text}\n{stub}"
    else:
        filler_text, filler_segments = _build_filler(target_chars - len(base))
        text = base + "\n\n" + filler_text

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
