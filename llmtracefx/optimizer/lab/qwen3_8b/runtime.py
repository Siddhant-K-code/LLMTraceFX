"""MLX-LM chat-template runtime adapter for the Qwen3-8B control.

``collectors.mlx.MLXLMRuntime`` tokenizes a prompt verbatim; it applies
no chat template because most of its existing callers (autopsy probes,
tuning fixtures) intentionally exercise the tokenizer directly. This
control instead needs every prompt to go through the checkpoint's own
mlx-lm chat template with ``enable_thinking=False`` (as pinned by
``manifest.generation.enable_thinking``), exactly like a real chat
completion. Rather than edit the shared, already-tested
``MLXLMRuntime``, this module wraps one by composition and overrides
only ``encode``, so every other collector-facing behavior (model
loading, seeding, memory snapshots, generation) is the identical,
already-covered code path the rest of the project relies on.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...collectors.mlx import (
    MLXCollectorError,
    MLXGenerationResponse,
    MLXLMRuntime,
    MLXMemorySnapshot,
)


class Qwen3ChatMLXLMRuntime:
    """Adapts ``MLXLMRuntime`` to tokenize through the checkpoint's own
    mlx-lm chat template instead of the raw prompt text."""

    def __init__(
        self,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: bool = False,
    ) -> None:
        self._delegate = MLXLMRuntime(temperature=temperature, top_p=top_p)
        self._enable_thinking = enable_thinking

    @property
    def mlx_version(self) -> str | None:
        return self._delegate.mlx_version

    @property
    def mlx_lm_version(self) -> str | None:
        return self._delegate.mlx_lm_version

    @property
    def runtime_name(self) -> str:
        return self._delegate.runtime_name

    @property
    def runtime_version(self) -> str | None:
        return self._delegate.runtime_version

    def load_model(self, path: Path) -> tuple[Any, Any]:
        return self._delegate.load_model(path)

    def encode(self, tokenizer: Any, prompt: str) -> list[int]:
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            raise MLXCollectorError(
                "this mlx-lm tokenizer does not expose apply_chat_template; "
                "the Qwen3-8B control requires a chat-template-capable "
                "checkpoint"
            )
        encoded = apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        if not isinstance(encoded, list):
            raise MLXCollectorError(
                "mlx-lm chat template did not return a token-id list for a "
                "text-only run"
            )
        return [int(token) for token in encoded]

    def seed(self, seed: int) -> None:
        self._delegate.seed(seed)

    def synchronize(self) -> None:
        self._delegate.synchronize()

    def reset_peak_memory(self) -> None:
        self._delegate.reset_peak_memory()

    def memory_snapshot(self) -> MLXMemorySnapshot:
        return self._delegate.memory_snapshot()

    def accelerator_name(self) -> str | None:
        return self._delegate.accelerator_name()

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        draft_model: Any | None,
        num_draft_tokens: int,
    ) -> Iterator[MLXGenerationResponse]:
        return self._delegate.stream_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=max_tokens,
            draft_model=draft_model,
            num_draft_tokens=num_draft_tokens,
        )


__all__ = ["Qwen3ChatMLXLMRuntime"]
