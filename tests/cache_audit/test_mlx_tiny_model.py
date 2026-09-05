"""Download-free positive control against the pinned real MLX-LM runtime."""

from __future__ import annotations

import pytest

from llmtracefx.cache_audit.adapters.mlx import (
    MLXLocalCacheAdapter,
    ProductionMLXRuntime,
)
from llmtracefx.cache_audit.schema import RequestSpec, ScenarioKind, Verdict


class _IntegerTokenizer:
    eos_token_id = 63
    bos_token = None
    chat_template = None
    clean_up_tokenization_spaces = False

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {}

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return " ".join(str(token) for token in tokens)


def _request(request_id: str, tokens: tuple[int, ...], order: int) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        scenario=ScenarioKind.IDENTICAL_PREFIX,
        pair_id="tiny-mlx",
        order=order,
        namespace_id="tiny-mlx",
        input_token_ids=tokens,
        input_token_count=len(tokens),
        output_tokens=2,
    )


def test_real_tiny_llama_reuse_preserves_output_and_prompt_work() -> None:
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    mx.random.seed(7)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=64,
            head_dim=8,
        )
    )
    mx.eval(model.parameters())
    tokenizer = TokenizerWrapper(_IntegerTokenizer(), eos_token_ids={63})
    adapter = MLXLocalCacheAdapter(
        runtime=ProductionMLXRuntime(max_cache_entries=4),
        model=model,
        tokenizer=tokenizer,
        model_key="download-free-tiny-llama",
        model_artifact_digest="sha256:" + "7" * 64,
        max_cache_entries=4,
    )

    cold, reused = adapter.run(
        [
            _request("tiny-cold", (1, 2, 3, 4), 0),
            _request("tiny-reused", (1, 2, 3, 4, 5), 1),
        ]
    )

    assert cold.verdict is Verdict.VERIFIED_MISS
    assert reused.verdict is Verdict.PARTIAL_REUSE
    assert reused.reuse.policy_reusable_tokens.value == 4
    assert reused.reuse.engine_cached_tokens.value == 4
    assert reused.reuse.observed_prompt_tokens.value == 1
    assert reused.output.token_identity.value is True
    assert reused.output.correctness.value is True
