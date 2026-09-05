"""Tests for the MLX-LM local cache adapter, driven entirely by fakes.

Nothing here imports ``mlx``/``mlx_lm``: :class:`FakeMLXRuntime` implements
:class:`~llmtracefx.cache_audit.adapters.mlx.MLXCacheRuntime` in pure Python,
internally mirroring ``LRUPromptCache`` via the same
:class:`~llmtracefx.cache_audit.expected.MLXCacheOracle` the adapter uses
independently, so tests never need a real model download or Apple Silicon.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from llmtracefx.cache_audit.adapters.mlx import (
    REQUIRED_MLX_LM_VERSION,
    REQUIRED_MLX_VERSION,
    REQUIRED_SYMBOLS,
    MLXCacheAdapterError,
    MLXGenerationStep,
    MLXLocalCacheAdapter,
    SavedCacheIdentity,
    _validate_local_model_path,
    check_mlx_capabilities,
    verify_saved_cache_sidecar,
    write_saved_cache_sidecar,
)
from llmtracefx.cache_audit.expected import MLXCacheOracle
from llmtracefx.cache_audit.schema import RequestSpec, ScenarioKind, Verdict
from llmtracefx.optimizer.schema import MetricProvenance

_BYTES_PER_TOKEN = 128
_MODEL_ARTIFACT_DIGEST = "sha256:" + "1" * 64
_CACHE_PAYLOAD_DIGEST = "sha256:" + "2" * 64


@dataclass
class _FakeCacheHandle:
    resident_tokens: list[int] = field(default_factory=list)


def _hash_token(seed_tokens: tuple[int, ...], index: int, corrupt: bool) -> int:
    text = ",".join(str(token) for token in seed_tokens) + f"|{index}|{corrupt}"
    digest = hashlib.sha256(text.encode("ascii")).digest()
    return int.from_bytes(digest[:2], "big") % 50_000


class FakeMLXRuntime:
    """In-memory stand-in for :class:`ProductionMLXRuntime`.

    ``fetch``/``insert`` are driven by a private :class:`MLXCacheOracle`
    instance so the fake reproduces the real MLX-LM N-1 trim policy "for
    free" instead of hand-coding cache splits per test. ``generate`` derives
    deterministic output tokens from ``resident_tokens_before + prompt``: a
    cache-assisted call and a fresh-cache baseline call are guaranteed to
    agree by construction (``resident_before + rest`` always reconstructs
    the full original prompt), unless ``corrupt_when_cached`` is set, which
    only perturbs the hash when the call actually reused a non-empty
    resident prefix -- giving a controllable output-mismatch scenario.
    """

    def __init__(
        self,
        *,
        platform_system: str = "Darwin",
        platform_machine: str = "arm64",
        mlx_version: str | None = REQUIRED_MLX_VERSION,
        mlx_lm_version: str | None = REQUIRED_MLX_LM_VERSION,
        missing_symbols: tuple[str, ...] = (),
        corrupt_when_cached: bool = False,
        recomputed_extra_when_cached: int = 0,
        trimmable: bool = True,
        max_entries: int = 10,
        max_bytes: int = 1 << 40,
    ) -> None:
        self._platform_system = platform_system
        self._platform_machine = platform_machine
        self._mlx_version = mlx_version
        self._mlx_lm_version = mlx_lm_version
        self._missing_symbols = missing_symbols
        self._corrupt_when_cached = corrupt_when_cached
        self._recomputed_extra_when_cached = recomputed_extra_when_cached
        self._trimmable = trimmable
        self._oracle = MLXCacheOracle(max_entries=max_entries, max_bytes=max_bytes)
        self._insert_counter = 0
        self._active_bytes = 0
        self._peak_bytes = 0
        self.synchronize_calls = 0
        self.reset_peak_calls = 0

    # -- capability surface --------------------------------------------
    @property
    def platform_system(self) -> str:
        return self._platform_system

    @property
    def platform_machine(self) -> str:
        return self._platform_machine

    @property
    def mlx_version(self) -> str | None:
        return self._mlx_version

    @property
    def mlx_lm_version(self) -> str | None:
        return self._mlx_lm_version

    def missing_required_symbols(self) -> tuple[str, ...]:
        return self._missing_symbols

    def load_model(self, path: Path) -> tuple[Any, Any, Hashable]:
        return f"model:{path}", f"tokenizer:{path}", str(path)

    # -- memory/synchronization -------------------------------------------
    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def reset_peak_memory(self) -> None:
        self.reset_peak_calls += 1
        self._peak_bytes = self._active_bytes

    def active_memory(self) -> int:
        return self._active_bytes

    def peak_memory(self) -> int:
        return self._peak_bytes

    def cache_memory(self) -> int:
        return self._active_bytes

    # -- cache/prefix-trie surface ----------------------------------------
    def make_cache(self, model: object) -> _FakeCacheHandle:
        return _FakeCacheHandle()

    def fetch(
        self, model_key: Hashable, tokens: Sequence[int]
    ) -> tuple[_FakeCacheHandle | None, tuple[int, ...]]:
        expectation = self._oracle.lookup(model_key, "_runtime_", tokens)
        request = tuple(tokens)
        if expectation.matched_entry_id is None:
            return None, request
        reusable = expectation.policy_reusable_tokens
        resident = request[:reusable]
        rest = request[reusable:]
        return _FakeCacheHandle(resident_tokens=list(resident)), rest

    def insert(
        self, model_key: Hashable, tokens: Sequence[int], cache: _FakeCacheHandle
    ) -> None:
        self._insert_counter += 1
        nbytes = len(tuple(tokens)) * _BYTES_PER_TOKEN
        self._oracle.insert(
            entry_id=f"engine-{self._insert_counter}",
            model_key=model_key,
            namespace_id="_runtime_",
            tokens=tokens,
            nbytes=nbytes,
            trimmable=self._trimmable,
        )
        self._active_bytes += nbytes
        self._peak_bytes = max(self._peak_bytes, self._active_bytes)

    def cache_nbytes(self, cache: _FakeCacheHandle) -> int:
        return len(cache.resident_tokens) * _BYTES_PER_TOKEN

    def cache_len(self, cache: _FakeCacheHandle) -> int:
        return len(cache.resident_tokens)

    def cache_can_trim(self, cache: _FakeCacheHandle) -> bool:
        return self._trimmable

    def cache_classes(self, cache: _FakeCacheHandle) -> tuple[str, ...]:
        return ("FakeKVCache",)

    def generate(
        self,
        model: object,
        tokenizer: object,
        cache: _FakeCacheHandle,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        prompt_progress_callback: Callable[[int, int], None],
    ) -> Iterator[MLXGenerationStep]:
        resident_before = tuple(cache.resident_tokens)
        prompt = tuple(prompt_tokens)
        if prompt:
            observed = len(prompt)
            if resident_before:
                observed += self._recomputed_extra_when_cached
            prompt_progress_callback(observed, observed)
        cache.resident_tokens.extend(prompt)
        apply_corruption = self._corrupt_when_cached and len(resident_before) > 0
        seed = resident_before + prompt
        for index in range(max_tokens):
            token = _hash_token(seed, index, apply_corruption)
            cache.resident_tokens.append(token)
            finish_reason = "stop" if index == max_tokens - 1 else None
            yield MLXGenerationStep(token=token, finish_reason=finish_reason)


def _spec(
    request_id: str,
    tokens: tuple[int, ...],
    *,
    order: int,
    scenario: ScenarioKind = ScenarioKind.COLD,
    output_tokens: int = 2,
    mutation_position: int | None = None,
) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        scenario=scenario,
        order=order,
        input_token_ids=tokens,
        input_token_count=len(tokens),
        output_tokens=output_tokens,
        mutation_position=mutation_position,
    )


def _adapter(runtime: FakeMLXRuntime | None = None) -> MLXLocalCacheAdapter:
    return MLXLocalCacheAdapter(
        runtime=runtime if runtime is not None else FakeMLXRuntime(),
        model="fake-model",
        tokenizer="fake-tokenizer",
        model_key="fake-model-key",
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
    )


# --- capability checks ---------------------------------------------------


def test_default_fake_runtime_is_fully_supported() -> None:
    capability = check_mlx_capabilities(FakeMLXRuntime())
    assert capability.supported is True
    assert capability.reasons == ()
    assert "engine_cached_tokens" in capability.observable_facts


def test_capability_check_reports_platform_system_reason() -> None:
    capability = check_mlx_capabilities(FakeMLXRuntime(platform_system="Linux"))
    assert capability.supported is False
    assert "platform_system_not_darwin" in capability.reasons


def test_capability_check_reports_platform_machine_reason() -> None:
    capability = check_mlx_capabilities(FakeMLXRuntime(platform_machine="x86_64"))
    assert capability.supported is False
    assert "platform_machine_not_arm64" in capability.reasons


def test_capability_check_reports_version_mismatch_reasons() -> None:
    capability = check_mlx_capabilities(
        FakeMLXRuntime(mlx_version="0.1.0", mlx_lm_version=None)
    )
    assert capability.supported is False
    assert any(
        reason.startswith("mlx_version_mismatch:") for reason in capability.reasons
    )
    assert any(
        reason.startswith("mlx_lm_version_mismatch:") for reason in capability.reasons
    )


def test_capability_check_reports_missing_symbol_reasons() -> None:
    capability = check_mlx_capabilities(
        FakeMLXRuntime(missing_symbols=("mlx_lm.stream_generate",))
    )
    assert capability.supported is False
    assert "missing_symbol:mlx_lm.stream_generate" in capability.reasons


def test_capability_check_reasons_are_ordered_and_never_short_circuit() -> None:
    capability = check_mlx_capabilities(
        FakeMLXRuntime(
            platform_system="Linux",
            platform_machine="x86_64",
            mlx_version="0.0.0",
            mlx_lm_version="0.0.0",
            missing_symbols=REQUIRED_SYMBOLS,
        )
    )
    expected = (
        "platform_system_not_darwin",
        "platform_machine_not_arm64",
        f"mlx_version_mismatch:required={REQUIRED_MLX_VERSION}:installed='0.0.0'",
        f"mlx_lm_version_mismatch:required={REQUIRED_MLX_LM_VERSION}:installed='0.0.0'",
        *(f"missing_symbol:{symbol}" for symbol in REQUIRED_SYMBOLS),
    )
    assert capability.reasons == expected
    assert capability.observable_facts == ()


def test_run_raises_when_capabilities_unsupported() -> None:
    adapter = _adapter(FakeMLXRuntime(platform_system="Linux"))
    with pytest.raises(MLXCacheAdapterError):
        adapter.run([_spec("cold", (1, 2, 3), order=0)])


# --- constructor / model-path validation ----------------------------------


def test_constructor_requires_either_loaded_model_or_model_path() -> None:
    with pytest.raises(MLXCacheAdapterError):
        MLXLocalCacheAdapter(runtime=FakeMLXRuntime())


def test_constructor_rejects_both_loaded_model_and_model_path(tmp_path: Path) -> None:
    with pytest.raises(MLXCacheAdapterError):
        MLXLocalCacheAdapter(
            runtime=FakeMLXRuntime(),
            model="m",
            tokenizer="t",
            model_key="k",
            model_path=tmp_path,
        )


def test_validate_local_model_path_rejects_url() -> None:
    with pytest.raises(MLXCacheAdapterError, match="URL"):
        _validate_local_model_path("https://huggingface.co/org/model")


def test_validate_local_model_path_rejects_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(MLXCacheAdapterError, match="does not exist"):
        _validate_local_model_path(tmp_path / "does-not-exist")


def test_validate_local_model_path_accepts_existing_directory(tmp_path: Path) -> None:
    assert _validate_local_model_path(tmp_path) == tmp_path


def test_adapter_loads_local_model_path_without_fetching(tmp_path: Path) -> None:
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    adapter = MLXLocalCacheAdapter(runtime=FakeMLXRuntime(), model_path=model_dir)
    assert adapter.model_key == str(model_dir)


def test_adapter_rejects_url_model_path() -> None:
    with pytest.raises(MLXCacheAdapterError):
        MLXLocalCacheAdapter(
            runtime=FakeMLXRuntime(), model_path="https://huggingface.co/org/model"
        )


# --- reuse scenarios -------------------------------------------------------


def test_cold_request_is_verified_miss() -> None:
    adapter = _adapter()
    records = adapter.run([_spec("cold", (11, 12, 13, 14, 15), order=0)])
    assert len(records) == 1
    record = records[0]
    assert record.verdict is Verdict.VERIFIED_MISS
    assert record.reuse.engine_cached_tokens.value == 0
    assert record.reuse.policy_reusable_tokens.value == 0
    assert record.output.output_token_ids is not None
    assert len(record.output.output_token_ids) == 2


def test_identical_prefix_hits_full_mlx_n_minus_one_policy() -> None:
    adapter = _adapter()
    prompt = (11, 12, 13, 14, 15)
    first = adapter.run([_spec("cold", prompt, order=0)])[0]
    assert first.verdict is Verdict.VERIFIED_MISS

    second_spec = _spec(
        "identical", prompt, order=1, scenario=ScenarioKind.IDENTICAL_PREFIX
    )
    second = adapter.run([second_spec])[0]

    # The stored entry is prompt+output (7 tokens); MLX-LM's N-1 trim policy
    # reserves exactly one token, so only 4 of 5 prompt tokens are reused --
    # yet the full common prefix still covers the entire request, so this is
    # still a full verified hit, not a partial one.
    assert second.reuse.policy_reusable_tokens.value == len(prompt) - 1
    assert second.reuse.engine_cached_tokens.value == len(prompt) - 1
    assert second.reuse.semantic_prefix_tokens.value == len(prompt)
    assert second.verdict is Verdict.VERIFIED_HIT


def test_interior_mutation_is_partial_reuse() -> None:
    adapter = _adapter()
    prompt = (11, 12, 13, 14, 15)
    adapter.run([_spec("cold", prompt, order=0)])
    adapter.run(
        [_spec("identical", prompt, order=1, scenario=ScenarioKind.IDENTICAL_PREFIX)]
    )

    mutated = (11, 12, 999, 14, 15)
    record = adapter.run(
        [
            _spec(
                "mutation",
                mutated,
                order=2,
                scenario=ScenarioKind.WITHIN_BLOCK_MUTATION,
                mutation_position=2,
            )
        ]
    )[0]
    assert record.verdict is Verdict.PARTIAL_REUSE
    assert record.reuse.semantic_prefix_tokens.value == 2
    assert record.reuse.semantic_prefix_tokens.value < len(mutated)


def test_corrupted_cache_reuse_is_invalid_on_output_mismatch() -> None:
    adapter = _adapter(FakeMLXRuntime(corrupt_when_cached=True))
    prompt = (21, 22, 23, 24, 25)
    adapter.run([_spec("cold", prompt, order=0)])
    record = adapter.run(
        [_spec("identical", prompt, order=1, scenario=ScenarioKind.IDENTICAL_PREFIX)]
    )[0]
    assert record.output.correctness.value is False
    assert record.verdict is Verdict.INVALID
    assert record.verdict_reasons == ("output_token_identity_mismatch",)


def test_exact_empty_remainder_is_refused_not_silently_regenerated() -> None:
    adapter = _adapter()
    prompt = (31, 32, 33)
    first = adapter.run([_spec("cold", prompt, order=0)])[0]
    assert first.output.output_token_ids is not None

    full_sequence = tuple(prompt) + tuple(first.output.output_token_ids)
    second = adapter.run(
        [
            _spec(
                "exact-empty",
                full_sequence,
                order=1,
                scenario=ScenarioKind.DUPLICATE,
            )
        ]
    )[0]
    assert second.terminal_state.value == "refused"
    assert second.verdict is Verdict.UNSUPPORTED
    assert second.verdict_reasons == ("unsupported:exact_empty_remainder_unsupported",)
    assert second.output.output_token_ids is None
    assert any(item.blocks_verdict for item in second.limitations)


def test_quantized_cache_scenario_is_explicitly_unsupported() -> None:
    adapter = _adapter()
    record = adapter.run(
        [_spec("quant", (1, 2, 3), order=0, scenario=ScenarioKind.QUANTIZED_CACHE)]
    )[0]
    assert record.verdict is Verdict.UNSUPPORTED
    assert record.verdict_reasons == ("unsupported:quantized_cache_unsupported",)


def test_observed_hidden_prompt_work_is_classified_as_recomputed() -> None:
    adapter = _adapter(FakeMLXRuntime(recomputed_extra_when_cached=2))
    first, second = adapter.run(
        [
            _spec("cold", (1, 2, 3), order=0),
            _spec("warm", (1, 2, 3, 4), order=1),
        ]
    )

    assert first.verdict is Verdict.VERIFIED_MISS
    assert second.reuse.engine_cached_tokens.value == 3
    assert second.reuse.unexpected_recomputed_tokens.value == 2
    assert second.verdict is Verdict.RECOMPUTED


def test_rotating_cache_scenario_is_explicitly_unsupported() -> None:
    adapter = _adapter()
    record = adapter.run(
        [_spec("rotate", (1, 2, 3), order=0, scenario=ScenarioKind.ROTATING_CACHE)]
    )[0]
    assert record.verdict is Verdict.UNSUPPORTED
    assert record.verdict_reasons == ("unsupported:rotating_cache_unsupported",)


def test_non_trimmable_native_cache_reuse_is_refused() -> None:
    adapter = _adapter(FakeMLXRuntime(trimmable=False))
    first = adapter.run([_spec("long", (1, 2, 3, 4), order=0)])[0]
    assert first.verdict is Verdict.VERIFIED_MISS

    shorter = adapter.run([_spec("shorter", (1, 2, 3), order=1)])[0]
    assert shorter.verdict is Verdict.UNSUPPORTED
    assert shorter.verdict_reasons == (
        "unsupported:non_trimmable_cache_reuse_unsupported",
    )


def test_run_requires_exact_token_ids() -> None:
    adapter = _adapter()
    spec = RequestSpec(
        request_id="no-tokens",
        scenario=ScenarioKind.COLD,
        order=0,
        input_token_ids=None,
        input_token_count=4,
    )
    with pytest.raises(MLXCacheAdapterError):
        adapter.run([spec])


# --- memory & timing evidence domains -------------------------------------


def test_memory_and_timing_evidence_are_observed_and_wall_clock() -> None:
    adapter = _adapter()
    record = adapter.run([_spec("cold", (41, 42, 43), order=0)])[0]

    assert record.memory.runtime_active_bytes.value is not None
    assert record.memory.runtime_active_bytes.value >= 0
    assert record.memory.runtime_peak_bytes.value is not None
    assert record.memory.allocator_cache_bytes.value is not None
    assert record.memory.logical_cache_bytes.value is not None
    assert record.memory.logical_cache_bytes.value > 0
    assert record.cache_before is not None
    assert record.cache_before.entry_count.value == 0
    assert record.cache_after is not None
    assert record.cache_after.entry_count.value == 1
    assert record.cache_after.complete is False

    assert record.timing.total is not None
    assert record.timing.total.provenance is MetricProvenance.MEASURED_WALL_CLOCK
    assert record.timing.total.unit == "s"
    assert record.timing.total.value >= 0
    assert record.timing.in_process_first_token is not None
    assert record.timing.in_process_first_token.value >= 0


def test_refusal_record_has_no_timing_but_has_partial_memory_facts() -> None:
    adapter = _adapter()
    prompt = (51, 52, 53)
    first = adapter.run([_spec("cold", prompt, order=0)])[0]
    full_sequence = tuple(prompt) + tuple(first.output.output_token_ids or ())
    refused = adapter.run(
        [_spec("exact-empty", full_sequence, order=1, scenario=ScenarioKind.DUPLICATE)]
    )[0]
    assert refused.timing.total is None
    assert refused.memory.runtime_active_bytes.value is not None


# --- saved-cache sidecar identity ------------------------------------------


def test_saved_cache_identity_round_trips_through_json() -> None:
    identity = SavedCacheIdentity.for_binding(
        model_key="model-a",
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        cache_payload_digest=_CACHE_PAYLOAD_DIGEST,
        mlx_version="0.32.2",
        mlx_lm_version="0.31.3",
        token_ids=(1, 2, 3),
    )
    restored = SavedCacheIdentity.from_json(identity.to_json())
    assert restored == identity


def test_load_saved_cache_succeeds_and_invokes_loader_once(tmp_path: Path) -> None:
    adapter = _adapter()
    cache_path = tmp_path / "prompt.safetensors"
    cache_path.write_bytes(b"synthetic-cache-payload")
    spec = _spec("bind", (1, 2, 3, 4), order=0)
    identity = SavedCacheIdentity.for_cache_file(
        cache_path,
        model_key=adapter.model_key,
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        mlx_version=REQUIRED_MLX_VERSION,
        mlx_lm_version=REQUIRED_MLX_LM_VERSION,
        token_ids=spec.input_token_ids or (),
    )
    write_saved_cache_sidecar(cache_path, identity)

    loader = Mock(return_value="loaded-cache")
    result = adapter.load_saved_cache(cache_path, request=spec, loader=loader)

    assert result == "loaded-cache"
    loader.assert_called_once_with(cache_path)
    assert cache_path.read_bytes() == b"synthetic-cache-payload"


def test_load_saved_cache_refuses_on_token_mismatch_without_loading(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    cache_path = tmp_path / "prompt.safetensors"
    cache_path.write_bytes(b"synthetic-cache-payload")
    bound_spec = _spec("bind", (1, 2, 3, 4), order=0)
    identity = SavedCacheIdentity.for_cache_file(
        cache_path,
        model_key=adapter.model_key,
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        mlx_version=REQUIRED_MLX_VERSION,
        mlx_lm_version=REQUIRED_MLX_LM_VERSION,
        token_ids=bound_spec.input_token_ids or (),
    )
    write_saved_cache_sidecar(cache_path, identity)

    mismatched_spec = _spec("bind-mismatch", (9, 9, 9, 9), order=0)
    loader = Mock(return_value="loaded-cache")
    with pytest.raises(MLXCacheAdapterError, match="mismatch"):
        adapter.load_saved_cache(cache_path, request=mismatched_spec, loader=loader)
    loader.assert_not_called()


def test_load_saved_cache_refuses_when_sidecar_missing(tmp_path: Path) -> None:
    adapter = _adapter()
    cache_path = tmp_path / "prompt.safetensors"
    cache_path.write_bytes(b"synthetic-cache-payload")
    spec = _spec("bind", (1, 2, 3, 4), order=0)
    loader = Mock(return_value="loaded-cache")
    with pytest.raises(MLXCacheAdapterError, match="missing"):
        adapter.load_saved_cache(cache_path, request=spec, loader=loader)
    loader.assert_not_called()


def test_load_saved_cache_refuses_replaced_payload_without_loading(
    tmp_path: Path,
) -> None:
    adapter = _adapter()
    cache_path = tmp_path / "prompt.safetensors"
    cache_path.write_bytes(b"synthetic-cache-payload")
    spec = _spec("bind", (1, 2, 3, 4), order=0)
    identity = SavedCacheIdentity.for_cache_file(
        cache_path,
        model_key=adapter.model_key,
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        mlx_version=REQUIRED_MLX_VERSION,
        mlx_lm_version=REQUIRED_MLX_LM_VERSION,
        token_ids=spec.input_token_ids or (),
    )
    write_saved_cache_sidecar(cache_path, identity)
    cache_path.write_bytes(b"replacement-cache-payload")

    loader = Mock(return_value="loaded-cache")
    with pytest.raises(MLXCacheAdapterError, match="payload digest mismatch"):
        adapter.load_saved_cache(cache_path, request=spec, loader=loader)
    loader.assert_not_called()


def test_verify_saved_cache_sidecar_is_a_standalone_helper(tmp_path: Path) -> None:
    cache_path = tmp_path / "prompt.safetensors"
    cache_path.write_bytes(b"synthetic-cache-payload")
    identity = SavedCacheIdentity.for_cache_file(
        cache_path,
        model_key="k",
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        mlx_version="0.32.2",
        mlx_lm_version="0.31.3",
        token_ids=(1, 2),
    )
    write_saved_cache_sidecar(cache_path, identity)
    verify_saved_cache_sidecar(cache_path, identity)  # does not raise

    other = SavedCacheIdentity.for_cache_file(
        cache_path,
        model_key="different",
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        mlx_version="0.32.2",
        mlx_lm_version="0.31.3",
        token_ids=(1, 2),
    )
    with pytest.raises(MLXCacheAdapterError):
        verify_saved_cache_sidecar(cache_path, other)


def test_write_saved_cache_sidecar_rejects_missing_payload(tmp_path: Path) -> None:
    identity = SavedCacheIdentity.for_binding(
        model_key="k",
        model_artifact_digest=_MODEL_ARTIFACT_DIGEST,
        cache_payload_digest=_CACHE_PAYLOAD_DIGEST,
        mlx_version="0.32.2",
        mlx_lm_version="0.31.3",
        token_ids=(1, 2),
    )
    with pytest.raises(MLXCacheAdapterError, match="regular"):
        write_saved_cache_sidecar(tmp_path / "missing.safetensors", identity)
