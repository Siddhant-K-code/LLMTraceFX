"""Collect normalized evidence from one MLX-LM inference run.

The normal collection path synchronizes MLX only at phase boundaries. It does
not force evaluation per layer or per token, which would materially change the
workload being measured. Native Metal performance counters are outside this
collector's scope.
"""

from __future__ import annotations

import math
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol, cast

from ...profiler.mlx_tracer import mlx_memory_snapshot
from ..manifest import collect_environment_manifest
from ..schema import (
    CommandInfo,
    ErrorInfo,
    ExperimentRecord,
    MemoryMetrics,
    ModelInfo,
    OutcomeInfo,
    RepetitionInfo,
    RuntimeInfo,
    SpeculativeDecodingInfo,
    TimingMetrics,
    TokenCounts,
    utc_now_iso,
)
from ._shared import (
    atomic_write_text,
    bytes_measurement,
    config_hash,
    milliseconds,
    record_platform,
    sha256_text,
)


class MLXCollectorError(RuntimeError):
    """Raised when MLX collection cannot be configured or started."""


@dataclass(frozen=True)
class MLXMemorySnapshot:
    """Allocator values exposed by MLX, in bytes."""

    active_bytes: int | None = None
    cache_bytes: int | None = None
    peak_bytes: int | None = None


class MLXGenerationResponse(Protocol):
    """Subset of ``mlx_lm.GenerationResponse`` consumed by the collector."""

    text: str
    from_draft: bool
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


class MLXRuntime(Protocol):
    """Injectable MLX-LM boundary used by the collector and its tests."""

    @property
    def mlx_version(self) -> str | None: ...

    @property
    def mlx_lm_version(self) -> str | None: ...

    @property
    def runtime_name(self) -> str: ...

    @property
    def runtime_version(self) -> str | None: ...

    def load_model(self, path: Path) -> tuple[Any, Any]: ...

    def encode(self, tokenizer: Any, prompt: str) -> list[int]: ...

    def seed(self, seed: int) -> None: ...

    def synchronize(self) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def memory_snapshot(self) -> MLXMemorySnapshot: ...

    def accelerator_name(self) -> str | None: ...

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        draft_model: Any | None,
        num_draft_tokens: int,
    ) -> Iterator[MLXGenerationResponse]: ...


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


class MLXLMRuntime:
    """Production adapter for MLX-LM on Apple Silicon."""

    def __init__(self, *, temperature: float = 0.0, top_p: float = 1.0) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise MLXCollectorError(
                "MLX collection requires Apple Silicon running macOS"
            )
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise MLXCollectorError(
                "MLX collection requires the optional runtime dependencies. "
                "Install them with `uv sync --extra mlx` or "
                "`pip install 'llmtracefx[mlx]'`."
            ) from exc

        self._mx = mx
        self._load = load
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self.configure_sampling(temperature=temperature, top_p=top_p)

    def configure_sampling(self, *, temperature: float, top_p: float) -> None:
        self._sampler = self._make_sampler(temp=temperature, top_p=top_p)

    @property
    def mlx_version(self) -> str | None:
        return _installed_version("mlx")

    @property
    def mlx_lm_version(self) -> str | None:
        return _installed_version("mlx-lm")

    @property
    def runtime_name(self) -> str:
        return "mlx-lm"

    @property
    def runtime_version(self) -> str | None:
        return self.mlx_lm_version

    def load_model(self, path: Path) -> tuple[Any, Any]:
        loaded = self._load(str(path), lazy=False, return_config=False)
        if len(loaded) != 2:
            raise MLXCollectorError("mlx_lm.load returned an unexpected result shape")
        return loaded[0], loaded[1]

    def encode(self, tokenizer: Any, prompt: str) -> list[int]:
        bos_token = getattr(tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not prompt.startswith(bos_token)
        encoded = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        return [int(token) for token in encoded]

    def seed(self, seed: int) -> None:
        self._mx.random.seed(seed)

    def synchronize(self) -> None:
        synchronize = getattr(self._mx, "synchronize", None)
        if synchronize is None:
            raise MLXCollectorError(
                "This MLX version does not expose mx.synchronize(), which "
                "phase-boundary timing depends on for correctness"
            )
        synchronize()

    def reset_peak_memory(self) -> None:
        reset = getattr(self._mx, "reset_peak_memory", None)
        if reset is not None:
            reset()

    def memory_snapshot(self) -> MLXMemorySnapshot:
        snapshot = mlx_memory_snapshot(self._mx)
        return MLXMemorySnapshot(
            active_bytes=snapshot.get("active_memory_bytes"),
            cache_bytes=snapshot.get("cache_memory_bytes"),
            peak_bytes=snapshot.get("peak_memory_bytes"),
        )

    def accelerator_name(self) -> str | None:
        device_info_fn = getattr(self._mx, "device_info", None)
        if device_info_fn is None:
            metal = getattr(self._mx, "metal", None)
            device_info_fn = (
                getattr(metal, "device_info", None) if metal is not None else None
            )
        if device_info_fn is None:
            return None
        device_info = device_info_fn()
        if not isinstance(device_info, dict):
            return None
        for key in ("device_name", "architecture"):
            value = device_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

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
        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "sampler": self._sampler,
        }
        if draft_model is not None:
            kwargs["draft_model"] = draft_model
            kwargs["num_draft_tokens"] = num_draft_tokens
        return cast(
            Iterator[MLXGenerationResponse],
            self._stream_generate(model, tokenizer, prompt_tokens, **kwargs),
        )


@dataclass
class _MLXVLMGenerationResponse:
    text: str
    from_draft: bool
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


class MLXVLMRuntime:
    """Text-only adapter for MLX-VLM checkpoints such as Qwen3.8.

    The adapter deliberately requires an existing local checkpoint path. It
    applies the checkpoint's chat template with no image/audio/video inputs,
    tokenizes before the collector's generation phase, and passes those exact
    token IDs to ``mlx_vlm.stream_generate``. This keeps prompt-token counts
    measured by the model's own processor and avoids an implicit second
    tokenization inside the timed prefill boundary.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        enable_thinking: bool = False,
        prefill_step_size: int = 2048,
    ) -> None:
        if temperature < 0:
            raise MLXCollectorError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise MLXCollectorError("top_p must be within (0, 1]")
        if prefill_step_size < 1:
            raise MLXCollectorError("prefill_step_size must be positive")
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise MLXCollectorError(
                "MLX-VLM collection requires Apple Silicon running macOS"
            )
        try:
            import mlx.core as mx
            from mlx_vlm import load, stream_generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError as exc:
            raise MLXCollectorError(
                "MLX-VLM collection requires the optional runtime dependencies. "
                "Install them with `uv sync --extra mlx`."
            ) from exc

        self._mx = mx
        self._load = load
        self._stream_generate = stream_generate
        self._apply_chat_template = apply_chat_template
        self._temperature = temperature
        self._top_p = top_p
        self._enable_thinking = enable_thinking
        self._prefill_step_size = prefill_step_size
        self._loaded: dict[Path, tuple[Any, Any]] = {}
        self._active_config: Any | None = None

    @property
    def mlx_version(self) -> str | None:
        return _installed_version("mlx")

    @property
    def mlx_lm_version(self) -> str | None:
        return _installed_version("mlx-vlm")

    @property
    def runtime_name(self) -> str:
        return "mlx-vlm"

    @property
    def runtime_version(self) -> str | None:
        return _installed_version("mlx-vlm")

    def load_model(self, path: Path) -> tuple[Any, Any]:
        resolved = path.resolve()
        loaded = self._loaded.get(resolved)
        if loaded is None:
            loaded = self._load(str(resolved), lazy=False)
            if len(loaded) != 2:
                raise MLXCollectorError(
                    "mlx_vlm.load returned an unexpected result shape"
                )
            self._loaded[resolved] = (loaded[0], loaded[1])
        model, processor = self._loaded[resolved]
        self._active_config = model.config
        return model, processor

    def encode(self, processor: Any, prompt: str) -> list[int]:
        if self._active_config is None:
            raise MLXCollectorError("MLX-VLM model must be loaded before tokenization")
        formatted = self._apply_chat_template(
            processor,
            self._active_config,
            prompt,
            num_images=0,
            num_audios=0,
            enable_thinking=self._enable_thinking,
        )
        if not isinstance(formatted, str):
            raise MLXCollectorError(
                "MLX-VLM chat template returned a non-text prompt for a text-only run"
            )
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        encoded = tokenizer.encode(formatted, add_special_tokens=True)
        return [int(token) for token in encoded]

    def seed(self, seed: int) -> None:
        self._mx.random.seed(seed)

    def synchronize(self) -> None:
        self._mx.synchronize()

    def reset_peak_memory(self) -> None:
        reset = getattr(self._mx, "reset_peak_memory", None)
        if reset is not None:
            reset()

    def memory_snapshot(self) -> MLXMemorySnapshot:
        snapshot = mlx_memory_snapshot(self._mx)
        return MLXMemorySnapshot(
            active_bytes=snapshot.get("active_memory_bytes"),
            cache_bytes=snapshot.get("cache_memory_bytes"),
            peak_bytes=snapshot.get("peak_memory_bytes"),
        )

    def accelerator_name(self) -> str | None:
        device_info = cast(Any, self._mx.device_info())
        if not isinstance(device_info, dict):
            return None
        for key in ("device_name", "architecture"):
            value = device_info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def stream_generate(
        self,
        model: Any,
        processor: Any,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        draft_model: Any | None,
        num_draft_tokens: int,
    ) -> Iterator[MLXGenerationResponse]:
        if draft_model is not None:
            raise MLXCollectorError(
                "generic MLX-LM draft models are not accepted by MLX-VLM runs"
            )
        input_ids = self._mx.array([prompt_tokens])
        for response in self._stream_generate(
            model,
            processor,
            "",
            input_ids=input_ids,
            max_tokens=max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            enable_thinking=self._enable_thinking,
            prefill_step_size=self._prefill_step_size,
            verbose=False,
        ):
            yield _MLXVLMGenerationResponse(
                text=response.text,
                from_draft=bool(getattr(response, "is_draft", False)),
                prompt_tokens=int(response.prompt_tokens),
                generation_tokens=int(response.generation_tokens),
                finish_reason=response.finish_reason,
            )


@dataclass(frozen=True)
class MLXCollectionConfig:
    """Inputs and reproducibility metadata for one local MLX-LM run."""

    run_id: str
    model_path: Path
    model_id: str
    prompt: str
    output_dir: Path
    command_argv: tuple[str, ...]
    max_tokens: int = 128
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False
    prefill_step_size: int | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    model_family: str | None = None
    accelerator: str | None = None
    draft_model_path: Path | None = None
    num_draft_tokens: int = 2
    timeout_seconds: float | None = None
    warmup_repetitions: int = 0
    measured_repetitions: int = 1
    repetition_index: int = 0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise MLXCollectorError("run_id must be non-empty")
        if not self.model_id:
            raise MLXCollectorError("model_id must be non-empty")
        if not self.model_path.exists():
            raise MLXCollectorError(
                f"model_path does not exist: {self.model_path}. "
                "Download or convert the model separately before collection."
            )
        if self.draft_model_path is not None and not self.draft_model_path.exists():
            raise MLXCollectorError(
                f"draft_model_path does not exist: {self.draft_model_path}"
            )
        if not self.command_argv or not all(self.command_argv):
            raise MLXCollectorError(
                "command_argv must contain non-empty argument strings"
            )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise MLXCollectorError("max_tokens must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise MLXCollectorError("seed must be an integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature
        ):
            raise MLXCollectorError("temperature must be non-negative")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise MLXCollectorError("top_p must be within (0, 1]")
        if not isinstance(self.enable_thinking, bool):
            raise MLXCollectorError("enable_thinking must be a boolean")
        if self.prefill_step_size is not None and (
            isinstance(self.prefill_step_size, bool)
            or not isinstance(self.prefill_step_size, int)
            or self.prefill_step_size < 1
        ):
            raise MLXCollectorError("prefill_step_size must be positive when set")
        if (
            isinstance(self.num_draft_tokens, bool)
            or not isinstance(self.num_draft_tokens, int)
            or self.num_draft_tokens < 1
        ):
            raise MLXCollectorError("num_draft_tokens must be a positive integer")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise MLXCollectorError("timeout_seconds must be positive when set")
        if (
            isinstance(self.warmup_repetitions, bool)
            or not isinstance(self.warmup_repetitions, int)
            or self.warmup_repetitions < 0
        ):
            raise MLXCollectorError("warmup_repetitions must be a non-negative integer")
        if (
            isinstance(self.measured_repetitions, bool)
            or not isinstance(self.measured_repetitions, int)
            or self.measured_repetitions < 1
        ):
            raise MLXCollectorError("measured_repetitions must be a positive integer")
        if (
            isinstance(self.repetition_index, bool)
            or not isinstance(self.repetition_index, int)
            or self.repetition_index < 0
            or self.repetition_index >= self.measured_repetitions
        ):
            raise MLXCollectorError(
                "repetition_index must identify a measured repetition"
            )


@dataclass(frozen=True)
class MLXCollectionResult:
    """Canonical record plus the generated response text."""

    record: ExperimentRecord
    response_text: str


def mlx_collection_contract_hash(
    *,
    model_id: str,
    model_revision: str | None,
    tokenizer_revision: str | None,
    quantization: str | None,
    model_family: str | None,
    max_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    prefill_step_size: int | None,
    draft_enabled: bool,
    num_draft_tokens: int,
    timeout_seconds: float | None,
) -> str:
    """Hash every semantic collector setting, excluding paths and repetition."""
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "quantization": quantization,
        "model_family": model_family,
        "max_tokens": max_tokens,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "enable_thinking": enable_thinking,
        "prefill_step_size": prefill_step_size,
        "draft_enabled": draft_enabled,
        "num_draft_tokens": num_draft_tokens,
        "timeout_seconds": timeout_seconds,
    }
    return config_hash(payload)


def _config_hash(config: MLXCollectionConfig) -> str:
    return mlx_collection_contract_hash(
        model_id=config.model_id,
        model_revision=config.model_revision,
        tokenizer_revision=config.tokenizer_revision,
        quantization=config.quantization,
        model_family=config.model_family,
        max_tokens=config.max_tokens,
        seed=config.seed,
        temperature=config.temperature,
        top_p=config.top_p,
        enable_thinking=config.enable_thinking,
        prefill_step_size=config.prefill_step_size,
        draft_enabled=config.draft_model_path is not None,
        num_draft_tokens=config.num_draft_tokens,
        timeout_seconds=config.timeout_seconds,
    )


def collect_mlx(
    config: MLXCollectionConfig,
    *,
    runtime: MLXRuntime,
    clock: Callable[[], float] = time.perf_counter,
) -> MLXCollectionResult:
    """Run one MLX-LM generation and persist normalized evidence.

    Runtime failures are represented as failed experiment records. Invalid
    collector configuration and artifact write failures remain explicit
    exceptions.
    """

    started_at = utc_now_iso()
    total_started = clock()
    load_started: float | None = None
    load_ended: float | None = None
    tokenize_started: float | None = None
    tokenize_ended: float | None = None
    generation_started: float | None = None
    first_token_at: float | None = None
    generation_ended: float | None = None
    prompt_tokens: list[int] = []
    generated_tokens = 0
    accepted_draft_tokens = 0
    response_parts: list[str] = []
    memory = MLXMemorySnapshot()
    error: ErrorInfo | None = None

    def check_timeout(stage: str) -> None:
        if (
            config.timeout_seconds is not None
            and clock() - total_started > config.timeout_seconds
        ):
            raise TimeoutError(
                f"MLX collection exceeded {config.timeout_seconds:g}s "
                f"timeout during {stage}"
            )

    try:
        load_started = clock()
        model, tokenizer = runtime.load_model(config.model_path)
        draft_model = (
            runtime.load_model(config.draft_model_path)[0]
            if config.draft_model_path is not None
            else None
        )
        runtime.synchronize()
        load_ended = clock()
        check_timeout("model load")

        runtime.reset_peak_memory()
        tokenize_started = clock()
        prompt_tokens = runtime.encode(tokenizer, config.prompt)
        tokenize_ended = clock()
        check_timeout("tokenization")
        runtime.seed(config.seed)
        runtime.synchronize()

        generation_started = clock()
        previous_generation_tokens = 0
        for response in runtime.stream_generate(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens=config.max_tokens,
            draft_model=draft_model,
            num_draft_tokens=config.num_draft_tokens,
        ):
            observed_at = clock()
            check_timeout("generation")
            if first_token_at is None:
                first_token_at = observed_at
            response_parts.append(response.text)
            is_eos_summary = response.finish_reason == "stop"
            if (
                response.generation_tokens > previous_generation_tokens
                and not is_eos_summary
            ):
                if response.from_draft:
                    accepted_draft_tokens += 1
                previous_generation_tokens = response.generation_tokens
                generated_tokens = response.generation_tokens

        runtime.synchronize()
        generation_ended = clock()
        memory = runtime.memory_snapshot()
    except (
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        MemoryError,
        TimeoutError,
    ) as exc:
        generation_ended = clock()
        error = ErrorInfo(category=type(exc).__name__, message=str(exc))

    total_ended = clock()
    record = ExperimentRecord(
        run_id=config.run_id,
        started_at=started_at,
        ended_at=utc_now_iso(),
        platform=record_platform(
            accelerator=config.accelerator or runtime.accelerator_name()
        ),
        model=ModelInfo(
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            quantization=config.quantization,
            model_family=config.model_family,
        ),
        runtime=RuntimeInfo(
            name=getattr(runtime, "runtime_name", "mlx-lm"),
            version=getattr(runtime, "runtime_version", runtime.mlx_lm_version),
            backend="Metal",
            git_revision=None,
        ),
        command=CommandInfo(
            argv=config.command_argv,
            config_hash=_config_hash(config),
            workload_hash=sha256_text(config.prompt),
        ),
        repetition=RepetitionInfo(
            warmup_repetitions=config.warmup_repetitions,
            measured_repetitions=config.measured_repetitions,
            repetition_index=config.repetition_index,
            seed=config.seed,
        ),
        tokens=TokenCounts(
            input_tokens=len(prompt_tokens) if prompt_tokens else None,
            context_tokens=len(prompt_tokens) if prompt_tokens else None,
            generated_tokens=generated_tokens or None,
        ),
        timing=TimingMetrics(
            model_load=milliseconds(load_started, load_ended),
            tokenize=milliseconds(tokenize_started, tokenize_ended),
            prefill=milliseconds(generation_started, first_token_at),
            decode=milliseconds(first_token_at, generation_ended),
            total=milliseconds(total_started, total_ended),
        ),
        speculative=SpeculativeDecodingInfo(
            enabled=config.draft_model_path is not None,
            method=("draft-model" if config.draft_model_path is not None else None),
            configured_depth=(
                config.num_draft_tokens if config.draft_model_path is not None else None
            ),
            proposed_tokens=None,
            accepted_tokens=(
                accepted_draft_tokens if config.draft_model_path is not None else None
            ),
            verification_time=None,
        ),
        memory=MemoryMetrics(
            active=bytes_measurement(memory.active_bytes),
            cache=bytes_measurement(memory.cache_bytes),
            peak=bytes_measurement(memory.peak_bytes),
            wired=None,
        ),
        outcome=OutcomeInfo(success=error is None),
        error=error,
    )
    record.validate()

    response_text = "".join(response_parts)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    record.write_json(config.output_dir / "record.json")
    atomic_write_text(config.output_dir / "response.txt", response_text)
    manifest = collect_environment_manifest(
        extra_packages=(
            "mlx",
            "mlx-lm",
            "mlx-vlm",
            "transformers",
            "huggingface-hub",
        )
    )
    atomic_write_text(config.output_dir / "environment.json", manifest.to_json() + "\n")
    return MLXCollectionResult(record=record, response_text=response_text)
