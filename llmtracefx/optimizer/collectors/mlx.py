"""Collect normalized evidence from one MLX-LM inference run.

The normal collection path synchronizes MLX only at phase boundaries. It does
not force evaluation per layer or per token, which would materially change the
workload being measured. Native Metal performance counters are outside this
collector's scope.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

from ...profiler.mlx_tracer import mlx_memory_snapshot
from ..manifest import collect_environment_manifest
from ..schema import (
    CommandInfo,
    ErrorInfo,
    ExperimentRecord,
    Measurement,
    MemoryMetrics,
    MetricProvenance,
    ModelInfo,
    OutcomeInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SpeculativeDecodingInfo,
    TimingMetrics,
    TokenCounts,
    utc_now_iso,
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


class MLXRuntime(Protocol):
    """Injectable MLX-LM boundary used by the collector and its tests."""

    @property
    def mlx_version(self) -> str | None: ...

    @property
    def mlx_lm_version(self) -> str | None: ...

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

    def __init__(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise MLXCollectorError(
                "MLX collection requires Apple Silicon running macOS"
            )
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
            from mlx_lm import load, stream_generate
        except ImportError as exc:
            raise MLXCollectorError(
                "MLX collection requires the optional runtime dependencies. "
                "Install them with `uv sync --extra mlx` or "
                "`pip install 'llmtracefx[mlx]'`."
            ) from exc

        self._mx = mx
        self._load = load
        self._stream_generate = stream_generate

    @property
    def mlx_version(self) -> str | None:
        return _installed_version("mlx")

    @property
    def mlx_lm_version(self) -> str | None:
        return _installed_version("mlx-lm")

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
        kwargs: dict[str, Any] = {"max_tokens": max_tokens}
        if draft_model is not None:
            kwargs["draft_model"] = draft_model
            kwargs["num_draft_tokens"] = num_draft_tokens
        return self._stream_generate(model, tokenizer, prompt_tokens, **kwargs)


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
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    accelerator: str | None = None
    draft_model_path: Path | None = None
    num_draft_tokens: int = 2

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
            isinstance(self.num_draft_tokens, bool)
            or not isinstance(self.num_draft_tokens, int)
            or self.num_draft_tokens < 1
        ):
            raise MLXCollectorError("num_draft_tokens must be a positive integer")


@dataclass(frozen=True)
class MLXCollectionResult:
    """Canonical record plus the generated response text."""

    record: ExperimentRecord
    response_text: str


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _config_hash(config: MLXCollectionConfig) -> str:
    payload = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "quantization": config.quantization,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "draft_enabled": config.draft_model_path is not None,
        "num_draft_tokens": config.num_draft_tokens,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _milliseconds(started: float | None, ended: float | None) -> Measurement | None:
    if started is None or ended is None:
        return None
    return Measurement(
        value=max(0.0, ended - started) * 1000,
        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
        unit="ms",
    )


def _bytes_measurement(value: int | None) -> Measurement | None:
    if value is None:
        return None
    return Measurement(
        value=float(value),
        provenance=MetricProvenance.MEASURED_NATIVE,
        unit="bytes",
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _record_platform(
    runtime: MLXRuntime, accelerator_override: str | None
) -> PlatformInfo:
    manifest = collect_environment_manifest(extra_packages=("mlx", "mlx-lm"))
    accelerator = accelerator_override or runtime.accelerator_name()
    return PlatformInfo(
        os_name=manifest.os_name,
        os_version=manifest.os_release,
        architecture=manifest.architecture,
        cpu_cores=manifest.cpu_count,
        total_memory_gb=manifest.total_memory_gb,
        accelerator=accelerator,
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

        runtime.reset_peak_memory()
        tokenize_started = clock()
        prompt_tokens = runtime.encode(tokenizer, config.prompt)
        tokenize_ended = clock()
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
            if first_token_at is None:
                first_token_at = observed_at
            response_parts.append(response.text)
            if response.generation_tokens > previous_generation_tokens:
                if response.from_draft:
                    accepted_draft_tokens += 1
                previous_generation_tokens = response.generation_tokens
            generated_tokens = max(generated_tokens, response.generation_tokens)

        runtime.synchronize()
        generation_ended = clock()
        memory = runtime.memory_snapshot()
    except (RuntimeError, ValueError, OSError, MemoryError) as exc:
        generation_ended = clock()
        error = ErrorInfo(category=type(exc).__name__, message=str(exc))

    total_ended = clock()
    record = ExperimentRecord(
        run_id=config.run_id,
        started_at=started_at,
        ended_at=utc_now_iso(),
        platform=_record_platform(runtime, config.accelerator),
        model=ModelInfo(
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            quantization=config.quantization,
        ),
        runtime=RuntimeInfo(
            name="mlx-lm",
            version=runtime.mlx_lm_version,
            backend="Metal",
            git_revision=None,
        ),
        command=CommandInfo(
            argv=config.command_argv,
            config_hash=_config_hash(config),
            workload_hash=_sha256_text(config.prompt),
        ),
        repetition=RepetitionInfo(
            warmup_repetitions=0,
            measured_repetitions=1,
            repetition_index=0,
            seed=config.seed,
        ),
        tokens=TokenCounts(
            input_tokens=len(prompt_tokens) if prompt_tokens else None,
            context_tokens=len(prompt_tokens) if prompt_tokens else None,
            generated_tokens=generated_tokens or None,
        ),
        timing=TimingMetrics(
            model_load=_milliseconds(load_started, load_ended),
            tokenize=_milliseconds(tokenize_started, tokenize_ended),
            prefill=_milliseconds(generation_started, first_token_at),
            decode=_milliseconds(first_token_at, generation_ended),
            total=_milliseconds(total_started, total_ended),
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
            active=_bytes_measurement(memory.active_bytes),
            cache=_bytes_measurement(memory.cache_bytes),
            peak=_bytes_measurement(memory.peak_bytes),
            wired=None,
        ),
        outcome=OutcomeInfo(success=error is None),
        error=error,
    )
    record.validate()

    response_text = "".join(response_parts)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    record.write_json(config.output_dir / "record.json")
    _atomic_write_text(config.output_dir / "response.txt", response_text)
    manifest = collect_environment_manifest(extra_packages=("mlx", "mlx-lm"))
    _atomic_write_text(
        config.output_dir / "environment.json", manifest.to_json() + "\n"
    )
    return MLXCollectionResult(record=record, response_text=response_text)
