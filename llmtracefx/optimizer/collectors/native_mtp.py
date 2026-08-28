"""Native Qwen multi-token-prediction (MTP) capability detection and evidence.

This module answers one question honestly: can this project invoke a
model's own multi-token-prediction heads through a stable, public MLX
Python API, in a way whose reported metrics (accepted/proposed counts,
block depth, verification timing) can be reliably attributed to native
MTP rather than to generic external draft-model speculative decoding
(the mechanism already collected by ``llmtracefx.optimizer.collectors.mlx``
via ``draft_model``/``num_draft_tokens``)?

As of this module's authoring, the answer is **no** for every model
family checked below. Verified upstream facts:

* ``mlx-lm`` (the runtime this project's ``collect-mlx`` wraps) strips
  multi-token-prediction weights during model loading for every model
  family that ships them. See, on ``ml-explore/mlx-lm`` main:
  ``mlx_lm/models/qwen3_next.py`` (``sanitize`` drops any key containing
  ``"mtp."``), ``mlx_lm/models/qwen3_5.py`` (same), plus the same pattern
  in ``ernie4_5_moe.py``, ``mimo.py``, ``step3p5.py``, ``exaone_moe.py``,
  ``bailing_moe_v3.py``, ``nemotron_h.py``, ``mimo_v2_flash.py``,
  ``kimi_linear.py`` and ``longcat_flash.py``. There is no code path in
  ``mlx-lm`` that loads or invokes an MTP head; ``mlx_lm.generate`` /
  ``mlx_lm.stream_generate`` only support the classic two-model
  draft/target speculative decoding already implemented by this
  project's MLX-LM collector.
* ``mlx-vlm`` (``Blaizzy/mlx-vlm``) has an experimental
  ``mlx_vlm.speculative.drafters`` package with native-MTP dispatch for
  a narrow set of architecture families (``qwen4_exp``, ``deepseek_v4``,
  ``glm4_moe_lite``, ``inkling_mm_model``), reached via ``--draft-kind
  mtp`` / ``draft_kind="mtp"``. This is not a stable public release: the
  ``qwen4_exp`` model type is explicitly named/labeled experimental,
  published third-party MTP-drafter checkpoints document that they
  require "git main" builds, and public MLX conversions of the base
  model drop the MTP tensors entirely (a separate, hand-restored
  "drafter" checkpoint must be supplied). Critically, even where this
  exists, it is dispatched through the *same* ``draft_model`` request
  path used for generic speculative decoding — the collector cannot
  distinguish "native MTP" evidence from "generic draft-model
  speculation" from the generation response alone. Only the checkpoint
  provenance (an official MTP-derived sidecar matching the target's
  architecture) distinguishes the two, and that is a best-effort,
  metadata-only check (see ``validate_checkpoint_compatibility`` below),
  not a runtime guarantee.

Given that, this module never silently reinterprets generic draft-model
speculation as native MTP. When capability detection reports the
runtime cannot support it, ``collect_native_mtp`` records an explicit,
honest "unsupported" ``ExperimentRecord`` (and a standalone capability
report) instead of attempting a misleading run.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol

from ..manifest import collect_environment_manifest
from ..schema import (
    CommandInfo,
    ErrorInfo,
    ExperimentRecord,
    MemoryMetrics,
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
from ._shared import (
    atomic_write_text,
    bytes_measurement,
    config_hash,
    milliseconds,
    record_platform,
    sha256_text,
)

CAPABILITY_SCHEMA_VERSION = "1"


class NativeMTPCollectorError(RuntimeError):
    """Raised for invalid native-MTP collector configuration or checkpoints."""


# --- Verified upstream capability facts -------------------------------------

#: Architecture families ``mlx-lm`` (main, as verified above) strips MTP
#: weights for during ``sanitize()``. Confirmed by direct inspection of
#: ml-explore/mlx-lm model source; not inferred or guessed.
MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES: frozenset[str] = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_moe",
        "ernie4_5_moe",
        "mimo",
        "step3p5",
        "exaone_moe",
        "bailing_moe_v3",
        "nemotron_h",
        "mimo_v2_flash",
        "kimi_linear",
        "longcat_flash",
    }
)

#: Architecture families with experimental, git-main-only native-MTP
#: dispatch in ``Blaizzy/mlx-vlm`` (``mlx_vlm.speculative.drafters``,
#: ``--draft-kind mtp``). Not part of any stable/tagged mlx-vlm release
#: at verification time, and still funneled through the generic
#: draft-model request path.
MLX_VLM_EXPERIMENTAL_MTP_FAMILIES: frozenset[str] = frozenset(
    {
        "qwen4_exp",
        "qwen4_exp_text",
        "deepseek_v4",
        "glm4_moe_lite",
        "inkling_mm_model",
    }
)

UPSTREAM_REFERENCES: tuple[str, ...] = (
    "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_next.py",
    "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py",
    "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py",
    "https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/README.md",
    "https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/server/cli.py",
)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class NativeMTPCapabilityReport:
    """Whether this environment can produce trustworthy native-MTP evidence.

    This is a standalone artifact (not an ``ExperimentRecord``): a
    capability determination is a property of the installed runtime and
    the requested architecture family, not of any single measured run.
    """

    schema_version: str
    model_family: str
    mlx_lm_version: str | None
    mlx_vlm_version: str | None
    supported: bool
    reason: str
    checked_signals: tuple[str, ...]
    references: tuple[str, ...] = UPSTREAM_REFERENCES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "mlx_lm_version": self.mlx_lm_version,
            "mlx_vlm_version": self.mlx_vlm_version,
            "supported": self.supported,
            "reason": self.reason,
            "checked_signals": list(self.checked_signals),
            "references": list(self.references),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def write_json(self, path: str | Path) -> None:
        atomic_write_text(Path(path), self.to_json() + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NativeMTPCapabilityReport:
        try:
            return cls(
                schema_version=str(
                    data.get("schema_version", CAPABILITY_SCHEMA_VERSION)
                ),
                model_family=data["model_family"],
                mlx_lm_version=data.get("mlx_lm_version"),
                mlx_vlm_version=data.get("mlx_vlm_version"),
                supported=bool(data["supported"]),
                reason=data["reason"],
                checked_signals=tuple(data.get("checked_signals", ())),
                references=tuple(data.get("references", UPSTREAM_REFERENCES)),
            )
        except KeyError as exc:
            raise NativeMTPCollectorError(
                f"NativeMTPCapabilityReport is missing required field: {exc}"
            ) from exc


def detect_native_mtp_capability(
    model_family: str,
    *,
    mlx_lm_version: str | None,
    mlx_vlm_version: str | None,
) -> NativeMTPCapabilityReport:
    """Determine whether native MTP evidence can be trusted for a family.

    Deliberately conservative: returns ``supported=False`` unless a
    verified, metrics-differentiated stable API is known. See the module
    docstring for the verified upstream facts this relies on.
    """
    checked = (
        f"mlx-lm installed: {mlx_lm_version is not None}",
        f"mlx-vlm installed: {mlx_vlm_version is not None}",
        f"model_family in MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES: "
        f"{model_family in MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES}",
        f"model_family in MLX_VLM_EXPERIMENTAL_MTP_FAMILIES: "
        f"{model_family in MLX_VLM_EXPERIMENTAL_MTP_FAMILIES}",
    )

    if model_family in MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES:
        return NativeMTPCapabilityReport(
            schema_version=CAPABILITY_SCHEMA_VERSION,
            model_family=model_family,
            mlx_lm_version=mlx_lm_version,
            mlx_vlm_version=mlx_vlm_version,
            supported=False,
            reason=(
                f"mlx-lm strips '{model_family}' multi-token-prediction "
                "weights during model loading (sanitize() removes any "
                "'mtp.'-prefixed tensor) and exposes no code path that "
                "loads or invokes an MTP head. Only generic external "
                "draft-model speculative decoding (draft_model/"
                "num_draft_tokens on mlx_lm.generate/stream_generate) is "
                "available; use collect-mlx --draft-model-path for that, "
                "labeled 'draft-model', not native MTP."
            ),
            checked_signals=checked,
        )

    if model_family in MLX_VLM_EXPERIMENTAL_MTP_FAMILIES:
        return NativeMTPCapabilityReport(
            schema_version=CAPABILITY_SCHEMA_VERSION,
            model_family=model_family,
            mlx_lm_version=mlx_lm_version,
            mlx_vlm_version=mlx_vlm_version,
            supported=False,
            reason=(
                f"mlx-vlm has experimental native-MTP dispatch for "
                f"'{model_family}' (mlx_vlm.speculative.drafters, "
                "--draft-kind mtp), but it is not part of any stable/"
                "tagged mlx-vlm release, requires a separately restored "
                "MTP-drafter checkpoint (public conversions drop the "
                "tensors), and is dispatched through the same "
                "draft_model request path as generic speculative "
                "decoding. Proposed/accepted/verification metrics "
                "cannot be reliably attributed to native MTP rather "
                "than generic draft-model speculation from the "
                "generation response alone, so this project does not "
                "label it 'native-mtp' evidence."
            ),
            checked_signals=checked,
        )

    return NativeMTPCapabilityReport(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        model_family=model_family,
        mlx_lm_version=mlx_lm_version,
        mlx_vlm_version=mlx_vlm_version,
        supported=False,
        reason=(
            f"model family '{model_family}' is not in this module's "
            "verified list of families with either MTP-weight-stripping "
            "behavior (mlx-lm) or experimental MTP dispatch (mlx-vlm). "
            "No known stable, metrics-differentiated native-MTP API was "
            "found for it; treat as unsupported until verified against "
            "current upstream source and added to this module's tables."
        ),
        checked_signals=checked,
    )


# --- Checkpoint loading and compatibility validation ------------------------


def _read_config_json(path: Path, *, label: str) -> dict[str, Any]:
    config_path = path / "config.json"
    if not path.exists():
        raise NativeMTPCollectorError(
            f"{label} does not exist: {path}. Download or convert the "
            "checkpoint separately before collection; it is never "
            "downloaded implicitly."
        )
    if not config_path.exists():
        raise NativeMTPCollectorError(f"{label} is missing config.json: {config_path}")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NativeMTPCollectorError(
            f"{label} config.json is not valid JSON: {config_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise NativeMTPCollectorError(
            f"{label} config.json must contain a JSON object: {config_path}"
        )
    return data


def _arch_signature(config: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable architecture fields, unwrapping VLM text_config.

    ``num_hidden_layers`` is extracted for informational/debugging
    purposes only (see ``validate_checkpoint_compatibility``); it is
    never compared for equality.
    """
    source = config
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        source = text_config
    return {
        key: source.get(key)
        for key in ("hidden_size", "vocab_size", "num_hidden_layers")
        if key in source
    }


def validate_checkpoint_compatibility(
    target_config: dict[str, Any], sidecar_config: dict[str, Any]
) -> None:
    """Fail clearly when the target/sidecar checkpoints look incompatible.

    Compares only ``hidden_size`` and ``vocab_size`` from public
    ``config.json`` metadata; a mismatch there means the two checkpoints
    cannot share a tokenizer/embedding space and are structurally
    incompatible. This is a best-effort, metadata-only check -- it
    cannot prove weight compatibility, only rule out checkpoints that
    are structurally mismatched.

    ``num_hidden_layers`` is deliberately **not** enforced: a native-MTP
    sidecar/drafter checkpoint is expected to have a different (almost
    always much smaller) layer count than its target model -- that is
    the whole point of a lightweight MTP head/drafter. When available on
    both checkpoints, it is surfaced as informational context in a
    hidden/vocab mismatch error, never as a compatibility requirement by
    itself.
    """
    target_sig = _arch_signature(target_config)
    sidecar_sig = _arch_signature(sidecar_config)

    if not target_sig:
        raise NativeMTPCollectorError(
            "target checkpoint config.json does not expose hidden_size/"
            "vocab_size; cannot validate compatibility with the sidecar"
        )
    if not sidecar_sig:
        raise NativeMTPCollectorError(
            "sidecar checkpoint config.json does not expose hidden_size/"
            "vocab_size; cannot validate compatibility with the target"
        )

    mismatches = [
        f"{key}: target={target_sig[key]!r} sidecar={sidecar_sig[key]!r}"
        for key in ("hidden_size", "vocab_size")
        if key in target_sig
        and key in sidecar_sig
        and target_sig[key] != sidecar_sig[key]
    ]
    if mismatches:
        layer_note = ""
        if "num_hidden_layers" in target_sig or "num_hidden_layers" in sidecar_sig:
            layer_note = (
                " (informational, not enforced: target num_hidden_layers="
                f"{target_sig.get('num_hidden_layers')!r}, sidecar "
                f"num_hidden_layers={sidecar_sig.get('num_hidden_layers')!r})"
            )
        raise NativeMTPCollectorError(
            "target/sidecar checkpoints are architecturally incompatible: "
            + "; ".join(mismatches)
            + layer_note
        )


# --- Injectable runtime (extension point; unused by any capable path today) -


class NativeMTPGenerationResponse(Protocol):
    """Subset of a hypothetical native-MTP generation response.

    No shipping MLX runtime exposes this today (see module docstring).
    This exists so the collector logic can be exercised end-to-end
    against a fake runtime and is ready to wrap a real one if a stable,
    metrics-differentiated API is ever published upstream.
    """

    text: str
    generation_tokens: int
    finish_reason: str | None
    accepted_block_tokens: int | None
    """Tokens accepted from one native-MTP verification block, if exposed."""
    proposed_block_tokens: int | None
    """Tokens proposed in one native-MTP verification block, if exposed."""


class NativeMTPRuntime(Protocol):
    """Injectable native-MTP boundary. No production adapter exists yet."""

    @property
    def mlx_version(self) -> str | None: ...

    @property
    def mlx_lm_version(self) -> str | None: ...

    def load_target(self, path: Path) -> tuple[Any, Any]: ...

    def load_sidecar(self, path: Path, target_model: Any) -> Any: ...

    def encode(self, tokenizer: Any, prompt: str) -> list[int]: ...

    def seed(self, seed: int) -> None: ...

    def synchronize(self) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def memory_snapshot(self) -> Any: ...

    def accelerator_name(self) -> str | None: ...

    def generate_with_native_mtp(
        self,
        target_model: Any,
        sidecar: Any,
        tokenizer: Any,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        configured_depth: int,
    ) -> Iterator[NativeMTPGenerationResponse]: ...


@dataclass(frozen=True)
class NativeMTPCollectionConfig:
    """Inputs for one local native-MTP collection attempt."""

    run_id: str
    target_model_path: Path
    mtp_sidecar_path: Path
    model_id: str
    prompt: str
    output_dir: Path
    command_argv: tuple[str, ...]
    max_tokens: int = 128
    seed: int = 0
    configured_depth: int = 2
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None
    accelerator: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise NativeMTPCollectorError("run_id must be non-empty")
        if not self.model_id:
            raise NativeMTPCollectorError("model_id must be non-empty")
        if not self.target_model_path.exists():
            raise NativeMTPCollectorError(
                f"target_model_path does not exist: {self.target_model_path}. "
                "Download or convert the model separately before collection."
            )
        if not self.mtp_sidecar_path.exists():
            raise NativeMTPCollectorError(
                f"mtp_sidecar_path does not exist: {self.mtp_sidecar_path}. "
                "Download or convert the sidecar separately before collection."
            )
        if not self.command_argv or not all(self.command_argv):
            raise NativeMTPCollectorError(
                "command_argv must contain non-empty argument strings"
            )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise NativeMTPCollectorError("max_tokens must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise NativeMTPCollectorError("seed must be an integer")
        if (
            isinstance(self.configured_depth, bool)
            or not isinstance(self.configured_depth, int)
            or self.configured_depth < 1
        ):
            raise NativeMTPCollectorError("configured_depth must be a positive integer")


@dataclass(frozen=True)
class NativeMTPCollectionResult:
    """Canonical record, capability report, and generated response text."""

    record: ExperimentRecord
    capability: NativeMTPCapabilityReport
    response_text: str = ""


def _config_hash_for(
    config: NativeMTPCollectionConfig, *, capability_supported: bool
) -> str:
    payload = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "tokenizer_revision": config.tokenizer_revision,
        "quantization": config.quantization,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "configured_depth": config.configured_depth,
        "capability_supported": capability_supported,
    }
    return config_hash(payload)


def capability_report_for_target(target_model_path: Path) -> NativeMTPCapabilityReport:
    """Read a target checkpoint's ``config.json`` and detect capability."""
    target_config = _read_config_json(target_model_path, label="target")
    model_family = str(target_config.get("model_type") or "unknown")
    return detect_native_mtp_capability(
        model_family,
        mlx_lm_version=_installed_version("mlx-lm"),
        mlx_vlm_version=_installed_version("mlx-vlm"),
    )


def collect_native_mtp(
    config: NativeMTPCollectionConfig,
    *,
    runtime: NativeMTPRuntime | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> NativeMTPCollectionResult:
    """Validate checkpoints, detect capability, and record honest evidence.

    When capability detection reports the runtime cannot produce
    trustworthy native-MTP evidence (true for every family verified
    against current upstream source -- see the module docstring), this
    writes an explicit failed ``ExperimentRecord`` plus a standalone
    ``capability_report.json`` rather than silently running generic
    draft-model speculation and mislabeling it as MTP.
    """
    started_at = utc_now_iso()
    total_started = clock()

    target_config = _read_config_json(config.target_model_path, label="target")
    sidecar_config = _read_config_json(config.mtp_sidecar_path, label="sidecar")
    validate_checkpoint_compatibility(target_config, sidecar_config)

    model_family = str(target_config.get("model_type") or "unknown")
    mlx_lm_version = _installed_version("mlx-lm")
    mlx_vlm_version = _installed_version("mlx-vlm")
    capability = detect_native_mtp_capability(
        model_family,
        mlx_lm_version=mlx_lm_version,
        mlx_vlm_version=mlx_vlm_version,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    capability.write_json(config.output_dir / "capability_report.json")

    platform_info = record_platform(
        accelerator=(
            config.accelerator
            if config.accelerator is not None
            else (runtime.accelerator_name() if runtime is not None else None)
        ),
        extra_packages=("mlx", "mlx-lm", "mlx-vlm"),
    )

    if not capability.supported:
        total_ended = clock()
        record = ExperimentRecord(
            run_id=config.run_id,
            started_at=started_at,
            ended_at=utc_now_iso(),
            platform=platform_info,
            model=ModelInfo(
                model_id=config.model_id,
                model_revision=config.model_revision,
                tokenizer_revision=config.tokenizer_revision,
                quantization=config.quantization,
                model_family=model_family,
            ),
            runtime=RuntimeInfo(
                name="mlx-lm",
                version=mlx_lm_version,
                backend="Metal",
                git_revision=None,
            ),
            command=CommandInfo(
                argv=config.command_argv,
                config_hash=_config_hash_for(config, capability_supported=False),
                workload_hash=sha256_text(config.prompt),
            ),
            repetition=RepetitionInfo(
                warmup_repetitions=0,
                measured_repetitions=1,
                repetition_index=0,
                seed=config.seed,
            ),
            timing=TimingMetrics(total=milliseconds(total_started, total_ended)),
            speculative=SpeculativeDecodingInfo(enabled=False, method=None),
            outcome=OutcomeInfo(success=False),
            error=ErrorInfo(
                category="NativeMTPUnsupported",
                message=capability.reason,
            ),
        )
        record.validate()
        record.write_json(config.output_dir / "record.json")
        return NativeMTPCollectionResult(record=record, capability=capability)

    # Capability path: no production runtime implements this today (see
    # module docstring). Exercised via NativeMTPRuntime fakes in tests so
    # this collector is ready to wrap a genuinely stable, distinguishable
    # native-MTP API if one is published upstream.
    if runtime is None:
        raise NativeMTPCollectorError(
            "capability detection reported native-MTP support for "
            f"'{model_family}', but no NativeMTPRuntime adapter was "
            "supplied to execute it"
        )

    return _collect_with_capable_runtime(
        config,
        runtime=runtime,
        clock=clock,
        capability=capability,
        platform_info=platform_info,
        started_at=started_at,
        total_started=total_started,
    )


def _collect_with_capable_runtime(
    config: NativeMTPCollectionConfig,
    *,
    runtime: NativeMTPRuntime,
    clock: Callable[[], float],
    capability: NativeMTPCapabilityReport,
    platform_info: PlatformInfo,
    started_at: str,
    total_started: float,
) -> NativeMTPCollectionResult:
    load_started: float | None = None
    load_ended: float | None = None
    tokenize_started: float | None = None
    tokenize_ended: float | None = None
    generation_started: float | None = None
    first_token_at: float | None = None
    generation_ended: float | None = None
    prompt_tokens: list[int] = []
    generated_tokens = 0
    accepted_tokens = 0
    proposed_tokens: int | None = None
    saw_any_proposed_signal = False
    response_parts: list[str] = []
    memory: Any = None
    error: ErrorInfo | None = None

    try:
        load_started = clock()
        target_model, tokenizer = runtime.load_target(config.target_model_path)
        sidecar = runtime.load_sidecar(config.mtp_sidecar_path, target_model)
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
        for response in runtime.generate_with_native_mtp(
            target_model,
            sidecar,
            tokenizer,
            prompt_tokens,
            max_tokens=config.max_tokens,
            configured_depth=config.configured_depth,
        ):
            observed_at = clock()
            if first_token_at is None:
                first_token_at = observed_at
            response_parts.append(response.text)
            is_eos_summary = response.finish_reason == "stop"
            if (
                response.generation_tokens > previous_generation_tokens
                and not is_eos_summary
            ):
                previous_generation_tokens = response.generation_tokens
                generated_tokens = response.generation_tokens
            if response.accepted_block_tokens is not None:
                accepted_tokens += response.accepted_block_tokens
            if response.proposed_block_tokens is not None:
                saw_any_proposed_signal = True
                proposed_tokens = (
                    proposed_tokens or 0
                ) + response.proposed_block_tokens

        runtime.synchronize()
        generation_ended = clock()
        memory = runtime.memory_snapshot()
    except (KeyError, RuntimeError, ValueError, OSError, MemoryError) as exc:
        generation_ended = clock()
        error = ErrorInfo(category=type(exc).__name__, message=str(exc))

    total_ended = clock()
    memory_active = getattr(memory, "active_bytes", None) if memory else None
    memory_cache = getattr(memory, "cache_bytes", None) if memory else None
    memory_peak = getattr(memory, "peak_bytes", None) if memory else None

    record = ExperimentRecord(
        run_id=config.run_id,
        started_at=started_at,
        ended_at=utc_now_iso(),
        platform=platform_info,
        model=ModelInfo(
            model_id=config.model_id,
            model_revision=config.model_revision,
            tokenizer_revision=config.tokenizer_revision,
            quantization=config.quantization,
            model_family=capability.model_family,
        ),
        runtime=RuntimeInfo(
            name="mlx-lm",
            version=runtime.mlx_lm_version,
            backend="Metal",
            git_revision=None,
        ),
        command=CommandInfo(
            argv=config.command_argv,
            config_hash=_config_hash_for(config, capability_supported=True),
            workload_hash=sha256_text(config.prompt),
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
            model_load=milliseconds(load_started, load_ended),
            tokenize=milliseconds(tokenize_started, tokenize_ended),
            prefill=milliseconds(generation_started, first_token_at),
            decode=milliseconds(first_token_at, generation_ended),
            total=milliseconds(total_started, total_ended),
        ),
        speculative=SpeculativeDecodingInfo(
            enabled=True,
            method="native-mtp",
            configured_depth=config.configured_depth,
            proposed_tokens=proposed_tokens if saw_any_proposed_signal else None,
            accepted_tokens=accepted_tokens if error is None else None,
            verification_time=None,
        ),
        memory=MemoryMetrics(
            active=bytes_measurement(memory_active),
            cache=bytes_measurement(memory_cache),
            peak=bytes_measurement(memory_peak),
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
    manifest = collect_environment_manifest(extra_packages=("mlx", "mlx-lm", "mlx-vlm"))
    atomic_write_text(config.output_dir / "environment.json", manifest.to_json() + "\n")
    return NativeMTPCollectionResult(
        record=record, capability=capability, response_text=response_text
    )
