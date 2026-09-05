"""MLX-LM local cache adapter for the cache-audit harness.

This module drives the *real* ``mlx_lm.models.cache.LRUPromptCache`` (via an
injectable :class:`MLXCacheRuntime`) against exact ``RequestSpec`` token IDs,
while independently mirroring the same reuse policy through
:class:`~llmtracefx.cache_audit.expected.MLXCacheOracle` so every verdict
cross-checks engine-attested behavior against an evidence-based expectation
computed with no knowledge of what the engine actually did.

Nothing here imports ``mlx``/``mlx_lm`` at module import time: those packages
are optional, Apple-Silicon-only dependencies, and doing so would make this
module unimportable everywhere else. All real imports are deferred to
:class:`ProductionMLXRuntime`, which is only ever constructed (and only ever
actually imports MLX) when a caller needs the production runtime. Tests
inject a fake implementing :class:`MLXCacheRuntime` instead, so none of the
adapter behavior here requires MLX to be installed or a model to be
downloaded.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from llmtracefx.optimizer._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    read_bounded_regular_text,
)
from llmtracefx.optimizer.collectors._shared import atomic_write_text
from llmtracefx.optimizer.schema import (
    Measurement,
    MetricProvenance,
    SchemaValidationError,
)

from ..expected import MLXCacheOracle
from ..schema import (
    CacheStateSnapshot,
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    MemoryEvidence,
    OutputEvidence,
    RequestEvidence,
    RequestSpec,
    ReuseEvidence,
    ScenarioKind,
    TerminalState,
    TimingEvidence,
    unavailable,
)
from ..verdicts import classify_request
from .base import CacheAuditCapability

#: Pinned by ``pyproject.toml``'s ``mlx`` extra; a capability check refuses
#: any other installed version rather than assuming compatible behavior.
REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_MLX_LM_VERSION = "0.31.3"

#: Exact symbols this adapter depends on, in the order they are probed.
REQUIRED_SYMBOLS = (
    "mlx_lm.models.cache.LRUPromptCache",
    "mlx_lm.models.cache.can_trim_prompt_cache",
    "mlx_lm.stream_generate",
    "mlx.core.synchronize",
    "mlx.core.get_active_memory",
    "mlx.core.get_peak_memory",
    "mlx.core.get_cache_memory",
    "mlx.core.reset_peak_memory",
)

_OBSERVABLE_FACTS = (
    "engine_cached_tokens",
    "engine_created_tokens",
    "observed_prompt_tokens",
    "output_token_ids",
    "baseline_token_ids",
    "runtime_active_bytes",
    "runtime_peak_bytes",
    "allocator_cache_bytes",
    "logical_cache_bytes",
    "in_process_first_token",
    "total",
)
_UNAVAILABLE_FACTS = (
    "reusable_blocks",
    "partial_block_tokens",
    "engine_cached_blocks",
    "physical_cache_blocks",
    "preemption_observed",
    "client_ttft",
    "queue",
    "scheduling",
    "prefill",
    "decode",
)


class MLXCacheAdapterError(RuntimeError):
    """Raised when the MLX cache adapter cannot proceed safely."""


@dataclass(frozen=True)
class MLXGenerationStep:
    """One token yielded by :meth:`MLXCacheRuntime.generate`."""

    token: int
    finish_reason: str | None = None


class MLXCacheRuntime(Protocol):
    """Injectable MLX-LM boundary consumed by :class:`MLXLocalCacheAdapter`.

    Every method is a direct analogue of one ``mlx``/``mlx_lm`` call so tests
    can substitute a fake without downloading a model or requiring Apple
    Silicon, while :class:`ProductionMLXRuntime` only ever calls the real
    library.
    """

    @property
    def platform_system(self) -> str: ...

    @property
    def platform_machine(self) -> str: ...

    @property
    def mlx_version(self) -> str | None: ...

    @property
    def mlx_lm_version(self) -> str | None: ...

    def missing_required_symbols(self) -> tuple[str, ...]:
        """Return the subset of REQUIRED_SYMBOLS this runtime cannot resolve."""
        ...

    def load_model(self, path: Path) -> tuple[Any, Any, Hashable]:
        """Load a local checkpoint. Must never fetch from a remote hub."""
        ...

    def synchronize(self) -> None: ...

    def reset_peak_memory(self) -> None: ...

    def active_memory(self) -> int: ...

    def peak_memory(self) -> int: ...

    def cache_memory(self) -> int: ...

    def make_cache(self, model: Any) -> Any:
        """Mirror ``mlx_lm.models.cache.make_prompt_cache`` for a fresh cache."""
        ...

    def fetch(
        self, model_key: Hashable, tokens: Sequence[int]
    ) -> tuple[Any | None, tuple[int, ...]]:
        """Mirror ``LRUPromptCache.fetch_nearest_cache``: returns (cache, rest)."""
        ...

    def insert(self, model_key: Hashable, tokens: Sequence[int], cache: Any) -> None:
        """Mirror ``LRUPromptCache.insert_cache``."""
        ...

    def cache_nbytes(self, cache: Any) -> int: ...

    def cache_len(self, cache: Any) -> int: ...

    def cache_can_trim(self, cache: Any) -> bool: ...

    def cache_classes(self, cache: Any) -> tuple[str, ...]: ...

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        cache: Any,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        prompt_progress_callback: Callable[[int, int], None],
    ) -> Iterator[MLXGenerationStep]:
        """Mirror ``mlx_lm.stream_generate`` bound to ``cache``."""
        ...


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _default_symbol_probe() -> tuple[str, ...]:
    """Import the exact MLX-LM symbols this adapter depends on.

    Returns the subset of :data:`REQUIRED_SYMBOLS` that could not be
    resolved. Imports happen lazily and only from here so importing this
    module never requires MLX to be installed.
    """

    missing: list[str] = []
    try:
        import mlx.core as mx
    except ImportError:
        return REQUIRED_SYMBOLS
    for name in (
        "synchronize",
        "get_active_memory",
        "get_peak_memory",
        "get_cache_memory",
        "reset_peak_memory",
    ):
        if getattr(mx, name, None) is None:
            missing.append(f"mlx.core.{name}")
    try:
        from mlx_lm import stream_generate  # noqa: F401
    except ImportError:
        missing.append("mlx_lm.stream_generate")
    try:
        from mlx_lm.models.cache import (  # noqa: F401
            LRUPromptCache,
            can_trim_prompt_cache,
        )
    except ImportError:
        missing.append("mlx_lm.models.cache.LRUPromptCache")
        missing.append("mlx_lm.models.cache.can_trim_prompt_cache")
    return tuple(missing)


def check_mlx_capabilities(
    runtime: MLXCacheRuntime, *, backend: str = "mlx_lm_local"
) -> CacheAuditCapability:
    """Fail-closed capability check with deterministic, ordered reasons."""

    reasons: list[str] = []
    if runtime.platform_system != "Darwin":
        reasons.append("platform_system_not_darwin")
    if runtime.platform_machine != "arm64":
        reasons.append("platform_machine_not_arm64")
    if runtime.mlx_version != REQUIRED_MLX_VERSION:
        reasons.append(
            "mlx_version_mismatch:"
            f"required={REQUIRED_MLX_VERSION}:installed={runtime.mlx_version!r}"
        )
    if runtime.mlx_lm_version != REQUIRED_MLX_LM_VERSION:
        reasons.append(
            "mlx_lm_version_mismatch:"
            f"required={REQUIRED_MLX_LM_VERSION}:installed={runtime.mlx_lm_version!r}"
        )
    for symbol in runtime.missing_required_symbols():
        reasons.append(f"missing_symbol:{symbol}")

    supported = not reasons
    return CacheAuditCapability(
        backend=backend,
        supported=supported,
        reasons=tuple(reasons),
        observable_facts=_OBSERVABLE_FACTS if supported else (),
        unavailable_facts=_UNAVAILABLE_FACTS,
    )


class ProductionMLXRuntime:
    """Talks to the real ``mlx`` / ``mlx_lm`` packages on Apple Silicon.

    Construction never imports MLX. Imports are deferred to the first call
    that actually needs them, so this class can be constructed (and its
    version/symbol-reporting methods used) even where MLX is not installed,
    letting :func:`check_mlx_capabilities` fail closed with precise reasons
    instead of an import-time crash.
    """

    def __init__(
        self, *, max_cache_entries: int = 10, max_cache_bytes: int = 1 << 63
    ) -> None:
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._mx: Any | None = None
        self._mlx_lm: Any | None = None
        self._prompt_cache: Any | None = None

    @property
    def platform_system(self) -> str:
        return platform.system()

    @property
    def platform_machine(self) -> str:
        return platform.machine()

    @property
    def mlx_version(self) -> str | None:
        return _installed_version("mlx")

    @property
    def mlx_lm_version(self) -> str | None:
        return _installed_version("mlx-lm")

    def missing_required_symbols(self) -> tuple[str, ...]:
        return _default_symbol_probe()

    def _ready(self) -> tuple[Any, Any]:
        if self._mx is None or self._mlx_lm is None:
            missing = self.missing_required_symbols()
            if missing:
                raise MLXCacheAdapterError(
                    "MLX runtime is missing required symbols: " + ", ".join(missing)
                )
            import mlx.core as mx
            import mlx_lm

            self._mx = mx
            self._mlx_lm = mlx_lm
        return self._mx, self._mlx_lm

    def _cache(self) -> Any:
        if self._prompt_cache is None:
            self._ready()
            from mlx_lm.models.cache import LRUPromptCache

            self._prompt_cache = LRUPromptCache(
                max_size=self._max_cache_entries,
                max_bytes=self._max_cache_bytes,
            )
        return self._prompt_cache

    def load_model(self, path: Path) -> tuple[Any, Any, Hashable]:
        if not path.is_dir():
            raise MLXCacheAdapterError(f"model_path must be a local directory: {path}")
        _, mlx_lm = self._ready()
        model, tokenizer = mlx_lm.load(str(path), lazy=False)
        return model, tokenizer, str(path)

    def synchronize(self) -> None:
        mx, _ = self._ready()
        mx.synchronize()

    def reset_peak_memory(self) -> None:
        mx, _ = self._ready()
        mx.reset_peak_memory()

    def active_memory(self) -> int:
        mx, _ = self._ready()
        return int(mx.get_active_memory())

    def peak_memory(self) -> int:
        mx, _ = self._ready()
        return int(mx.get_peak_memory())

    def cache_memory(self) -> int:
        mx, _ = self._ready()
        return int(mx.get_cache_memory())

    def make_cache(self, model: Any) -> Any:
        self._ready()
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(model)

    def fetch(
        self, model_key: Hashable, tokens: Sequence[int]
    ) -> tuple[Any | None, tuple[int, ...]]:
        cache, rest = self._cache().fetch_nearest_cache(model_key, list(tokens))
        return cache, tuple(rest)

    def insert(self, model_key: Hashable, tokens: Sequence[int], cache: Any) -> None:
        self._cache().insert_cache(model_key, list(tokens), cache)

    def cache_nbytes(self, cache: Any) -> int:
        return int(sum(int(layer.nbytes) for layer in cache))

    def cache_len(self, cache: Any) -> int:
        return len(cache)

    def cache_can_trim(self, cache: Any) -> bool:
        self._ready()
        from mlx_lm.models.cache import can_trim_prompt_cache

        return bool(can_trim_prompt_cache(cache))

    def cache_classes(self, cache: Any) -> tuple[str, ...]:
        return tuple(type(layer).__name__ for layer in cache)

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        cache: Any,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
        prompt_progress_callback: Callable[[int, int], None],
    ) -> Iterator[MLXGenerationStep]:
        _, mlx_lm = self._ready()
        for response in mlx_lm.stream_generate(
            model,
            tokenizer,
            list(prompt_tokens),
            max_tokens=max_tokens,
            prompt_cache=cache,
            prompt_progress_callback=prompt_progress_callback,
        ):
            yield MLXGenerationStep(
                token=int(response.token), finish_reason=response.finish_reason
            )


@dataclass(frozen=True)
class SavedCacheIdentity:
    """Sidecar identity bound to one saved MLX prompt-cache file.

    ``mlx_lm.models.cache.save_prompt_cache``/``load_prompt_cache`` persist
    raw KV tensors in a ``.safetensors`` payload; nothing in that payload
    proves which model, MLX build, or prompt produced it. This sidecar is
    written next to the cache file and MUST be verified before the payload
    is ever loaded, so a cache saved for one binding can never be silently
    reused for another.
    """

    model_key_digest: str
    model_artifact_digest: str
    cache_payload_digest: str
    mlx_version: str
    mlx_lm_version: str
    token_ids_digest: str

    @staticmethod
    def _digest(text: str) -> str:
        payload = "llmtracefx-cache-identity-v1\0" + text
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def for_binding(
        cls,
        *,
        model_key: Hashable,
        model_artifact_digest: str,
        cache_payload_digest: str,
        mlx_version: str,
        mlx_lm_version: str,
        token_ids: Sequence[int],
    ) -> SavedCacheIdentity:
        return cls(
            model_key_digest=cls._digest(repr(model_key)),
            model_artifact_digest=model_artifact_digest,
            cache_payload_digest=cache_payload_digest,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            token_ids_digest=cls._digest(",".join(str(token) for token in token_ids)),
        )

    @classmethod
    def for_cache_file(
        cls,
        cache_path: Path,
        *,
        model_key: Hashable,
        model_artifact_digest: str,
        mlx_version: str,
        mlx_lm_version: str,
        token_ids: Sequence[int],
    ) -> SavedCacheIdentity:
        return cls.for_binding(
            model_key=model_key,
            model_artifact_digest=model_artifact_digest,
            cache_payload_digest=_sha256_regular_file(cache_path),
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            token_ids=token_ids,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_key_digest": self.model_key_digest,
                "model_artifact_digest": self.model_artifact_digest,
                "cache_payload_digest": self.cache_payload_digest,
                "mlx_version": self.mlx_version,
                "mlx_lm_version": self.mlx_lm_version,
                "token_ids_digest": self.token_ids_digest,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> SavedCacheIdentity:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise SchemaValidationError("saved-cache identity must be an object")
        expected = {
            "model_key_digest",
            "model_artifact_digest",
            "cache_payload_digest",
            "mlx_version",
            "mlx_lm_version",
            "token_ids_digest",
        }
        if set(data) != expected:
            raise SchemaValidationError("saved-cache identity fields differ")
        values = {}
        for key in expected:
            value = data[key]
            if not isinstance(value, str) or not value:
                raise SchemaValidationError(
                    f"saved-cache identity {key} must be a non-empty string"
                )
            values[key] = value
        digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
        for key in (
            "model_key_digest",
            "model_artifact_digest",
            "cache_payload_digest",
            "token_ids_digest",
        ):
            if digest_pattern.fullmatch(values[key]) is None:
                raise SchemaValidationError(
                    f"saved-cache identity {key} is not a SHA-256 digest"
                )
        return cls(**values)


def _sidecar_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".identity.json")


def _sha256_regular_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise MLXCacheAdapterError(
            f"artifact must be a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_model_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = []
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise MLXCacheAdapterError(
                f"model directory must not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise MLXCacheAdapterError("model directory contains no artifact files")
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(
            bytes.fromhex(_sha256_regular_file(candidate).removeprefix("sha256:"))
        )
    return "sha256:" + digest.hexdigest()


def write_saved_cache_sidecar(cache_path: Path, identity: SavedCacheIdentity) -> Path:
    """Write ``identity`` next to ``cache_path`` without touching the cache file."""

    if cache_path.is_symlink() or not cache_path.is_file():
        raise MLXCacheAdapterError(
            f"saved cache must be a regular non-symlink file: {cache_path}"
        )
    if identity.cache_payload_digest != _sha256_regular_file(cache_path):
        raise MLXCacheAdapterError(
            "saved-cache identity does not match the cache payload digest"
        )
    sidecar = _sidecar_path(cache_path)
    if sidecar.is_symlink():
        raise MLXCacheAdapterError(
            f"saved-cache sidecar must not be a symlink: {sidecar}"
        )
    atomic_write_text(sidecar, identity.to_json() + "\n")
    return sidecar


def verify_saved_cache_sidecar(cache_path: Path, expected: SavedCacheIdentity) -> None:
    """Refuse to bind a saved cache whose sidecar identity doesn't match.

    Only the small sidecar JSON is opened here; ``cache_path`` itself (the
    ``.safetensors`` payload) is never read, so a missing or mismatched
    sidecar is rejected before any unbound tensor bytes would be loaded.
    """

    sidecar = _sidecar_path(cache_path)
    if cache_path.is_symlink() or not cache_path.is_file():
        raise MLXCacheAdapterError(
            f"saved cache must be a regular non-symlink file: {cache_path}"
        )
    if sidecar.is_symlink() or not sidecar.is_file():
        raise MLXCacheAdapterError(f"missing saved-cache identity sidecar: {sidecar}")
    try:
        actual = SavedCacheIdentity.from_json(
            read_bounded_regular_text(sidecar, MAX_METADATA_ARTIFACT_BYTES)
        )
    except (OSError, ValueError, KeyError) as exc:
        raise MLXCacheAdapterError(
            f"unreadable saved-cache identity sidecar: {sidecar}"
        ) from exc
    if actual.cache_payload_digest != _sha256_regular_file(cache_path):
        raise MLXCacheAdapterError(
            f"saved-cache payload digest mismatch for {cache_path}"
        )
    if actual != expected:
        raise MLXCacheAdapterError(
            f"saved-cache identity mismatch for {cache_path}: the sidecar does "
            "not match the currently bound model, MLX build, or token IDs"
        )


def _validate_local_model_path(model_path: str | Path) -> Path:
    """Accept only an existing local directory; never a URL or hub ID."""

    text = str(model_path)
    parsed = urlparse(text)
    if parsed.scheme not in ("", "file"):
        raise MLXCacheAdapterError(
            f"model_path must be a local directory, not a URL: {text!r}"
        )
    path = Path(parsed.path if parsed.scheme == "file" else text).expanduser()
    if not path.exists():
        raise MLXCacheAdapterError(f"model_path does not exist: {path}")
    if not path.is_dir():
        raise MLXCacheAdapterError(f"model_path must be a local directory: {path}")
    return path


def _fact(
    value: Any,
    basis: EvidenceBasis,
    source: str,
    *,
    scope: str = "request",
    limitations: tuple[str, ...] = (),
) -> EvidenceFact:
    return EvidenceFact(
        value=value,
        basis=basis,
        source=source,
        scope=scope,
        limitations=limitations,
    )


def _seconds(value: float) -> Measurement:
    return Measurement(
        value=value, provenance=MetricProvenance.MEASURED_WALL_CLOCK, unit="s"
    )


class MLXLocalCacheAdapter:
    """Drives a real MLX-LM ``LRUPromptCache`` against an injected runtime.

    Accepts either an already-loaded ``model``/``tokenizer``/``model_key``
    triple, or a local ``model_path`` directory that is loaded through the
    runtime. A remote URL or a path that does not already exist locally is
    rejected outright: this adapter never fetches a model.
    """

    def __init__(
        self,
        *,
        runtime: MLXCacheRuntime | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_key: Hashable | None = None,
        model_artifact_digest: str | None = None,
        model_path: str | Path | None = None,
        max_cache_entries: int = 10,
        max_cache_bytes: int = 1 << 63,
    ) -> None:
        loaded_provided = (
            model is not None or tokenizer is not None or model_key is not None
        )
        if loaded_provided and model_path is not None:
            raise MLXCacheAdapterError(
                "provide either an already-loaded model/tokenizer/model_key "
                "or a local model_path, not both"
            )

        self._runtime: MLXCacheRuntime = (
            runtime
            if runtime is not None
            else ProductionMLXRuntime(
                max_cache_entries=max_cache_entries,
                max_cache_bytes=max_cache_bytes,
            )
        )

        if model_path is not None:
            capability = check_mlx_capabilities(self._runtime, backend=self.backend)
            if not capability.supported:
                raise MLXCacheAdapterError(
                    "cannot load a local model_path: " + "; ".join(capability.reasons)
                )
            local_path = _validate_local_model_path(model_path)
            model_artifact_digest = _sha256_model_directory(local_path)
            model, tokenizer, model_key = self._runtime.load_model(local_path)
        elif model is not None and tokenizer is not None and model_key is not None:
            if (
                model_artifact_digest is None
                or re.fullmatch(r"sha256:[0-9a-f]{64}", model_artifact_digest) is None
            ):
                raise MLXCacheAdapterError(
                    "already-loaded models require a SHA-256 model_artifact_digest"
                )
        else:
            raise MLXCacheAdapterError(
                "provide an already-loaded model, tokenizer, and model_key, "
                "or a local model_path"
            )

        self._model = model
        self._tokenizer = tokenizer
        self._model_key: Hashable = model_key
        assert model_artifact_digest is not None
        self._model_artifact_digest = model_artifact_digest
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._oracle = MLXCacheOracle(
            max_entries=max_cache_entries, max_bytes=max_cache_bytes
        )
        self._evicted_entry_ids: set[str] = set()
        self._cache_classes: set[str] = set()

    @property
    def backend(self) -> str:
        return "mlx_lm_local"

    @property
    def model_key(self) -> Hashable:
        return self._model_key

    @property
    def model_artifact_digest(self) -> str:
        return self._model_artifact_digest

    def capabilities(self) -> CacheAuditCapability:
        return check_mlx_capabilities(self._runtime, backend=self.backend)

    def load_saved_cache(
        self,
        cache_path: Path,
        *,
        request: RequestSpec,
        loader: Callable[[Path], Any],
    ) -> Any:
        """Verify a saved cache's sidecar identity, then load it.

        ``loader`` (which would ultimately call something like
        ``mlx_lm.models.cache.load_prompt_cache``) is only ever invoked once
        the sidecar identity matches; a mismatch or missing sidecar raises
        before the underlying ``.safetensors`` payload is opened at all.
        """

        if request.input_token_ids is None:
            raise MLXCacheAdapterError(
                "cannot bind a saved cache without exact request token IDs"
            )
        expected = SavedCacheIdentity.for_cache_file(
            cache_path,
            model_key=self._model_key,
            model_artifact_digest=self._model_artifact_digest,
            mlx_version=self._runtime.mlx_version or "",
            mlx_lm_version=self._runtime.mlx_lm_version or "",
            token_ids=request.input_token_ids,
        )
        verify_saved_cache_sidecar(cache_path, expected)
        return loader(cache_path)

    def _runtime_key(self, namespace_id: str) -> Hashable:
        return (self._model_key, namespace_id)

    def _oracle_snapshot(self) -> CacheStateSnapshot:
        return CacheStateSnapshot(
            entry_count=_fact(
                self._oracle.entry_count,
                EvidenceBasis.INDEPENDENTLY_DERIVED,
                "oracle.resident_entries",
            ),
            logical_bytes=_fact(
                self._oracle.nbytes,
                EvidenceBasis.INDEPENDENTLY_DERIVED,
                "oracle.resident_bytes",
            ),
            valid_token_offsets=unavailable(
                self.backend, "native_cache_offsets_not_enumerable"
            ),
            cache_classes=(
                EvidenceFact(
                    value=sorted(self._cache_classes),
                    basis=EvidenceBasis.OBSERVED,
                    source="mlx.prompt_cache.layer_types",
                )
                if self._cache_classes
                else unavailable(self.backend, "cache_classes_not_yet_observed")
            ),
            complete=False,
        )

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]:
        capability = self.capabilities()
        if not capability.supported:
            raise MLXCacheAdapterError(
                "MLX runtime is unsupported: " + "; ".join(capability.reasons)
            )
        return [self._run_one(request) for request in requests]

    def _unsupported_record(self, spec: RequestSpec, *, code: str) -> RequestEvidence:
        record = RequestEvidence(
            spec=spec,
            reuse=ReuseEvidence(
                semantic_prefix_tokens=unavailable(self.backend, code),
                policy_reusable_tokens=unavailable(self.backend, code),
                reusable_blocks=unavailable(self.backend, code),
                partial_block_tokens=unavailable(self.backend, code),
                engine_cached_tokens=unavailable(self.backend, code),
                engine_cached_blocks=unavailable(self.backend, code),
                engine_created_tokens=unavailable(self.backend, code),
                observed_prompt_tokens=unavailable(self.backend, code),
                policy_required_prompt_tokens=unavailable(self.backend, code),
                unexpected_recomputed_tokens=unavailable(self.backend, code),
                eviction_observed=unavailable(self.backend, code),
                preemption_observed=unavailable(self.backend, code),
            ),
            timing=TimingEvidence(),
            memory=MemoryEvidence(
                runtime_active_bytes=unavailable(self.backend, code),
                runtime_peak_bytes=unavailable(self.backend, code),
                allocator_cache_bytes=unavailable(self.backend, code),
                logical_cache_bytes=unavailable(self.backend, code),
                physical_cache_blocks=unavailable(self.backend, code),
            ),
            output=OutputEvidence(
                output_token_ids=None,
                baseline_token_ids=None,
                token_identity=unavailable(self.backend, code),
                correctness=unavailable(self.backend, code),
                finish_reason=None,
            ),
            terminal_state=TerminalState.REFUSED,
            limitations=(Limitation(code=code, message=code, blocks_verdict=True),),
        )
        return classify_request(record)

    def _run_one(self, request: RequestSpec) -> RequestEvidence:
        if request.input_token_ids is None:
            raise MLXCacheAdapterError("MLX execution requires exact request token IDs")
        if request.scenario is ScenarioKind.QUANTIZED_CACHE:
            return self._unsupported_record(request, code="quantized_cache_unsupported")
        if request.scenario is ScenarioKind.ROTATING_CACHE:
            return self._unsupported_record(request, code="rotating_cache_unsupported")

        cache_before = self._oracle_snapshot()
        expectation = self._oracle.lookup(
            self._model_key, request.namespace_id, request.input_token_ids
        )
        if expectation.match_kind == "non_trimmable":
            return self._unsupported_record(
                request, code="non_trimmable_cache_reuse_unsupported"
            )
        runtime_key = self._runtime_key(request.namespace_id)
        cache, rest = self._runtime.fetch(runtime_key, request.input_token_ids)
        if cache is not None:
            self._cache_classes.update(self._runtime.cache_classes(cache))
        engine_cached_tokens = request.input_token_count - len(rest)

        if len(rest) == 0:
            return self._refuse_exact_empty_remainder(
                request, expectation, cache, engine_cached_tokens, cache_before
            )

        return self._execute(
            request,
            expectation,
            cache,
            rest,
            runtime_key,
            engine_cached_tokens,
            cache_before,
        )

    def _refuse_exact_empty_remainder(
        self,
        request: RequestSpec,
        expectation: Any,
        cache: Any,
        engine_cached_tokens: int,
        cache_before: CacheStateSnapshot,
    ) -> RequestEvidence:
        # MLX-LM's ``fetch_nearest_cache`` only returns an empty remainder for
        # an *exact* trie match. Resuming generation from that state requires
        # feeding the model at least one token to seed the next forward pass;
        # nothing observable tells us which token that should be, so this is
        # refused explicitly rather than guessing (e.g. by silently replaying
        # the last cached token, which would fabricate model input).
        code = "exact_empty_remainder_unsupported"
        record = RequestEvidence(
            spec=request,
            reuse=ReuseEvidence(
                semantic_prefix_tokens=_fact(
                    expectation.semantic_prefix_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.longest_common_prefix",
                ),
                policy_reusable_tokens=_fact(
                    expectation.policy_reusable_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.mlx_lru_policy",
                ),
                reusable_blocks=unavailable(self.backend, "token_granular_cache"),
                partial_block_tokens=unavailable(self.backend, "token_granular_cache"),
                engine_cached_tokens=_fact(
                    engine_cached_tokens,
                    EvidenceBasis.ENGINE_ATTESTED,
                    "prompt_cache.fetch_nearest_cache",
                ),
                engine_cached_blocks=unavailable(self.backend, "token_granular_cache"),
                engine_created_tokens=_fact(
                    0,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "adapter.uncached_remainder_length",
                ),
                observed_prompt_tokens=unavailable(self.backend, code),
                policy_required_prompt_tokens=_fact(
                    expectation.policy_required_prompt_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.mlx_lru_policy",
                ),
                unexpected_recomputed_tokens=unavailable(self.backend, code),
                eviction_observed=unavailable(self.backend, code),
                preemption_observed=unavailable(
                    self.backend, "preemption_not_implemented"
                ),
                prior_residency_observed=_fact(
                    cache is not None,
                    EvidenceBasis.OBSERVED,
                    "prompt_cache.fetch_nearest_cache",
                    scope="cache_namespace",
                ),
                residency_absence_observed=unavailable(
                    self.backend, "controlled_absence_probe_not_run"
                ),
            ),
            timing=TimingEvidence(),
            memory=MemoryEvidence(
                runtime_active_bytes=_fact(
                    self._runtime.active_memory(),
                    EvidenceBasis.OBSERVED,
                    "mlx.get_active_memory",
                    scope="process_global_allocator_gauge",
                ),
                runtime_peak_bytes=_fact(
                    self._runtime.peak_memory(),
                    EvidenceBasis.OBSERVED,
                    "mlx.get_peak_memory",
                    scope="process_global_allocator_gauge_since_reset",
                ),
                allocator_cache_bytes=_fact(
                    self._runtime.cache_memory(),
                    EvidenceBasis.OBSERVED,
                    "mlx.get_cache_memory",
                    scope="process_global_allocator_gauge",
                ),
                logical_cache_bytes=_fact(
                    self._runtime.cache_nbytes(cache) if cache is not None else 0,
                    EvidenceBasis.OBSERVED,
                    "prompt_cache.entry_nbytes",
                ),
                physical_cache_blocks=unavailable(self.backend, "token_granular_cache"),
            ),
            output=OutputEvidence(
                output_token_ids=None,
                baseline_token_ids=None,
                token_identity=unavailable(self.backend, code),
                correctness=unavailable(self.backend, code),
                finish_reason=None,
            ),
            terminal_state=TerminalState.REFUSED,
            limitations=(
                Limitation(
                    code=code,
                    message=(
                        "fetch_nearest_cache returned an exact match with an "
                        "empty remainder; refusing rather than fabricating a "
                        "seed token for generation"
                    ),
                    blocks_verdict=True,
                ),
            ),
            cache_before=cache_before,
            cache_after=cache_before,
        )
        return classify_request(record)

    def _execute(
        self,
        request: RequestSpec,
        expectation: Any,
        cache: Any,
        rest: tuple[int, ...],
        runtime_key: Hashable,
        engine_cached_tokens: int,
        cache_before: CacheStateSnapshot,
    ) -> RequestEvidence:
        assert request.input_token_ids is not None
        if cache is None:
            cache = self._runtime.make_cache(self._model)
        self._cache_classes.update(self._runtime.cache_classes(cache))

        progress: dict[str, int] = {"actual": 0, "total": 0}

        def _progress(processed: int, total: int) -> None:
            progress["actual"] = processed
            progress["total"] = total

        self._runtime.reset_peak_memory()
        wall_start = time.perf_counter()
        first_token_seconds: float | None = None
        output_tokens: list[int] = []
        finish_reason: str | None = None
        for step in self._runtime.generate(
            self._model,
            self._tokenizer,
            cache,
            rest,
            max_tokens=request.output_tokens,
            prompt_progress_callback=_progress,
        ):
            if first_token_seconds is None:
                self._runtime.synchronize()
                first_token_seconds = time.perf_counter() - wall_start
            output_tokens.append(step.token)
            finish_reason = step.finish_reason
            if len(output_tokens) >= request.output_tokens:
                break
        self._runtime.synchronize()
        total_seconds = time.perf_counter() - wall_start
        active_after = self._runtime.active_memory()
        peak_after = self._runtime.peak_memory()
        cache_after = self._runtime.cache_memory()

        full_sequence = tuple(request.input_token_ids) + tuple(output_tokens)
        cache_nbytes = self._runtime.cache_nbytes(cache)
        self._runtime.insert(runtime_key, full_sequence, cache)
        newly_evicted = self._oracle.insert(
            entry_id=request.request_id,
            model_key=self._model_key,
            namespace_id=request.namespace_id,
            tokens=full_sequence,
            nbytes=cache_nbytes,
            trimmable=self._runtime.cache_can_trim(cache),
        )
        self._evicted_entry_ids.update(newly_evicted)
        # Baseline: regenerate the exact same prompt from scratch on a fresh
        # cache. This is the only ground truth this harness has for output
        # correctness -- it proves cache reuse did not change what the model
        # would have produced without it.
        baseline_cache = self._runtime.make_cache(self._model)
        baseline_tokens: list[int] = []
        for step in self._runtime.generate(
            self._model,
            self._tokenizer,
            baseline_cache,
            request.input_token_ids,
            max_tokens=request.output_tokens,
            prompt_progress_callback=lambda *_: None,
        ):
            baseline_tokens.append(step.token)
            if len(baseline_tokens) >= request.output_tokens:
                break

        token_identity = tuple(output_tokens) == tuple(baseline_tokens)
        unexpected_recomputed = max(
            0, progress["actual"] - expectation.policy_required_prompt_tokens
        )

        record = RequestEvidence(
            spec=request,
            reuse=ReuseEvidence(
                semantic_prefix_tokens=_fact(
                    expectation.semantic_prefix_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.longest_common_prefix",
                ),
                policy_reusable_tokens=_fact(
                    expectation.policy_reusable_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.mlx_lru_policy",
                ),
                reusable_blocks=unavailable(self.backend, "token_granular_cache"),
                partial_block_tokens=unavailable(self.backend, "token_granular_cache"),
                engine_cached_tokens=_fact(
                    engine_cached_tokens,
                    EvidenceBasis.ENGINE_ATTESTED,
                    "prompt_cache.fetch_nearest_cache",
                ),
                engine_cached_blocks=unavailable(self.backend, "token_granular_cache"),
                engine_created_tokens=_fact(
                    len(rest),
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "adapter.uncached_remainder_length",
                ),
                observed_prompt_tokens=_fact(
                    progress["actual"],
                    EvidenceBasis.OBSERVED,
                    "stream_generate.prompt_progress_callback",
                ),
                policy_required_prompt_tokens=_fact(
                    expectation.policy_required_prompt_tokens,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.mlx_lru_policy",
                ),
                unexpected_recomputed_tokens=_fact(
                    unexpected_recomputed,
                    EvidenceBasis.INDEPENDENTLY_DERIVED,
                    "oracle.prompt_work_delta",
                ),
                eviction_observed=unavailable(
                    self.backend,
                    "native_eviction_or_controlled_absence_probe_unavailable",
                ),
                preemption_observed=unavailable(
                    self.backend, "preemption_not_implemented"
                ),
                prior_residency_observed=_fact(
                    engine_cached_tokens > 0,
                    EvidenceBasis.OBSERVED,
                    "prompt_cache.fetch_nearest_cache",
                    scope="cache_namespace",
                ),
                residency_absence_observed=unavailable(
                    self.backend, "controlled_absence_probe_not_run"
                ),
            ),
            timing=TimingEvidence(
                in_process_first_token=(
                    _seconds(first_token_seconds)
                    if first_token_seconds is not None
                    else None
                ),
                total=_seconds(total_seconds),
                scope="in_process_generation_section",
                exclusions=(
                    "prompt_cache_lookup",
                    "prompt_cache_insertion",
                    "no_cache_baseline",
                ),
            ),
            memory=MemoryEvidence(
                runtime_active_bytes=_fact(
                    active_after,
                    EvidenceBasis.OBSERVED,
                    "mlx.get_active_memory",
                    scope="process_global_allocator_gauge",
                ),
                runtime_peak_bytes=_fact(
                    peak_after,
                    EvidenceBasis.OBSERVED,
                    "mlx.get_peak_memory",
                    scope="process_global_allocator_gauge_since_reset",
                ),
                allocator_cache_bytes=_fact(
                    cache_after,
                    EvidenceBasis.OBSERVED,
                    "mlx.get_cache_memory",
                    scope="process_global_allocator_gauge",
                ),
                logical_cache_bytes=_fact(
                    cache_nbytes, EvidenceBasis.OBSERVED, "prompt_cache.entry_nbytes"
                ),
                physical_cache_blocks=unavailable(self.backend, "token_granular_cache"),
            ),
            output=OutputEvidence(
                output_token_ids=tuple(output_tokens),
                baseline_token_ids=tuple(baseline_tokens),
                token_identity=_fact(
                    token_identity,
                    EvidenceBasis.OBSERVED,
                    "baseline_regeneration.token_match",
                ),
                correctness=_fact(
                    token_identity,
                    EvidenceBasis.OBSERVED,
                    "baseline_regeneration.correctness",
                ),
                finish_reason=finish_reason,
            ),
            terminal_state=TerminalState.COMPLETED,
            cache_before=cache_before,
            cache_after=self._oracle_snapshot(),
        )
        return classify_request(record)
