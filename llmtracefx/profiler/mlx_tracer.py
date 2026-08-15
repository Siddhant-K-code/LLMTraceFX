"""Small, optional MLX tracer that writes LLMTraceFX-compatible JSON."""

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PathLike = str | Path


class MLXTraceRecorder:
    """Record synchronized MLX operations at token granularity.

    MLX evaluates arrays lazily. :meth:`measure` evaluates the returned arrays
    and synchronizes the device so the recorded duration includes GPU work.
    The optional Metal capture is useful for inspecting the same region in
    Xcode while the JSON output can be analyzed directly by LLMTraceFX.
    """

    def __init__(
        self,
        output_path: PathLike,
        *,
        metal_capture_path: PathLike | None = None,
        mlx_module: Any | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ):
        self.output_path = Path(output_path)
        self.metal_capture_path = (
            Path(metal_capture_path) if metal_capture_path is not None else None
        )
        self._mlx = mlx_module if mlx_module is not None else self._load_mlx()
        self._clock_ns = clock_ns
        self._started_ns: int | None = None
        self._active_token: dict[str, Any] | None = None
        self._tokens: list[dict[str, Any]] = []
        self._metal_capture_active = False

    @staticmethod
    def _load_mlx() -> Any:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MLX tracing requires MLX. On Apple Silicon run "
                "`uv sync --extra mlx` or `pip install mlx`."
            ) from exc
        return mx

    def __enter__(self) -> "MLXTraceRecorder":
        self._started_ns = self._clock_ns()
        if self.metal_capture_path is not None:
            if self.metal_capture_path.exists():
                raise FileExistsError(
                    f"Metal capture path already exists: {self.metal_capture_path}"
                )
            self.metal_capture_path.parent.mkdir(parents=True, exist_ok=True)
            metal = self._metal_api()
            is_available = getattr(metal, "is_available", None)
            if is_available is not None and not is_available():
                raise RuntimeError("Metal capture requires an available Metal backend")
            metal.start_capture(str(self.metal_capture_path))
            self._metal_capture_active = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._metal_capture_active:
            self._metal_api().stop_capture()
            self._metal_capture_active = False
        if exc_type is None:
            self.write()

    @contextmanager
    def token(self, token_id: int, token_text: str = "") -> Iterator[None]:
        """Group measured operations under one generated token."""
        self._ensure_started()
        if self._active_token is not None:
            raise RuntimeError("Nested token traces are not supported")

        token = {
            "id": token_id,
            "text": token_text,
            "operations": [],
        }
        self._active_token = token
        try:
            yield
        finally:
            self._tokens.append(token)
            self._active_token = None

    @contextmanager
    def operation(
        self, name: str, *, metadata: dict[str, Any] | None = None
    ) -> Iterator[None]:
        """Time a block after synchronizing pending MLX work.

        Call ``mx.eval(...)`` inside the block when it creates lazy arrays, or
        use :meth:`measure`, which evaluates returned arrays automatically.
        """
        self._ensure_active_token()
        self._synchronize()
        started_ns = self._clock_ns()
        try:
            yield
        finally:
            self._synchronize()
            ended_ns = self._clock_ns()
            self._record_operation(name, started_ns, ended_ns, metadata)

    def measure(
        self,
        name: str,
        function: Callable[..., Any],
        *args: Any,
        evaluate: bool = True,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run, evaluate, synchronize, and record one MLX operation."""
        self._ensure_active_token()
        self._synchronize()
        started_ns = self._clock_ns()
        result = function(*args, **kwargs)
        if evaluate:
            self._evaluate(result)
        self._synchronize()
        ended_ns = self._clock_ns()
        self._record_operation(name, started_ns, ended_ns, metadata)
        return result

    def write(self) -> Path:
        """Write the current trace and return its path."""
        if self._active_token is not None:
            raise RuntimeError("Finish the active token before writing the trace")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "llmtracefx.mlx.v1",
            "framework": "mlx",
            "hardware": "MLX",
            "device": self._device_info(),
            "time_unit": "ms",
            "tokens": self._tokens,
        }
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.output_path

    def _record_operation(
        self,
        name: str,
        started_ns: int,
        ended_ns: int,
        metadata: dict[str, Any] | None,
    ) -> None:
        token = self._ensure_active_token()
        operation_metadata = dict(metadata or {})
        operation_metadata.update(self._memory_snapshot())
        operation_metadata.setdefault("backend", "metal")

        origin_ns = self._started_ns if self._started_ns is not None else started_ns
        token["operations"].append(
            {
                "name": name,
                "start_time": (started_ns - origin_ns) / 1_000_000,
                "duration": max(0, ended_ns - started_ns) / 1_000_000,
                "metadata": operation_metadata,
            }
        )

    def _evaluate(self, result: Any) -> None:
        # mx.eval accepts an arbitrarily nested list, tuple, or dict of arrays.
        # Passing the tree intact also lets MLX ignore non-array leaves.
        self._mlx.eval(result)

    def _synchronize(self) -> None:
        synchronize = getattr(self._mlx, "synchronize", None)
        if synchronize is None:
            raise RuntimeError("This MLX version does not expose mx.synchronize()")
        synchronize()

    def _device_info(self) -> dict[str, Any]:
        device_info = getattr(self._mlx, "device_info", None)
        if device_info is not None:
            return self._json_safe_dict(device_info())
        metal = self._metal_api()
        device_info = getattr(metal, "device_info", None)
        return self._json_safe_dict(device_info()) if device_info else {}

    def _memory_snapshot(self) -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for function_name, output_name in (
            ("get_active_memory", "active_memory_bytes"),
            ("get_peak_memory", "peak_memory_bytes"),
            ("get_cache_memory", "cache_memory_bytes"),
        ):
            function = getattr(self._mlx, function_name, None)
            if function is not None:
                snapshot[output_name] = int(function())
        return snapshot

    def _metal_api(self) -> Any:
        metal = getattr(self._mlx, "metal", None)
        if metal is None:
            raise RuntimeError(
                "The active MLX build does not provide the Metal backend"
            )
        return metal

    def _ensure_started(self) -> None:
        if self._started_ns is None:
            self._started_ns = self._clock_ns()

    def _ensure_active_token(self) -> dict[str, Any]:
        self._ensure_started()
        if self._active_token is None:
            raise RuntimeError("Start a token trace with `with trace.token(...)`")
        return self._active_token

    @staticmethod
    def _json_safe_dict(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        safe = {}
        for key, item in value.items():
            if isinstance(item, (str, int, float, bool)) or item is None:
                safe[str(key)] = item
            else:
                safe[str(key)] = str(item)
        return safe
