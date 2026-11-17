"""Load worker for in-process mlx-lm generation.

``mlx_lm.server`` is OpenAI-compatible and ``sweep-api`` already drives it.
This worker covers the other case: one model loaded in this process with
``mlx_lm.load`` and driven through ``mlx_lm.stream_generate``, the same
path ``collect-mlx`` measures.

That path runs one generation at a time. Checked against
ml-explore/mlx-lm at :data:`MLX_LM_COMMIT` and ml-explore/mlx at
:data:`MLX_COMMIT`:

* ``mlx_lm.generate`` creates ``generation_stream`` with
  ``mx.new_thread_local_stream``, so the model must be loaded and run on
  one thread.
* ``mlx_lm.server`` itself runs every generation on one dedicated thread
  and batches requests there. ``stream_generate`` has no batching.

So the worker owns one thread that loads the model and runs every request,
and refuses a second request while one is running instead of queueing it.
The sweep command refuses any load that could put more than one request in
flight, so the refusal here is a guard, not a code path a sweep relies on.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..collectors._shared import sha256_text
from ..collectors.mlx import MLXRuntime
from ..collectors.openai_api import FAILURE_TIMEOUT
from ..schema import MetricProvenance
from .api_worker import ROW_FIELD_PROVENANCE, ApiRequestSpec, RequestRow, _tpot
from .dispatcher import DispatchContext
from .sweep import ATTRIBUTION, SweepPoint

MLX_LM_REPOSITORY = "ml-explore/mlx-lm"
MLX_LM_COMMIT = "c4c9c762cc797fd979fc98645f5db5112753bbe9"
MLX_REPOSITORY = "ml-explore/mlx"
MLX_COMMIT = "fbedcee5f4a0c7b2cbb642ea91d4c555460d3e48"

MAX_SUPPORTED_IN_FLIGHT = 1
SINGLE_STREAM_REASON = (
    "in-process mlx-lm runs one stream_generate call at a time on the thread "
    "that loaded the model (generation_stream is thread-local), and "
    "stream_generate does not batch requests"
)

FINISH_CODES: Mapping[str, str] = {"stop": "stop", "length": "length"}
FAILURE_RUNTIME_ERROR = "runtime_error"
FAILURE_NO_FINISH = "no_finish_reason"
TTFT_FIRST_COUNTED_TOKEN = "first_counted_token"

MLX_FIELD_PROVENANCE: dict[str, str] = {
    **{k: v for k, v in ROW_FIELD_PROVENANCE.items() if k != "cached_prompt_tokens"},
    "output_tokens": MetricProvenance.MEASURED_NATIVE.value,
    "prompt_tokens": MetricProvenance.MEASURED_NATIVE.value,
}
"""``output_tokens`` is counted from ``stream_generate`` responses with the
``collect-mlx`` rule and ``prompt_tokens`` is the length of the encoded
prompt; both come from the runtime itself. There is no prompt cache, so
``cached_prompt_tokens`` is always empty and has no label."""

_RUNTIME_ERRORS = (KeyError, RuntimeError, ValueError, OSError, MemoryError)
_MAX_LABEL_CHARS = 64


class MLXLoadError(ValueError):
    """An MLX load sweep cannot be configured or the model cannot be served."""


class MLXBusyError(RuntimeError):
    """A second request arrived while the single in-process stream was busy."""


@dataclass(frozen=True)
class MLXLoadConfig:
    run_id: str
    model_path: Path
    model_id: str
    max_tokens: int
    seed: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    request_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise MLXLoadError("--model-id must be non-empty")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise MLXLoadError("--max-output-tokens must be a positive integer")
        if self.temperature < 0:
            raise MLXLoadError("--temperature must be at least 0")
        if not 0 < self.top_p <= 1:
            raise MLXLoadError("--top-p must be greater than 0 and at most 1")
        if not self.request_timeout_seconds > 0:
            raise MLXLoadError("--request-timeout must be positive")


def request_settings(config: MLXLoadConfig) -> dict[str, Any]:
    """What every request runs with, without the prompt or the model path."""
    return {
        "api": "in-process mlx_lm.stream_generate (not OpenAI-compatible)",
        "model_id": config.model_id,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "seed_scope": "mx.random.seed is set before every request",
        "temperature": config.temperature,
        "top_p": config.top_p,
        "draft_model": None,
        "prompt_handling": "raw text encoded by the tokenizer, no chat template",
        "request_timeout_seconds": config.request_timeout_seconds,
        "timeout_scope": "checked between generated tokens",
        "upstream_reference": upstream_reference(),
    }


def upstream_reference() -> dict[str, Any]:
    return {
        "mlx_lm": {"repository": MLX_LM_REPOSITORY, "commit": MLX_LM_COMMIT},
        "mlx": {"repository": MLX_REPOSITORY, "commit": MLX_COMMIT},
    }


def _workload_hash(config: MLXLoadConfig, prompt: str) -> str:
    payload = {
        "model_id": config.model_id,
        "prompt": prompt,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_points(points: list[SweepPoint]) -> None:
    """Refuse any point that could have more than one request in flight."""
    for point in points:
        in_flight = point.policy.max_in_flight
        if in_flight > MAX_SUPPORTED_IN_FLIGHT:
            raise MLXLoadError(
                f"point {point.label} can have {in_flight} requests in flight; "
                f"in-process MLX supports {MAX_SUPPORTED_IN_FLIGHT} "
                f"({SINGLE_STREAM_REASON}). Use --concurrency 1, or "
                "--request-rate with --max-in-flight 1, or sweep "
                "mlx_lm.server with sweep-api"
            )


@dataclass(frozen=True)
class MLXRequestRow(RequestRow):
    """A ``RequestRow`` from in-process generation.

    ``http_status`` is always empty. ``runtime_error_type`` names the
    exception class when generation raised; its message is not kept.
    """

    runtime_error_type: str | None = None


@dataclass
class _Job:
    run: Callable[[], Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class MLXModelInfo:
    load_seconds: float
    mlx_version: str | None
    mlx_lm_version: str | None
    runtime_name: str
    runtime_version: str | None
    accelerator: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_load_seconds": self.load_seconds,
            "model_load_seconds_provenance": (
                MetricProvenance.MEASURED_WALL_CLOCK.value
            ),
            "mlx_version": self.mlx_version,
            "mlx_lm_version": self.mlx_lm_version,
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "accelerator": self.accelerator,
        }


class MLXSession:
    """One loaded model and the one thread allowed to touch it.

    :meth:`start` creates the runtime on the owner thread, loads the model
    there and waits for it. Every request then runs on that thread.
    """

    def __init__(
        self,
        runtime_factory: Callable[[], MLXRuntime],
        config: MLXLoadConfig,
        *,
        clock: Callable[[], float],
    ) -> None:
        self._factory = runtime_factory
        self._config = config
        self._clock = clock
        self._jobs: queue.SimpleQueue[_Job | None] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._busy = threading.Lock()
        self.runtime: MLXRuntime | None = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.info: MLXModelInfo | None = None

    def start(self) -> MLXModelInfo:
        if self._thread is not None:
            raise RuntimeError("MLXSession already started")
        self._thread = threading.Thread(
            target=self._serve, name="llmtracefx-mlx-owner", daemon=True
        )
        self._thread.start()
        info = self.call(self._load)
        if not isinstance(info, MLXModelInfo):
            raise RuntimeError("model load returned no model info")
        self.info = info
        return info

    def _load(self) -> MLXModelInfo:
        runtime = self._factory()
        started = self._clock()
        model, tokenizer = runtime.load_model(self._config.model_path)
        runtime.synchronize()
        load_seconds = self._clock() - started
        self.runtime, self.model, self.tokenizer = runtime, model, tokenizer
        return MLXModelInfo(
            load_seconds=load_seconds,
            mlx_version=runtime.mlx_version,
            mlx_lm_version=runtime.mlx_lm_version,
            runtime_name=runtime.runtime_name,
            runtime_version=runtime.runtime_version,
            accelerator=runtime.accelerator_name(),
        )

    def _serve(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job.result = job.run()
            except BaseException as exc:  # handed back to the caller's thread
                job.error = exc
            finally:
                job.done.set()

    def call(self, run: Callable[[], Any]) -> Any:
        """Run ``run`` on the owner thread and wait for it."""
        if self._thread is None:
            raise RuntimeError("MLXSession is not started")
        job = _Job(run)
        self._jobs.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    @property
    def owner_thread(self) -> threading.Thread | None:
        return self._thread

    def close(self) -> None:
        if self._thread is None:
            return
        self._jobs.put(None)
        self._thread.join()

    def acquire(self) -> bool:
        return self._busy.acquire(blocking=False)

    def release(self) -> None:
        self._busy.release()


@dataclass
class _Generation:
    start: float
    first_token_at: float | None = None
    first_content_at: float | None = None
    content_chunks: int = 0
    output_tokens: int = 0
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    end: float | None = None
    failure: str | None = None
    error_type: str | None = None


def make_mlx_sender(
    session: MLXSession, config: MLXLoadConfig
) -> Callable[[ApiRequestSpec, DispatchContext], MLXRequestRow]:
    """A ``send`` for ``dispatch`` that runs one generation on ``session``.

    Raises :class:`MLXBusyError` when a request is already running, so the
    dispatcher records a client error instead of this worker waiting in
    line. Specs with a system prompt, ``max_tokens`` or ``extra_body`` are
    refused; every request of a sweep runs with ``config``.
    """

    def send(spec: ApiRequestSpec, context: DispatchContext) -> MLXRequestRow:
        if (
            spec.system_prompt is not None
            or spec.max_tokens is not None
            or spec.extra_body is not None
        ):
            raise MLXLoadError("MLX load specs take only a prompt and a run id suffix")
        if not session.acquire():
            raise MLXBusyError(
                "a request is already running on the in-process MLX model; "
                f"it supports {MAX_SUPPORTED_IN_FLIGHT} at a time"
            )
        try:
            start = context.clock()
            generation = session.call(
                lambda: _generate(session, config, spec.prompt, start, context.clock)
            )
            if not isinstance(generation, _Generation):
                raise RuntimeError("generation returned no result")
        finally:
            session.release()
        suffix = (
            spec.run_id_suffix
            if spec.run_id_suffix is not None
            else f"{context.index:06d}"
        )
        return _row(
            generation,
            context,
            run_id=f"{config.run_id}-{suffix}",
            prompt=spec.prompt,
            workload_hash=_workload_hash(config, spec.prompt),
        )

    return send


def _generate(
    session: MLXSession,
    config: MLXLoadConfig,
    prompt: str,
    start: float,
    clock: Callable[[], float],
) -> _Generation:
    runtime = session.runtime
    if runtime is None:
        raise RuntimeError("MLXSession has no loaded model")
    generation = _Generation(start=start)
    deadline = start + config.request_timeout_seconds
    try:
        prompt_tokens = runtime.encode(session.tokenizer, prompt)
        generation.prompt_tokens = len(prompt_tokens)
        runtime.seed(config.seed)
        previous = 0
        for response in runtime.stream_generate(
            session.model,
            session.tokenizer,
            prompt_tokens,
            max_tokens=config.max_tokens,
            draft_model=None,
            num_draft_tokens=0,
        ):
            now = clock()
            if response.text:
                generation.content_chunks += 1
                if generation.first_content_at is None:
                    generation.first_content_at = now
            if (
                response.generation_tokens > previous
                and response.finish_reason != "stop"
            ):
                previous = response.generation_tokens
                generation.output_tokens = previous
                if generation.first_token_at is None:
                    generation.first_token_at = now
            if response.finish_reason is not None:
                generation.finish_reason = response.finish_reason[:_MAX_LABEL_CHARS]
            if now > deadline:
                generation.failure = FAILURE_TIMEOUT
                break
        runtime.synchronize()
    except _RUNTIME_ERRORS as exc:
        generation.failure = FAILURE_RUNTIME_ERROR
        generation.error_type = type(exc).__name__
    generation.end = clock()
    if generation.failure is None and generation.finish_reason not in FINISH_CODES:
        generation.failure = FAILURE_NO_FINISH
    return generation


def _row(
    generation: _Generation,
    context: DispatchContext,
    *,
    run_id: str,
    prompt: str,
    workload_hash: str,
) -> MLXRequestRow:
    def ms(at: float | None) -> float | None:
        return None if at is None else (at - generation.start) * 1000.0

    success = generation.failure is None
    end = generation.end if generation.end is not None else generation.start
    ttft_ms = ms(generation.first_token_at)
    e2e_ms = ms(end)
    output_tokens = generation.output_tokens if success else None
    tpot_ms, tpot_status = _tpot(ttft_ms, e2e_ms, output_tokens)
    return MLXRequestRow(
        index=context.index,
        sent=True,
        run_id=run_id,
        scheduled_offset_s=context.scheduled_offset_s,
        dispatched_offset_s=context.dispatched_offset_s,
        completed_offset_s=end - context.clock_origin,
        in_flight_at_dispatch=context.in_flight_at_dispatch,
        success=success,
        failure_category=generation.failure,
        http_status=None,
        finish_reason=generation.finish_reason,
        finish_reason_code=FINISH_CODES.get(generation.finish_reason or ""),
        ttft_ms=ttft_ms,
        ttft_source=None if ttft_ms is None else TTFT_FIRST_COUNTED_TOKEN,
        first_content_ms=ms(generation.first_content_at),
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        prompt_tokens=generation.prompt_tokens,
        cached_prompt_tokens=None,
        tpot_ms=tpot_ms,
        tpot_status=tpot_status,
        content_delta_count=generation.content_chunks,
        prompt_sha256=sha256_text(prompt),
        workload_hash=workload_hash,
        runtime_error_type=generation.error_type,
    )


def backend_block(config: MLXLoadConfig, info: MLXModelInfo | None) -> dict[str, Any]:
    return {
        "name": "mlx-lm",
        "api": "in-process stream_generate, not OpenAI-compatible",
        "upstream_reference": upstream_reference(),
        "model_id": config.model_id,
        "max_supported_in_flight": MAX_SUPPORTED_IN_FLIGHT,
        "max_supported_in_flight_reason": SINGLE_STREAM_REASON,
        "concurrency_limit": (
            "points that could have more than one request in flight are "
            "refused; a request that arrives while one is running is refused, "
            "not queued"
        ),
        "runtime": (
            info.to_dict()
            if info is not None
            else {"status": "not loaded", "reason": "dry run; no model was loaded"}
        ),
        "warmup": (
            "none; the first request of the sweep includes any one-time "
            "compile and cache cost"
        ),
        "chat_template_applied": False,
        "sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "seed": config.seed,
        },
    }


def mlx_attribution(backend: Mapping[str, Any]) -> dict[str, Any]:
    attribution = dict(ATTRIBUTION)
    attribution["can_claim"] = [
        (
            "the latency distribution and goodput of in-process mlx-lm "
            "generation for this request mix, at this offered load, on this "
            "machine, one request at a time"
        ),
        (
            "each per-request value is a real wall clock reading taken in the "
            "process that ran the model, with tokenization inside the TTFT"
        ),
    ]
    attribution["cannot_claim"] = [
        *ATTRIBUTION["cannot_claim"],
        (
            "any effect of server batching or concurrent requests; at most one "
            "request runs at a time"
        ),
        (
            "comparability with sweep-api rows from mlx_lm.server: no HTTP, no "
            "chat template and no batching here"
        ),
    ]
    attribution["backend"] = dict(backend)
    return attribution


__all__ = [
    "MAX_SUPPORTED_IN_FLIGHT",
    "MLX_COMMIT",
    "MLX_FIELD_PROVENANCE",
    "MLX_LM_COMMIT",
    "MLXBusyError",
    "MLXLoadConfig",
    "MLXLoadError",
    "MLXModelInfo",
    "MLXRequestRow",
    "MLXSession",
    "backend_block",
    "check_points",
    "make_mlx_sender",
    "mlx_attribution",
    "request_settings",
]
