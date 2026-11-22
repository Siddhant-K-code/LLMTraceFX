"""Load worker for Ollama's native ``/api/chat`` and ``/api/generate``.

The protocol, body and stream reading live in ``collectors.ollama`` so
``collect-ollama`` and ``sweep-ollama`` measure a request the same way.
Rows have the base fields of the OpenAI-compatible worker, so goodput and
the scheduling diagnostics read them unchanged, plus the server's own
timers from the final chunk.

Ollama does not report how many requests it runs at once
(``OLLAMA_NUM_PARALLEL``), so unlike ``sweep-llama-cpp`` this sweep cannot
refuse load above the server's capacity. A request above it waits inside
Ollama. That wait is in ``total_duration`` (``server_e2e_ms``) and in the
client's latency, but in no other server field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..collectors._shared import sha256_bytes, sha256_text
from ..collectors.ollama import (
    API_PATHS,
    TTFT_FIRST_OUTPUT_CHUNK,
    OllamaError,
    OllamaServerMetrics,
    OllamaSettings,
    OllamaStream,
    VersionProbe,
    parse_server_metrics,
    resolve_credential,
    stream_request,
    upstream_reference,
)
from ..collectors.openai_api import StreamingTransport
from ..collectors.server_reported import SOURCE_OLLAMA
from ..schema import MetricProvenance
from .api_worker import (
    ROW_FIELD_PROVENANCE,
    SERVER_REPORTED_ROW_FIELDS,
    ApiRequestSpec,
    RequestRow,
    _tpot,
)
from .dispatcher import DispatchContext
from .sweep import ATTRIBUTION

OLLAMA_FIELD_PROVENANCE: dict[str, str] = {
    **{
        k: v
        for k, v in ROW_FIELD_PROVENANCE.items()
        if k not in SERVER_REPORTED_ROW_FIELDS or k == "server_e2e_ms"
    },
    "server_load_ms": MetricProvenance.PROVIDER_REPORTED.value,
    "server_prompt_eval_ms": MetricProvenance.PROVIDER_REPORTED.value,
    "server_eval_ms": MetricProvenance.PROVIDER_REPORTED.value,
}
"""Base row fields keep the HTTP worker's labels. ``output_tokens`` is
``eval_count``, ``prompt_tokens`` is ``prompt_eval_count`` and
``cached_prompt_tokens`` is ``prompt_eval_cached_count``. ``server_e2e_ms``
is ``total_duration``, ``server_load_ms`` is ``load_duration``,
``server_prompt_eval_ms`` is ``prompt_eval_duration`` and
``server_eval_ms`` is ``eval_duration``, all converted from nanoseconds.
The other ``server_*`` fields of the HTTP worker are never filled here and
carry no label."""


@dataclass(frozen=True)
class OllamaRequestRow(RequestRow):
    """A ``RequestRow`` with Ollama's own timers from the final chunk.

    ``finish_reason`` holds the raw ``done_reason``. The server timers are
    not the client's TTFT and decode time, which also include the network
    and any wait the server's fields do not cover.
    """

    server_load_ms: float | None = None
    server_prompt_eval_ms: float | None = None
    server_eval_ms: float | None = None


def backend_block(
    settings: OllamaSettings, version: VersionProbe | None
) -> dict[str, Any]:
    return {
        "name": "ollama",
        "api": f"native {API_PATHS[settings.api]}, not OpenAI-compatible",
        "model": settings.model,
        "upstream_reference": upstream_reference(),
        "server": (
            version.to_dict()
            if version is not None
            else {"status": "not probed", "reason": "dry run; nothing was sent"}
        ),
        "max_supported_in_flight": None,
        "concurrency_limit": (
            "not checked: Ollama does not report OLLAMA_NUM_PARALLEL. Requests "
            "above it wait inside Ollama; the wait shows in server_e2e_ms "
            "(total_duration) and the client latency, in no other server field"
        ),
        "template_applied_by_server": True,
        "think": settings.think,
    }


def ollama_attribution(backend: Mapping[str, Any]) -> dict[str, Any]:
    attribution = dict(ATTRIBUTION)
    attribution["cannot_claim"] = [
        *ATTRIBUTION["cannot_claim"],
        (
            "that load stayed within the server's parallel capacity: Ollama "
            "does not report OLLAMA_NUM_PARALLEL, so excess requests queue "
            "inside it without a refusal"
        ),
        (
            "that the model stayed loaded: compare server_load_ms across "
            "requests, since a reload or eviction lands there"
        ),
    ]
    attribution["backend"] = dict(backend)
    return attribution


def make_ollama_sender(
    settings: OllamaSettings,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
    run_id: str,
) -> Callable[[ApiRequestSpec, DispatchContext], OllamaRequestRow]:
    """A thread-safe ``send`` for ``dispatch`` that streams one request.

    ``transport`` must be safe to use from several threads at once.
    ``UrllibStreamingTransport`` is: it opens a new connection per request.
    Specs with a system prompt, ``max_tokens``, ``extra_body`` or history
    are refused; the body is fixed by ``settings`` so every request of a
    sweep is alike.
    """
    credential = resolve_credential(settings, environ)

    def send(spec: ApiRequestSpec, context: DispatchContext) -> OllamaRequestRow:
        if (
            spec.system_prompt is not None
            or spec.max_tokens is not None
            or spec.extra_body is not None
            or spec.history
        ):
            raise OllamaError(
                "Ollama sweep specs take only a prompt and a run id suffix"
            )
        suffix = (
            spec.run_id_suffix
            if spec.run_id_suffix is not None
            else f"{context.index:06d}"
        )
        stream, body = stream_request(
            settings,
            prompt=spec.prompt,
            system_prompt=None,
            transport=transport,
            credential=credential,
            clock=context.clock,
            keep_text=False,
        )
        metrics = None if stream.final is None else parse_server_metrics(stream.final)
        return _row(
            stream,
            metrics,
            context,
            run_id=f"{run_id}-{suffix}",
            prompt=spec.prompt,
            workload_hash=sha256_bytes(body),
        )

    return send


def _row(
    stream: OllamaStream,
    metrics: OllamaServerMetrics | None,
    context: DispatchContext,
    *,
    run_id: str,
    prompt: str,
    workload_hash: str,
) -> OllamaRequestRow:
    end = stream.final_at if stream.final_at is not None else stream.end
    assert end is not None
    ttft_ms = stream.ms(stream.first_output_at)
    e2e_ms = stream.ms(end)
    output_tokens = None if metrics is None else metrics.count("eval_count")
    tpot_ms, tpot_status = _tpot(ttft_ms, e2e_ms, output_tokens)
    server: dict[str, Any] = {}
    if metrics is not None and metrics.reported:
        server = {
            "server_metrics_source": SOURCE_OLLAMA,
            "server_e2e_ms": metrics.ms("total_duration"),
            "server_load_ms": metrics.ms("load_duration"),
            "server_prompt_eval_ms": metrics.ms("prompt_eval_duration"),
            "server_eval_ms": metrics.ms("eval_duration"),
        }
    return OllamaRequestRow(
        index=context.index,
        sent=True,
        run_id=run_id,
        scheduled_offset_s=context.scheduled_offset_s,
        dispatched_offset_s=context.dispatched_offset_s,
        completed_offset_s=end - context.clock_origin,
        in_flight_at_dispatch=context.in_flight_at_dispatch,
        success=stream.failure is None,
        failure_category=stream.failure,
        http_status=stream.status,
        finish_reason=stream.done_reason(),
        finish_reason_code=stream.finish_code(),
        ttft_ms=ttft_ms,
        ttft_source=None if ttft_ms is None else TTFT_FIRST_OUTPUT_CHUNK,
        first_content_ms=stream.ms(stream.first_content_at),
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        prompt_tokens=None if metrics is None else metrics.count("prompt_eval_count"),
        cached_prompt_tokens=(
            None if metrics is None else metrics.count("prompt_eval_cached_count")
        ),
        tpot_ms=tpot_ms,
        tpot_status=tpot_status,
        content_delta_count=stream.content_chunks,
        prompt_sha256=sha256_text(prompt),
        workload_hash=workload_hash,
        **server,
    )


__all__ = [
    "OLLAMA_FIELD_PROVENANCE",
    "OllamaRequestRow",
    "backend_block",
    "make_ollama_sender",
    "ollama_attribution",
]
