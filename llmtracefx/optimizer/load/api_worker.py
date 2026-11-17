"""Bind the load dispatcher to the OpenAI-compatible stream collector.

Each dispatched request runs through ``stream_chat_completion``, the same
code path ``collect-api`` uses, and is reduced to a ``RequestRow``: timing,
token counts and outcome, with no prompt or response text. No per-request
``record.json`` is written. A request measured under concurrency shares the
server with others, so it must not leave an artifact that reads as an
isolated measurement.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..collectors._shared import sha256_text
from ..collectors.openai_api import (
    APICollectionConfig,
    APICollectionResult,
    OpenAIStreamCollectorError,
    StreamingTransport,
    _resolve_credential,
    assert_credential_not_embedded,
    build_request_plan,
    stream_chat_completion,
)
from ..schema import MetricProvenance
from .dispatcher import DispatchContext, DispatchOutcome, DispatchReport

_SUFFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_OUTPUT_EVENT_KINDS = frozenset({"content", "reasoning"})

TPOT_DERIVED = "derived"
TPOT_NOT_APPLICABLE = "not_applicable"
"""One output token or none: there is no inter-token interval to average."""
TPOT_UNAVAILABLE = "unavailable"
"""TTFT, E2E or the provider's output token count is missing."""

TTFT_EARLIEST_OUTPUT_EVENT = "earliest_output_event"
TTFT_FIRST_CONTENT = "first_content_token_offset"

ROW_FIELD_PROVENANCE: dict[str, str] = {
    "scheduled_offset_s": MetricProvenance.DERIVED.value,
    "dispatched_offset_s": MetricProvenance.MEASURED_WALL_CLOCK.value,
    "completed_offset_s": MetricProvenance.MEASURED_WALL_CLOCK.value,
    "ttft_ms": MetricProvenance.MEASURED_WALL_CLOCK.value,
    "first_content_ms": MetricProvenance.MEASURED_WALL_CLOCK.value,
    "e2e_ms": MetricProvenance.MEASURED_WALL_CLOCK.value,
    "output_tokens": MetricProvenance.PROVIDER_REPORTED.value,
    "prompt_tokens": MetricProvenance.PROVIDER_REPORTED.value,
    "cached_prompt_tokens": MetricProvenance.PROVIDER_REPORTED.value,
    "tpot_ms": MetricProvenance.DERIVED.value,
}
"""How each ``RequestRow`` value was obtained. Offsets are on the
dispatcher's clock; latencies are measured from the moment the collector
sent the request. Labels and counters (``http_status``,
``finish_reason``, ``in_flight_at_dispatch``) are observations, not
metrics, and carry no provenance. ``tpot_ms`` is ``(e2e_ms - ttft_ms) / (output_tokens -
1)``, so it inherits the provider's token count."""


@dataclass(frozen=True)
class ApiRequestSpec:
    """One request to issue under load.

    The spec owns the request content: ``prompt`` and ``system_prompt``
    replace the base config's values, and a ``None`` system prompt means
    no system message. ``max_tokens`` overrides the base config's
    ``max_output_tokens`` when set. ``extra_body`` is merged over the base
    config's ``extra_body``, with the spec winning on a shared key.
    ``run_id_suffix`` names the request inside the run; it defaults to the
    zero-padded dispatch index. ``history`` holds earlier conversation
    turns sent before ``prompt``, as ``(role, content)`` pairs; the
    collector config checks its shape.
    """

    prompt: str
    system_prompt: str | None = None
    run_id_suffix: str | None = None
    max_tokens: int | None = None
    extra_body: Mapping[str, Any] | None = None
    history: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise OpenAIStreamCollectorError("request prompt must be non-empty text")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise OpenAIStreamCollectorError("system_prompt must be text or None")
        if self.run_id_suffix is not None and not _SUFFIX_PATTERN.match(
            str(self.run_id_suffix)
        ):
            raise OpenAIStreamCollectorError(
                "run_id_suffix must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}, got "
                f"{self.run_id_suffix!r}"
            )
        if self.max_tokens is not None and (
            not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens < 1
        ):
            raise OpenAIStreamCollectorError(
                f"max_tokens must be a positive integer, got {self.max_tokens!r}"
            )
        if self.extra_body is not None and not isinstance(self.extra_body, Mapping):
            raise OpenAIStreamCollectorError("extra_body must be a mapping or None")


def request_config(
    base: APICollectionConfig, spec: ApiRequestSpec, index: int
) -> APICollectionConfig:
    """The collector config for request ``index``. Validates on construction."""
    suffix = spec.run_id_suffix if spec.run_id_suffix is not None else f"{index:06d}"
    extra: dict[str, Any] | None = None
    if base.extra_body is not None or spec.extra_body is not None:
        extra = dict(base.extra_body or {})
        extra.update(spec.extra_body or {})
    return dataclasses.replace(
        base,
        run_id=f"{base.run_id}-{suffix}",
        prompt=spec.prompt,
        system_prompt=spec.system_prompt,
        history=spec.history,
        max_output_tokens=(
            spec.max_tokens if spec.max_tokens is not None else base.max_output_tokens
        ),
        extra_body=extra,
    )


def preflight(
    base: APICollectionConfig,
    specs: Sequence[ApiRequestSpec],
    *,
    environ: Mapping[str, str],
) -> list[APICollectionConfig]:
    """Build and check every request config before any request is sent.

    A bad spec, a missing credential, or a credential pasted into a prompt
    is refused here, so it cannot surface halfway through a load run as a
    string of per-request client errors.
    """
    configs = [request_config(base, spec, index) for index, spec in enumerate(specs)]
    run_ids = [config.run_id for config in configs]
    if len(set(run_ids)) != len(run_ids):
        raise OpenAIStreamCollectorError("request run ids must be unique")
    _resolve_credential(base, environ)
    for config in configs:
        assert_credential_not_embedded(config, environ)
        build_request_plan(config, environ=environ)
    return configs


@dataclass(frozen=True)
class RequestRow:
    """One request's measured outcome under load. Holds no prompt or output text.

    ``success`` is ``None`` when the request was never sent or the client
    raised before a result existed. Token fields are ``None`` when the
    provider did not report them; nothing is estimated.
    """

    index: int
    sent: bool
    run_id: str | None = None
    scheduled_offset_s: float | None = None
    dispatched_offset_s: float | None = None
    completed_offset_s: float | None = None
    in_flight_at_dispatch: int | None = None
    client_error: str | None = None
    success: bool | None = None
    failure_category: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    finish_reason_code: str | None = None
    ttft_ms: float | None = None
    ttft_source: str | None = None
    first_content_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    tpot_ms: float | None = None
    tpot_status: str | None = None
    content_delta_count: int | None = None
    prompt_sha256: str | None = None
    workload_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _ttft(result: APICollectionResult) -> tuple[float | None, str | None]:
    """Time to the first generated delta, reasoning included.

    Matches the exporter's ``gen_ai.response.time_to_first_chunk``: the
    earliest retained content or reasoning event that carried text, else
    the first content offset when the stream had no reasoning at all.
    """
    timeline = result.evidence.timeline
    offsets = [
        event.offset_ms
        for event in timeline.events
        if event.kind in _OUTPUT_EVENT_KINDS and event.characters > 0
    ]
    if offsets:
        return min(offsets), TTFT_EARLIEST_OUTPUT_EVENT
    statistics = result.evidence.statistics
    first_content = timeline.first_content_token_offset_ms
    if statistics.reasoning_delta_count == 0 and first_content is not None:
        return first_content, TTFT_FIRST_CONTENT
    return None, None


def _tpot(
    ttft_ms: float | None, e2e_ms: float | None, output_tokens: int | None
) -> tuple[float | None, str]:
    if ttft_ms is None or e2e_ms is None or output_tokens is None:
        return None, TPOT_UNAVAILABLE
    if output_tokens <= 1:
        return None, TPOT_NOT_APPLICABLE
    return (e2e_ms - ttft_ms) / (output_tokens - 1), TPOT_DERIVED


def row_from_result(
    result: APICollectionResult, context: DispatchContext, prompt: str
) -> RequestRow:
    """Reduce one collector result to a ``RequestRow`` placed on the dispatch timeline."""
    evidence = result.evidence
    timeline = evidence.timeline
    usage = evidence.usage
    ttft_ms, ttft_source = _ttft(result)
    e2e_ms = timeline.completed_offset_ms
    tpot_ms, tpot_status = _tpot(ttft_ms, e2e_ms, usage.completion_tokens)
    completed_offset = (
        None
        if result.completed_monotonic is None
        else result.completed_monotonic - context.clock_origin
    )
    return RequestRow(
        index=context.index,
        sent=True,
        run_id=evidence.run_id,
        scheduled_offset_s=context.scheduled_offset_s,
        dispatched_offset_s=context.dispatched_offset_s,
        completed_offset_s=completed_offset,
        in_flight_at_dispatch=context.in_flight_at_dispatch,
        success=evidence.success,
        failure_category=(
            None if evidence.failure is None else evidence.failure.category
        ),
        http_status=result.http_status,
        finish_reason=evidence.finish_reason,
        finish_reason_code=evidence.finish_reason_code,
        ttft_ms=ttft_ms,
        ttft_source=ttft_source,
        first_content_ms=timeline.first_content_token_offset_ms,
        e2e_ms=e2e_ms,
        output_tokens=usage.completion_tokens,
        prompt_tokens=usage.prompt_tokens,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        tpot_ms=tpot_ms,
        tpot_status=tpot_status,
        content_delta_count=evidence.statistics.content_delta_count,
        prompt_sha256=sha256_text(prompt),
        workload_hash=evidence.plan.workload_hash,
    )


def make_api_sender(
    base_config: APICollectionConfig,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
) -> Callable[[ApiRequestSpec, DispatchContext], RequestRow]:
    """A thread-safe ``send`` for ``dispatch`` that streams one chat completion.

    The collector runs on the dispatcher's clock, so the row's offsets and
    the collector's latencies share one timeline. ``transport`` must be
    safe to use from several threads at once; ``UrllibStreamingTransport``
    is, since it opens a new connection per request.
    """

    def send(spec: ApiRequestSpec, context: DispatchContext) -> RequestRow:
        config = request_config(base_config, spec, context.index)
        result = stream_chat_completion(
            config, transport=transport, environ=environ, clock=context.clock
        )
        return row_from_result(result, context, spec.prompt)

    return send


def request_rows(report: DispatchReport[RequestRow]) -> list[RequestRow]:
    """One row per dispatched request, in order, including unsent and client failures."""
    return [_row_for(outcome) for outcome in report.outcomes]


def _row_for(outcome: DispatchOutcome[RequestRow]) -> RequestRow:
    if outcome.result is not None:
        return outcome.result
    return RequestRow(
        index=outcome.index,
        sent=outcome.sent,
        scheduled_offset_s=outcome.scheduled_offset_s,
        dispatched_offset_s=outcome.dispatched_offset_s,
        completed_offset_s=outcome.completed_offset_s,
        in_flight_at_dispatch=outcome.in_flight_at_dispatch,
        client_error=outcome.client_error,
    )
