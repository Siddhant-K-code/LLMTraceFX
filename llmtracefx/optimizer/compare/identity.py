"""Comparable-unit and system identity keys for the ``compare`` command.

The tuner (``optimizer.tune.identity``) answers a narrower question: which
configuration of *one* model on *one* machine is fastest. It therefore puts
model id, accelerator and runtime inside its ``GroupKey``, so two different
systems can never be ranked against each other.

Cross-system comparison asks the opposite question, so the split has to move:

* ``ComparableUnitKey`` is the *task* two systems were both asked to perform.
  Two runs belong to the same unit only when the workload identity, the exact
  prompt hash, the context tier, the evaluator/quality metric, and the decode
  settings that bound how much output was allowed are all identical. Anything
  else is a different question and lands in a different stratum.
* ``SystemKey`` is *what was under test*: model and model revision, provider,
  runtime/backend, accelerator, quantization, reasoning effort, and decode
  mode. Systems are never averaged together, and two candidates that differ
  on any one of these fields stay distinct.

Neither key is ever inferred. Every field is read from an already-validated
``RowVerification``/``ExperimentRecord`` pair (plus, for API runs, the
collector's own ``api_evidence.json``), and a value that was not recorded
stays ``None`` rather than being guessed or defaulted to zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

COMPARE_IDENTITY_CONTEXT = "compare identity"


class CompareIdentityError(ValueError):
    """Raised when a comparable-unit/system key is invalid or malformed."""


def _require_str(data: Any, key: str, *, context: str) -> str:
    if not isinstance(data, dict) or key not in data:
        raise CompareIdentityError(f"{context} is missing required field: {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise CompareIdentityError(
            f"{context}.{key} must be a non-empty string, got {value!r}"
        )
    return value


def _optional_str(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CompareIdentityError(
            f"{context}.{key} must be a non-empty string or null, got {value!r}"
        )
    return value


def _optional_positive_int(
    data: dict[str, Any], key: str, *, context: str
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompareIdentityError(
            f"{context}.{key} must be an integer or null, got {value!r}"
        )
    if value < 1:
        raise CompareIdentityError(f"{context}.{key} must be >= 1, got {value}")
    return int(value)


def _optional_finite_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompareIdentityError(
            f"{context}.{key} must be a number or null, got {value!r}"
        )
    try:
        numeric = float(value)
    except OverflowError as exc:
        # A JSON integer literal too large for a float arrives here as a
        # Python int. ``float()`` raises ``OverflowError``, an
        # ``ArithmeticError`` rather than a ``ValueError``, so no caller
        # catches it and it escapes as a traceback. Every one of these
        # inputs is a user-supplied file this module treats as untrusted,
        # so it becomes the same typed validation failure as any other
        # malformed value.
        raise CompareIdentityError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric):
        raise CompareIdentityError(
            f"{context}.{key} must be a finite number, got {numeric!r}"
        )
    return numeric


def _format_setting(value: float | int | None) -> str:
    if value is None:
        return "unrecorded"
    if isinstance(value, int):
        return str(value)
    return f"{value:g}"


#: Marks a message list that is not the plain single-user-prompt shape. The
#: digest that follows it is over the message structure, never over any
#: message text, which this project does not persist.
REQUEST_SHAPE_PREFIX = "sha256:"


def request_shape_for(
    messages: Sequence[tuple[str, str]], *, workload_prompt_hash: str
) -> str | None:
    """Normalize a recorded message list into a comparable request shape.

    ``messages`` is ``(role, content_sha256)`` per message, exactly as the API
    collector persists it -- digests, never text.

    Returns ``None`` for the canonical shape: a single ``user`` message whose
    content hash is the workload prompt itself. That is precisely what a local
    run does when it feeds a prompt file to a model, and a local run records
    no message structure at all, so normalizing the two to ``None`` is what
    lets a local system and a hosted one land in the same comparable unit.

    Anything else -- a system prompt, a prepended conversation, a reordered or
    duplicated message -- returns a digest instead, which no local run can
    match. That is the intended outcome: a prompt sent under a system prompt
    is a different question, and must be reported as a separate stratum
    rather than compared against a bare prompt as though they were equivalent.
    """
    entries = list(messages)
    if len(entries) == 1:
        role, digest = entries[0]
        if role == "user" and digest == workload_prompt_hash:
            return None
    payload = json.dumps(
        [[role, digest] for role, digest in entries],
        sort_keys=False,
        separators=(",", ":"),
    )
    return REQUEST_SHAPE_PREFIX + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparableUnitKey:
    """The identical task two systems must both have performed.

    ``max_output_tokens``/``temperature``/``top_p`` are part of the identity
    on purpose. A run capped at 128 output tokens is not the same measurement
    as one capped at 2048, and a run whose cap was never recorded is not
    known to match either. ``None`` therefore means "unrecorded", and it only
    ever compares equal to another unrecorded value -- it is never treated as
    a wildcard that silently absorbs runs with a known setting.

    ``request_shape`` carries the rest of what was actually asked. A prompt
    hash alone does not identify a request: the same user prompt sent with a
    system prompt, or as the tail of a multi-turn conversation, is a
    different question. See ``request_shape_for`` for how it is normalized so
    that the ordinary single-user-prompt case still compares against a local
    run, which has no message structure to record.
    """

    workload_id: str
    workload_version: str
    workload_prompt_hash: str
    context_tier: str
    quality_metric: str | None
    max_output_tokens: int | None
    temperature: float | None
    top_p: float | None
    request_shape: str | None = None

    def label(self) -> str:
        shape = "" if self.request_shape is None else f" shape={self.request_shape}"
        return (
            f"{self.workload_id}@v{self.workload_version} [{self.context_tier}] "
            f"evaluator={self.quality_metric or 'unrecorded'} "
            f"max_output={_format_setting(self.max_output_tokens)} "
            f"temperature={_format_setting(self.temperature)} "
            f"top_p={_format_setting(self.top_p)}{shape}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "workload_prompt_hash": self.workload_prompt_hash,
            "context_tier": self.context_tier,
            "quality_metric": self.quality_metric,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "request_shape": self.request_shape,
        }

    def sort_key(self) -> tuple[Any, ...]:
        # Each optional field contributes an ``is None`` flag *and* its value
        # rather than folding "unrecorded" onto a sentinel number. A sentinel
        # such as -1 would make an unrecorded temperature sort identically to a
        # recorded -1.0, so two genuinely different comparable units would
        # collide on the key that decides their order in the report.
        return (
            self.workload_id,
            self.workload_version,
            self.context_tier,
            self.workload_prompt_hash,
            self.quality_metric is None,
            self.quality_metric or "",
            self.max_output_tokens is None,
            self.max_output_tokens or 0,
            self.temperature is None,
            self.temperature or 0.0,
            self.top_p is None,
            self.top_p or 0.0,
            self.request_shape is None,
            self.request_shape or "",
        )

    @classmethod
    def from_dict(cls, data: Any) -> ComparableUnitKey:
        if not isinstance(data, dict):
            raise CompareIdentityError("comparable_unit_key must be a JSON object")
        context = "comparable_unit_key"
        return cls(
            workload_id=_require_str(data, "workload_id", context=context),
            workload_version=_require_str(data, "workload_version", context=context),
            workload_prompt_hash=_require_str(
                data, "workload_prompt_hash", context=context
            ),
            context_tier=_require_str(data, "context_tier", context=context),
            quality_metric=_optional_str(data, "quality_metric", context=context),
            max_output_tokens=_optional_positive_int(
                data, "max_output_tokens", context=context
            ),
            temperature=_optional_finite_float(data, "temperature", context=context),
            top_p=_optional_finite_float(data, "top_p", context=context),
            request_shape=_optional_str(data, "request_shape", context=context),
        )


@dataclass(frozen=True)
class SystemKey:
    """One system under test, labeled by everything that makes it that system.

    Deliberately conservative: two runs are only ever pooled into the same
    system when every field matches exactly. A different quantization, a
    different reasoning effort, a different model revision, a different
    deployment endpoint or any difference at all in the recorded execution
    configuration is a different system, never a variant of the same one to
    be averaged in.

    ``execution_config_hash`` is the collector's own identity hash over the
    whole request configuration (endpoint, model, sampling, provider
    extensions, finish-reason vocabulary, timeout, retained event limit and
    system prompt). It is carried verbatim rather than recomputed, so a
    configuration difference this module does not model by name still
    separates two systems instead of silently pooling them. The trade is that
    a collector field which legitimately varies per run (Z.ai's optional
    body-level ``request_id``, for instance) will split runs that a reader
    might consider one system; that is the safe direction to err.
    """

    model_id: str
    model_revision: str | None
    provider: str | None
    runtime_name: str
    runtime_backend: str | None
    accelerator: str | None
    quantization: str | None
    reasoning_effort: str | None
    decode_mode: str
    endpoint: str | None = None
    thinking_type: str | None = None
    execution_config_hash: str | None = None

    @property
    def is_local(self) -> bool:
        """True when no remote provider executed this system's runs.

        Local-only measurements (peak memory in particular) are reported
        for these systems and withheld for everything else, rather than
        being compared against a value a hosted API cannot produce.
        """
        return self.provider is None

    def label(self) -> str:
        where = self.provider or (self.accelerator or "unknown hardware")
        revision = f"@{self.model_revision}" if self.model_revision else ""
        thinking = f" thinking={self.thinking_type}" if self.thinking_type else ""
        deployment = f" endpoint={self.endpoint}" if self.endpoint else ""
        return (
            f"{self.model_id}{revision} via {where} "
            f"[{self.runtime_name}/{self.runtime_backend or 'unknown'}] "
            f"quant={self.quantization or 'unrecorded'} "
            f"reasoning={self.reasoning_effort or 'unrecorded'}{thinking} "
            f"decode={self.decode_mode}{deployment}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "provider": self.provider,
            "runtime_name": self.runtime_name,
            "runtime_backend": self.runtime_backend,
            "accelerator": self.accelerator,
            "quantization": self.quantization,
            "reasoning_effort": self.reasoning_effort,
            "decode_mode": self.decode_mode,
            "endpoint": self.endpoint,
            "thinking_type": self.thinking_type,
            "execution_config_hash": self.execution_config_hash,
            "is_local": self.is_local,
        }

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.model_id,
            self.model_revision or "",
            self.provider or "",
            self.runtime_name,
            self.runtime_backend or "",
            self.accelerator or "",
            self.quantization or "",
            self.reasoning_effort or "",
            self.decode_mode,
            self.endpoint or "",
            self.thinking_type or "",
            self.execution_config_hash or "",
        )

    @classmethod
    def from_dict(cls, data: Any) -> SystemKey:
        if not isinstance(data, dict):
            raise CompareIdentityError("system_key must be a JSON object")
        context = "system_key"
        key = cls(
            model_id=_require_str(data, "model_id", context=context),
            model_revision=_optional_str(data, "model_revision", context=context),
            provider=_optional_str(data, "provider", context=context),
            runtime_name=_require_str(data, "runtime_name", context=context),
            runtime_backend=_optional_str(data, "runtime_backend", context=context),
            accelerator=_optional_str(data, "accelerator", context=context),
            quantization=_optional_str(data, "quantization", context=context),
            reasoning_effort=_optional_str(data, "reasoning_effort", context=context),
            decode_mode=_require_str(data, "decode_mode", context=context),
            endpoint=_optional_str(data, "endpoint", context=context),
            thinking_type=_optional_str(data, "thinking_type", context=context),
            execution_config_hash=_optional_str(
                data, "execution_config_hash", context=context
            ),
        )
        declared_local = data.get("is_local")
        if declared_local is not None:
            if not isinstance(declared_local, bool):
                raise CompareIdentityError(
                    f"{context}.is_local must be a boolean or null, got "
                    f"{declared_local!r}"
                )
            if declared_local != key.is_local:
                raise CompareIdentityError(
                    f"{context}.is_local ({declared_local!r}) contradicts the "
                    f"provider field ({key.provider!r}); refusing to load a key "
                    "whose locality claim does not follow from its own identity"
                )
        if key.is_local and key.endpoint is not None:
            raise CompareIdentityError(
                f"{context} declares no provider yet records a deployment "
                f"endpoint ({key.endpoint!r}); a local system has no endpoint"
            )
        return key
