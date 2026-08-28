"""llama.cpp text-output collector.

Parses the human-readable timing/perf lines that ``llama.cpp`` binaries
(``llama-cli``, ``llama-bench``, ``llama-speculative``, ...) print to
stdout/stderr, and converts them into the canonical
:class:`~llmtracefx.optimizer.schema.ExperimentRecord` schema.

Supports both the historical ``llama_print_timings:`` prefix and the
current ``llama_perf_context_print:`` prefix, since both report the same
fields. Missing optional lines (e.g. no speculative-decoding stats when
speculative decoding was not enabled) are tolerated and simply leave the
corresponding schema fields as ``None``. A line that *does* match a known
label but has a value that cannot be parsed as a number raises
``LlamaCppParseError`` rather than being silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import (
    CommandInfo,
    ExperimentRecord,
    Measurement,
    MetricProvenance,
    ModelInfo,
    OutcomeInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SpeculativeDecodingInfo,
    TimingMetrics,
    TokenCounts,
)


class LlamaCppParseError(ValueError):
    """Raised when a recognized llama.cpp output line has a bad value."""


_LOAD_RE = re.compile(r"load time\s*=\s*([^\s]+)\s*ms")
_PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([^\s]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
_EVAL_RE = re.compile(r"eval time\s*=\s*([^\s]+)\s*ms\s*/\s*(\d+)\s*runs")
_TOTAL_RE = re.compile(r"total time\s*=\s*([^\s]+)\s*ms\s*/\s*(\d+)\s*tokens")

_N_DRAFT_RE = re.compile(r"^n_draft\s*=\s*(\d+)")
_N_PREDICT_RE = re.compile(r"^n_predict\s*=\s*(\d+)")
_N_DRAFTED_RE = re.compile(r"^n_drafted\s*=\s*(\d+)")
_N_ACCEPT_RE = re.compile(r"^n_accept(?:ed)?\s*=\s*(\d+)")

_METAL_DEVICE_RE = re.compile(r"found device:\s*(.+)$")
_BACKEND_RE = re.compile(r"loaded\s+(\w+)\s+backend", re.IGNORECASE)


def _parse_float(label: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise LlamaCppParseError(
            f"could not parse {label} value '{raw}' as a number"
        ) from exc


@dataclass(frozen=True)
class ParsedLlamaCppOutput:
    """Structured fields recovered from a llama.cpp run's text output."""

    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    prompt_eval_tokens: int | None = None
    eval_ms: float | None = None
    eval_tokens: int | None = None
    total_ms: float | None = None
    total_tokens: int | None = None
    n_draft: int | None = None
    n_predict: int | None = None
    n_drafted: int | None = None
    n_accepted: int | None = None
    device_hint: str | None = None
    backend_hint: str | None = None

    @property
    def speculative_reported(self) -> bool:
        """Whether any speculative-decoding counters were present at all."""
        return self.n_drafted is not None or self.n_accepted is not None


def parse_llama_cpp_output(text: str) -> ParsedLlamaCppOutput:
    """Parse llama.cpp stdout/stderr text into structured fields.

    Tolerates missing optional lines. Raises ``LlamaCppParseError`` if a
    recognized line's numeric value is malformed.
    """
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    prompt_eval_tokens: int | None = None
    eval_ms: float | None = None
    eval_tokens: int | None = None
    total_ms: float | None = None
    total_tokens: int | None = None
    n_draft: int | None = None
    n_predict: int | None = None
    n_drafted: int | None = None
    n_accepted: int | None = None
    device_hint: str | None = None
    backend_hint: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "prompt eval time" in line:
            match = _PROMPT_EVAL_RE.search(line)
            if match:
                prompt_eval_ms = _parse_float("prompt eval time", match.group(1))
                prompt_eval_tokens = int(match.group(2))
            continue

        if "sample time" in line:
            # Sampling time is not part of the canonical timing schema
            # (it overlaps with decode and is not comparable across
            # runtimes); intentionally not captured here.
            continue

        if "load time" in line:
            match = _LOAD_RE.search(line)
            if match:
                load_ms = _parse_float("load time", match.group(1))
            continue

        if "eval time" in line:
            match = _EVAL_RE.search(line)
            if match:
                eval_ms = _parse_float("eval time", match.group(1))
                eval_tokens = int(match.group(2))
            continue

        if "total time" in line:
            match = _TOTAL_RE.search(line)
            if match:
                total_ms = _parse_float("total time", match.group(1))
                total_tokens = int(match.group(2))
            continue

        matched_counter = False
        for label, pattern, error_label in (
            ("n_draft", _N_DRAFT_RE, "n_draft"),
            ("n_predict", _N_PREDICT_RE, "n_predict"),
            ("n_drafted", _N_DRAFTED_RE, "n_drafted"),
            ("n_accepted", _N_ACCEPT_RE, "n_accept"),
        ):
            match = pattern.search(line)
            if not match:
                continue
            matched_counter = True
            try:
                value = int(match.group(1))
            except ValueError as exc:
                raise LlamaCppParseError(
                    f"could not parse {error_label} value from line: {line!r}"
                ) from exc
            if label == "n_draft":
                n_draft = value
            elif label == "n_predict":
                n_predict = value
            elif label == "n_drafted":
                n_drafted = value
            else:
                n_accepted = value
            break

        if matched_counter:
            continue

        device_match = _METAL_DEVICE_RE.search(line)
        if device_match:
            device_hint = device_match.group(1).strip()
            continue
        backend_match = _BACKEND_RE.search(line)
        if backend_match:
            backend_hint = backend_match.group(1).strip()
            continue

    return ParsedLlamaCppOutput(
        load_ms=load_ms,
        prompt_eval_ms=prompt_eval_ms,
        prompt_eval_tokens=prompt_eval_tokens,
        eval_ms=eval_ms,
        eval_tokens=eval_tokens,
        total_ms=total_ms,
        total_tokens=total_tokens,
        n_draft=n_draft,
        n_predict=n_predict,
        n_drafted=n_drafted,
        n_accepted=n_accepted,
        device_hint=device_hint,
        backend_hint=backend_hint,
    )


def _measurement(value: float | None, unit: str) -> Measurement | None:
    if value is None:
        return None
    # These come from llama.cpp's own internal phase timers (ggml_time_us
    # around load/prompt-eval/eval/total), i.e. the runtime's native
    # instrumentation -- not a coarse wall-clock wrapper around the whole
    # process, which is what the experiment runner captures separately.
    return Measurement(
        value=value, provenance=MetricProvenance.MEASURED_NATIVE, unit=unit
    )


def build_experiment_record(
    *,
    run_id: str,
    started_at: str,
    platform: PlatformInfo,
    model: ModelInfo,
    command: CommandInfo,
    repetition: RepetitionInfo,
    stdout_text: str,
    stderr_text: str = "",
    runtime_version: str | None = None,
    runtime_git_revision: str | None = None,
    speculative_method: str | None = None,
    ended_at: str | None = None,
    outcome: OutcomeInfo | None = None,
) -> ExperimentRecord:
    """Parse llama.cpp output and assemble a validated ``ExperimentRecord``.

    ``platform``, ``model``, ``command``, and ``repetition`` describe
    context that cannot be recovered from stdout/stderr alone (e.g. from
    an :class:`~llmtracefx.optimizer.manifest.EnvironmentManifest` and the
    runner's ``RunnerConfig``). ``speculative_method`` should be supplied
    by the caller (e.g. ``"mtp"``) when speculative decoding was enabled,
    since llama.cpp's plain timing output does not name the method.
    """
    parsed = parse_llama_cpp_output(stdout_text + "\n" + stderr_text)

    runtime = RuntimeInfo(
        name="llama.cpp",
        version=runtime_version,
        backend=parsed.backend_hint,
        git_revision=runtime_git_revision,
    )

    timing = TimingMetrics(
        model_load=_measurement(parsed.load_ms, "ms"),
        prefill=_measurement(parsed.prompt_eval_ms, "ms"),
        decode=_measurement(parsed.eval_ms, "ms"),
        total=_measurement(parsed.total_ms, "ms"),
    )

    tokens = TokenCounts(
        input_tokens=parsed.prompt_eval_tokens,
        generated_tokens=parsed.eval_tokens,
    )

    speculative = SpeculativeDecodingInfo(
        enabled=parsed.speculative_reported,
        method=speculative_method if parsed.speculative_reported else None,
        configured_depth=parsed.n_draft,
        proposed_tokens=parsed.n_drafted,
        accepted_tokens=parsed.n_accepted,
    )

    record = ExperimentRecord(
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        platform=platform,
        model=model,
        runtime=runtime,
        command=command,
        repetition=repetition,
        tokens=tokens,
        timing=timing,
        speculative=speculative,
        outcome=outcome or OutcomeInfo(),
    )
    record.validate()
    return record
