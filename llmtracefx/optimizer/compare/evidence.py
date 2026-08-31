"""Loading cross-system evidence from already-collected result directories.

Nothing here executes anything. It reads artifacts that a previous
``workloads run`` or ``workloads run-api`` invocation already wrote,
re-validates them, and turns each trustworthy run into a ``SystemRun``.

**What this can and cannot ingest.** The only accepted input is a results
directory shaped like ``workloads run --output-dir``: a ``runs/<run_id>/``
tree holding a ``verification.json`` and the ``final_record.json`` it points
at. Raw ``collect-api`` output is *not* directly ingestible. That command
writes a flat artifact set (``record.json``, ``api_evidence.json``,
``response.txt``, ``environment.json``, ``artifacts.json``) into whatever
``--output-dir`` it was given; it produces no ``verification.json``, so
nothing has evaluated the response, and an unevaluated run has no pass rate,
no quality score and therefore nothing this module can honestly compare.
Getting API rows into a comparable shape is the job of
``workloads run-api``, which executes the matrix against an
OpenAI-compatible endpoint and writes the same ``runs/<run_id>/`` tree the
local pipeline does. ``_raw_collector_directory_error`` detects the mistake
and says so, rather than reporting "no comparable units" and leaving the
operator to guess why.

Run/record identity checking is *not* re-implemented: per directory, this
module delegates to ``optimizer.tune.loader.load_evidence``, which re-reads
every ``final_record.json`` behind a ``verification.json`` and re-checks run
id and prompt-hash agreement. What this module does *not* inherit is that
loader's global keying on ``run_id`` alone. A matrix ``run_id`` is
``<workload>-<tier>-<decode_mode>``: it names the task and nothing about the
system, so executing one matrix against a local model and against a hosted
API produces two results directories whose run ids collide completely. That
is the ordinary case here and must not be an error, so directories are
loaded independently and merged under a source-scoped key. What separates
distinct evidence from a repeated reference is the *artifact*, not the
identity: two results trees holding the same matrix row are two repetitions,
which is how a policy's ``min_measured_repetitions`` is satisfied, while the
same directory named twice is one run.

On top of that, this module adds the two things cross-system comparison
needs and tuning does not:

* **Provider evidence.** For a run collected against a hosted API, the
  collector wrote an ``api_evidence.json`` beside the record. It carries the
  provider's own token accounting, the requested reasoning effort, the
  decode settings that were actually sent, the request identity, and the
  client-observed time-to-first-token. Before any of it is read, the
  surrounding artifact set is checked for completeness against the
  collector's own ``artifacts.json`` marker, and the sidecar's schema
  version, run id and model/workload identity are checked against the
  record. A sidecar that fails any of that excludes the run rather than
  degrading it.

* **Decode settings.** Output cap and sampling are part of the comparable
  unit, so they are read from evidence: from the API request plan when there
  is one, otherwise from the long options this project itself put into
  ``command.argv``. A setting that neither source records stays ``None``,
  which makes the run comparable only against other runs whose setting is
  equally unrecorded.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..collectors.openai_api import (
    _MAX_ARTIFACT_MANIFEST_BYTES,
    API_EVIDENCE_SCHEMA_VERSION,
    ARTIFACT_MANIFEST_NAME,
    _read_bounded_regular_file,
    artifact_set_is_complete,
)
from ..schema import ExperimentRecord
from ..tune.loader import ExcludedRun, LoadedEvidence, RunEvidence, load_evidence
from ..workloads.api_verify import RUN_MANIFEST_NAME, run_artifacts_are_complete
from ..workloads.verify import RowVerification
from .cost import TokenUsage
from .identity import ComparableUnitKey, SystemKey, request_shape_for

#: Written beside ``record.json`` by the OpenAI-compatible API collector.
API_EVIDENCE_FILENAME = "api_evidence.json"

#: The largest token count that survives a round trip through a float, and
#: therefore the largest one any rate or cost derived from it can be honest
#: about. Above it ``float(value)`` either raises ``OverflowError`` or loses
#: integer precision silently.
#:
#: The collector applies the same bound when it writes a count (see
#: ``openai_api._MAX_EXACT_TOKEN_COUNT``). It is re-established here rather
#: than assumed, for two reasons: this layer exists to read artifacts that an
#: *earlier* build wrote, and the collector's cap landed only recently; and
#: the counts are provider-controlled input in the first place, which is
#: exactly why the writer bounds them. Without it a single oversized count
#: turns cost estimation into an unhandled ``OverflowError`` (an
#: ``ArithmeticError``, so nothing downstream catches it) or, worse, into a
#: number that is quietly wrong.
MAX_EXACT_TOKEN_COUNT = 2**53

#: Long options this project's own collectors put into ``command.argv``.
#: Only these exact spellings are read; nothing is inferred from positional
#: arguments or from an option this table does not name.
_ARGV_INT_OPTIONS: dict[str, str] = {"--max-tokens": "max_output_tokens"}
_ARGV_FLOAT_OPTIONS: dict[str, str] = {
    "--temperature": "temperature",
    "--top-p": "top_p",
}


class CompareEvidenceError(ValueError):
    """Raised when the *set* of comparison inputs cannot be trusted."""


class ApiEvidenceError(ValueError):
    """Raised when an ``api_evidence.json`` sidecar is malformed."""


def _optional_non_negative_int(
    data: Any, key: str, *, context: str, maximum: int | None = None
) -> int | None:
    if not isinstance(data, dict):
        raise ApiEvidenceError(f"{context} must be a JSON object, got {data!r}")
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiEvidenceError(
            f"{context}.{key} must be an integer or null, got {value!r}"
        )
    if value < 0:
        raise ApiEvidenceError(f"{context}.{key} must be >= 0, got {value}")
    if maximum is not None and value > maximum:
        raise ApiEvidenceError(
            f"{context}.{key} is {value}, above the largest count that can be "
            f"used in an exact calculation ({maximum}); refusing to derive a "
            "rate or a cost from a number that cannot round trip through a "
            "float"
        )
    return int(value)


def _optional_finite_float(
    data: Any, key: str, *, context: str, minimum: float | None = None
) -> float | None:
    if not isinstance(data, dict):
        raise ApiEvidenceError(f"{context} must be a JSON object, got {data!r}")
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiEvidenceError(
            f"{context}.{key} must be a number or null, got {value!r}"
        )
    try:
        numeric = float(value)
    except OverflowError as exc:
        # A JSON integer literal larger than a float can represent reaches
        # here as a Python int. ``float()`` raises ``OverflowError``, which
        # is an ``ArithmeticError`` rather than a ``ValueError``, so nothing
        # downstream would catch it. Convert it into the same strict
        # validation failure every other malformed field produces.
        raise ApiEvidenceError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric):
        raise ApiEvidenceError(
            f"{context}.{key} must be a finite number, got {numeric!r}"
        )
    if minimum is not None and numeric < minimum:
        raise ApiEvidenceError(f"{context}.{key} must be >= {minimum}, got {numeric!r}")
    return numeric


def _optional_str(data: Any, key: str, *, context: str) -> str | None:
    if not isinstance(data, dict):
        raise ApiEvidenceError(f"{context} must be a JSON object, got {data!r}")
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ApiEvidenceError(
            f"{context}.{key} must be a non-empty string or null, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class DecodeSettings:
    """Output cap and sampling for one run, plus where each value came from."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    source: str | None = None
    """``"api_request_plan"``, ``"command_argv"``, or ``None`` if unrecorded."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "source": self.source,
        }


@dataclass(frozen=True)
class ApiEvidence:
    """The parts of an ``api_evidence.json`` cross-system comparison needs."""

    provider: str | None
    model_id: str | None
    model_revision: str | None
    reasoning_effort: str | None
    usage_reported: bool
    usage: TokenUsage
    usage_malformed_fields: tuple[str, ...]
    decode_settings: DecodeSettings
    client_ttft_ms: float | None
    """Client-observed offset to the first content token.

    This includes DNS, connection setup, TLS, request transfer and any
    server-side queueing, so it is *not* the same measurement as a local
    collector's ``timing.prefill``. The two are never pooled; see
    ``TtftBasis`` in ``compare.py``.
    """
    schema_version: str | None = None
    run_id: str | None = None
    workload_hash: str | None = None
    config_hash: str | None = None
    endpoint: str | None = None
    thinking_type: str | None = None
    messages: tuple[tuple[str, str], ...] = ()
    """``(role, content_sha256)`` per message. Digests only, never text."""

    def fingerprint(self) -> dict[str, Any]:
        """Everything this comparison reads out of the sidecar.

        Used to tell a genuine duplicate from two runs that differ only in
        evidence the verification and the record do not carry.
        """
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "workload_hash": self.workload_hash,
            "config_hash": self.config_hash,
            "endpoint": self.endpoint,
            "reasoning_effort": self.reasoning_effort,
            "thinking_type": self.thinking_type,
            "messages": [list(message) for message in self.messages],
            "usage_reported": self.usage_reported,
            "usage": self.usage.to_dict(),
            "usage_malformed_fields": list(self.usage_malformed_fields),
            "decode_settings": self.decode_settings.to_dict(),
            "client_ttft_ms": self.client_ttft_ms,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ApiEvidence:
        if not isinstance(data, dict):
            raise ApiEvidenceError("api_evidence.json must be a JSON object")
        plan = data.get("plan")
        if not isinstance(plan, dict):
            raise ApiEvidenceError("api_evidence.json is missing its 'plan' object")

        parameters = plan.get("request_parameters", {})
        if not isinstance(parameters, dict):
            raise ApiEvidenceError(
                "api_evidence.json plan.request_parameters must be an object"
            )
        extensions = plan.get("provider_extensions", {})
        if not isinstance(extensions, dict):
            raise ApiEvidenceError(
                "api_evidence.json plan.provider_extensions must be an object"
            )

        max_output = _optional_non_negative_int(
            parameters, "max_tokens", context="plan.request_parameters"
        )
        if max_output == 0:
            raise ApiEvidenceError(
                "api_evidence.json plan.request_parameters.max_tokens must be "
                ">= 1 when set, got 0"
            )
        settings = DecodeSettings(
            max_output_tokens=max_output,
            temperature=_optional_finite_float(
                parameters, "temperature", context="plan.request_parameters"
            ),
            top_p=_optional_finite_float(
                parameters, "top_p", context="plan.request_parameters"
            ),
            source="api_request_plan",
        )

        usage_raw = data.get("usage", {})
        if not isinstance(usage_raw, dict):
            raise ApiEvidenceError("api_evidence.json 'usage' must be an object")
        reported = usage_raw.get("reported", False)
        if not isinstance(reported, bool):
            raise ApiEvidenceError(
                f"api_evidence.json usage.reported must be a boolean, got "
                f"{reported!r}"
            )
        malformed = usage_raw.get("malformed_fields", [])
        if not isinstance(malformed, list) or not all(
            isinstance(item, str) for item in malformed
        ):
            raise ApiEvidenceError(
                "api_evidence.json usage.malformed_fields must be a list of strings"
            )

        timeline = data.get("timeline", {})
        if not isinstance(timeline, dict):
            raise ApiEvidenceError("api_evidence.json 'timeline' must be an object")

        thinking = extensions.get("thinking")
        thinking_type: str | None = None
        if thinking is not None:
            if not isinstance(thinking, dict):
                raise ApiEvidenceError(
                    "api_evidence.json plan.provider_extensions.thinking must be "
                    f"an object or absent, got {thinking!r}"
                )
            thinking_type = _optional_str(
                thinking, "type", context="plan.provider_extensions.thinking"
            )

        origin = _optional_str(plan, "endpoint_origin", context="plan")
        path = _optional_str(plan, "endpoint_path", context="plan")
        endpoint = None if origin is None else f"{origin}{path or ''}"

        raw_messages = plan.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ApiEvidenceError("api_evidence.json plan.messages must be a list")
        messages: list[tuple[str, str]] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                raise ApiEvidenceError(
                    f"api_evidence.json plan.messages[{index}] must be an object"
                )
            context = f"plan.messages[{index}]"
            role = _optional_str(message, "role", context=context)
            digest = _optional_str(message, "content_sha256", context=context)
            if role is None or digest is None:
                raise ApiEvidenceError(
                    f"api_evidence.json {context} must record both 'role' and "
                    "'content_sha256'; a message with no identity cannot be "
                    "compared"
                )
            messages.append((role, digest))

        return cls(
            schema_version=_optional_str(
                data, "schema_version", context="api_evidence"
            ),
            run_id=_optional_str(data, "run_id", context="api_evidence"),
            workload_hash=_optional_str(plan, "workload_hash", context="plan"),
            config_hash=_optional_str(plan, "config_hash", context="plan"),
            endpoint=endpoint,
            thinking_type=thinking_type,
            messages=tuple(messages),
            provider=_optional_str(plan, "provider", context="plan"),
            model_id=_optional_str(plan, "model_id", context="plan"),
            model_revision=_optional_str(plan, "model_revision", context="plan"),
            reasoning_effort=_optional_str(
                extensions, "reasoning_effort", context="plan.provider_extensions"
            ),
            usage_reported=reported,
            usage=TokenUsage(
                prompt_tokens=_optional_non_negative_int(
                    usage_raw,
                    "prompt_tokens",
                    context="usage",
                    maximum=MAX_EXACT_TOKEN_COUNT,
                ),
                completion_tokens=_optional_non_negative_int(
                    usage_raw,
                    "completion_tokens",
                    context="usage",
                    maximum=MAX_EXACT_TOKEN_COUNT,
                ),
                cached_prompt_tokens=_optional_non_negative_int(
                    usage_raw,
                    "cached_prompt_tokens",
                    context="usage",
                    maximum=MAX_EXACT_TOKEN_COUNT,
                ),
                reasoning_tokens=_optional_non_negative_int(
                    usage_raw,
                    "reasoning_tokens",
                    context="usage",
                    maximum=MAX_EXACT_TOKEN_COUNT,
                ),
            ),
            usage_malformed_fields=tuple(malformed),
            decode_settings=settings,
            # An offset from the start of the request cannot precede it, so
            # a negative value is a corrupt or fabricated timeline rather
            # than a fast one. Left unchecked it would flatter its system on
            # every time-to-first-token comparison.
            client_ttft_ms=_optional_finite_float(
                timeline,
                "first_content_token_offset_ms",
                context="timeline",
                minimum=0.0,
            ),
        )

    @classmethod
    def read_json(cls, path: Path) -> ApiEvidence:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        # ``json`` raises past its own limits with exceptions that are not
        # ``JSONDecodeError``: an integer literal over the interpreter's digit
        # cap raises a plain ``ValueError``, and deep nesting raises
        # ``RecursionError``. This sidecar is the one input to this layer
        # derived from provider-controlled bytes, which is exactly why the
        # collector guards its own parse the same way. Without this, such a
        # file escapes as an unhandled crash instead of excluding the run.
        except (ValueError, RecursionError) as exc:
            raise ApiEvidenceError(f"could not parse {path}: {exc}") from exc
        return cls.from_dict(payload)


def decode_settings_from_argv(argv: tuple[str, ...]) -> DecodeSettings:
    """Read decode settings from the exact long options this project emits.

    Only ``--option value`` pairs whose option name is in the tables above
    are read, and only when the value parses cleanly. An option that appears
    twice with conflicting values yields nothing for that field rather than
    a coin flip, because a run whose cap is ambiguous is not a run whose cap
    is known.
    """
    seen: dict[str, list[float | int]] = {}
    for index, argument in enumerate(argv[:-1]):
        raw = argv[index + 1]
        if argument in _ARGV_INT_OPTIONS:
            try:
                value: float | int = int(raw)
            except ValueError:
                continue
            if value < 1:
                continue
            seen.setdefault(_ARGV_INT_OPTIONS[argument], []).append(value)
        elif argument in _ARGV_FLOAT_OPTIONS:
            try:
                value = float(raw)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            seen.setdefault(_ARGV_FLOAT_OPTIONS[argument], []).append(value)

    def unique(name: str) -> Any:
        values = seen.get(name)
        if not values or len(set(values)) != 1:
            return None
        return values[0]

    max_output = unique("max_output_tokens")
    temperature = unique("temperature")
    top_p = unique("top_p")
    if max_output is None and temperature is None and top_p is None:
        return DecodeSettings()
    return DecodeSettings(
        max_output_tokens=int(max_output) if max_output is not None else None,
        temperature=float(temperature) if temperature is not None else None,
        top_p=float(top_p) if top_p is not None else None,
        source="command_argv",
    )


@dataclass(frozen=True)
class SystemRun:
    """One trustworthy run, placed in its comparable unit and its system."""

    run_id: str
    source_results_dir: str
    unit_key: ComparableUnitKey
    system_key: SystemKey
    verification: RowVerification
    verification_path: Path
    record: ExperimentRecord
    record_path: Path
    decode_settings: DecodeSettings
    api_evidence: ApiEvidence | None

    @property
    def total_ms(self) -> float | None:
        total = self.record.timing.total
        if total is None or not math.isfinite(total.value):
            return None
        return total.value

    @property
    def local_prefill_ms(self) -> float | None:
        prefill = self.record.timing.prefill
        if prefill is None or not math.isfinite(prefill.value):
            return None
        return prefill.value

    @property
    def peak_memory_bytes(self) -> float | None:
        peak = self.record.memory.peak
        if peak is None or not math.isfinite(peak.value):
            return None
        return peak.value


@dataclass(frozen=True)
class LoadedComparisonEvidence:
    """Every usable run, plus everything that was set aside and why."""

    runs: tuple[SystemRun, ...]
    excluded: tuple[ExcludedRun, ...]


def _contained_directory(candidate: Path, *, root: Path) -> Path | None:
    """The resolved ``candidate`` when it is a directory inside ``root``.

    Resolved before the containment test so a symlink aimed outside the tree
    fails it rather than being followed out of it.
    """
    try:
        resolved = candidate.resolve(strict=True)
        anchor = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(anchor) or not resolved.is_dir():
        return None
    return resolved


def _resolve_collection_dir(
    collection_dir: str, *, results_dir: str, run_id: str
) -> Path | None:
    """Resolve a recorded collection directory inside the results tree named.

    Same reasoning as ``tune.loader._resolve_artifact_path``:
    ``verify.execute_row`` always writes the collection under
    ``<output_dir>/runs/<run_id>/collection``, so that is correct by
    construction and is tried first.

    Reading the recorded string literally would be worse than useless here. A
    relative ``--output-dir`` records a path relative to the working
    directory of the run, so a literal read resolves it against whichever
    tree this process is standing in, and the API evidence of a *different*
    system would be attached to this run. The identity checks cannot catch
    that on their own, because a matrix ``run_id`` names the task and nothing
    about the system, so two trees from one matrix agree on every id.

    As in the shared loader, nothing outside ``results_dir`` is returned. The
    recorded string comes from the artifact under validation, so an absolute
    path, a ``..`` segment or a symlink in it is chosen by whoever wrote that
    artifact. Resolution happens before the containment test, so a symlink
    aimed out of the tree fails it instead of being followed out. When
    nothing resolves to a directory inside the tree, ``None`` is returned
    and the caller reports the API evidence as missing, rather than falling
    back to a path that would be read from wherever it pointed.
    """
    root = Path(results_dir)
    canonical = root / "runs" / run_id / "collection"
    contained = _contained_directory(canonical, root=root)
    if contained is not None:
        return contained
    literal = Path(collection_dir)
    probe = literal if literal.is_absolute() else root / collection_dir
    return _contained_directory(probe, root=root)


def _api_evidence_for(evidence: RunEvidence) -> tuple[ApiEvidence | None, str | None]:
    """Load the API sidecar for a run, if the verification points at one.

    The surrounding artifact set is validated before the sidecar is trusted.
    The collector writes ``artifacts.json`` last, after every other file has
    landed, and hashes each one into it precisely so a consumer can tell a
    complete set from an interrupted write or a file swapped independently of
    the set it belongs to. Its own docstring says a consumer must check this
    before trusting the directory, so this does, using the collector's
    function rather than a private reimplementation of the same rule.
    """
    collection_dir = evidence.verification.collection_dir
    hosted = evidence.final_record.runtime.provider is not None
    if collection_dir is None:
        if hosted:
            # The record says a hosted provider executed this run, so the
            # sidecar exists by construction and its absence means the
            # evidence set is incomplete. Accepting the run anyway would
            # publish it with no endpoint, no request identity, no
            # provider-reported usage and no time-to-first-token, while
            # still labelling it hosted -- and would skip the artifact-set
            # validation entirely.
            return None, (
                f"final_record.runtime.provider is "
                f"{evidence.final_record.runtime.provider!r}, so this run was "
                "executed by a hosted API, but verification.json records no "
                "collection directory; the API evidence for it is missing"
            )
        return None, None
    directory = _resolve_collection_dir(
        collection_dir,
        results_dir=evidence.source_results_dir,
        run_id=evidence.final_record.run_id,
    )
    if directory is None:
        if hosted:
            return None, (
                f"final_record.runtime.provider is "
                f"{evidence.final_record.runtime.provider!r}, so this run was "
                "executed by a hosted API, but its collection directory does "
                "not resolve to a directory inside the results directory that "
                "was asked for; a collection reached through an absolute "
                "path, a parent-directory segment or a symlink leaving the "
                "tree is not this run's API evidence, so it is missing"
            )
        return None, None
    sidecar = directory / API_EVIDENCE_FILENAME
    if not sidecar.is_file():
        if hosted:
            return None, (
                f"final_record.runtime.provider is "
                f"{evidence.final_record.runtime.provider!r}, so this run was "
                f"executed by a hosted API, but no {API_EVIDENCE_FILENAME} "
                "was found in its collection directory; the request identity, "
                "usage and timing for it are missing"
            )
        return None, None

    # Reasons are rendered into a shared HTML report, so they name the
    # artifact rather than the absolute path it lives at. The path itself is
    # already carried, and redacted, by the excluded-run record.
    if not (directory / ARTIFACT_MANIFEST_NAME).is_file():
        return None, (
            f"the collection directory holds an {API_EVIDENCE_FILENAME} but "
            f"no {ARTIFACT_MANIFEST_NAME} completeness marker; the collector "
            "writes that marker last, so its absence means the artifact set "
            "was never finished and nothing in it can be trusted"
        )
    marker_error = _artifact_marker_identity_error(
        directory, run_id=evidence.final_record.run_id
    )
    if marker_error is not None:
        return None, marker_error
    try:
        complete = artifact_set_is_complete(directory)
    except OSError as exc:  # pragma: no cover - defensive
        return None, f"could not verify the artifact set: {exc}"
    if not complete:
        return None, (
            f"the artifact set does not match its own {ARTIFACT_MANIFEST_NAME} "
            "marker; a file was replaced or the write was interrupted, so the "
            "usage and timing in it cannot be trusted"
        )

    try:
        return ApiEvidence.read_json(sidecar), None
    except (OSError, ApiEvidenceError) as exc:
        return None, f"could not read {API_EVIDENCE_FILENAME}: {exc}"


def _artifact_marker_identity_error(directory: Path, *, run_id: str) -> str | None:
    """Refuse a completeness marker that belongs to a different run.

    ``artifact_set_is_complete`` proves the files match the marker, not that
    the marker describes *this* run. A whole collection directory copied from
    another run passes that check while carrying another run's evidence, and
    that evidence would then be read as this run's usage and timing.

    The marker is read through the collector's own bounded regular-file
    reader rather than a bare ``read_text``. That reader is what refuses a
    symlink, a device node or a file large enough to be a denial of service,
    and reading the marker unguarded here would have stepped around the very
    guard the collector added for this file.
    """
    marker_path = directory / ARTIFACT_MANIFEST_NAME
    raw = _read_bounded_regular_file(marker_path, _MAX_ARTIFACT_MANIFEST_BYTES)
    if raw is None:
        return (
            f"{ARTIFACT_MANIFEST_NAME} is not a readable regular file within "
            f"the {_MAX_ARTIFACT_MANIFEST_BYTES} byte limit the collector "
            "writes it under"
        )
    try:
        marker = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        return f"could not parse {ARTIFACT_MANIFEST_NAME}: {exc}"
    if not isinstance(marker, dict):
        return f"{ARTIFACT_MANIFEST_NAME} is not a JSON object"
    recorded = marker.get("run_id")
    if recorded is None:
        return (
            f"{ARTIFACT_MANIFEST_NAME} records no run_id, so there is no way "
            "to tell whether this artifact set belongs to this run"
        )
    if not isinstance(recorded, str) or recorded != run_id:
        return (
            f"{ARTIFACT_MANIFEST_NAME} names run_id {recorded!r} but this run "
            f"is {run_id!r}; the collection directory belongs to a different "
            "run"
        )
    return None


def _provider_agreement_error(
    evidence: RunEvidence, api_evidence: ApiEvidence
) -> str | None:
    """Refuse a sidecar that does not describe the run the record describes.

    The two artifacts are written by the same collector in the same call, so
    any disagreement means one of them is stale, hand-edited, or belongs to a
    different run that happened to be copied into this directory. Either way
    the pair cannot be used to label a system, and every identity the two
    artifacts share is checked before a single number is read out of either.
    """
    record = evidence.final_record

    if api_evidence.schema_version is None:
        return (
            f"{API_EVIDENCE_FILENAME} records no schema_version, so there is "
            "no way to know which contract its fields follow"
        )
    if api_evidence.schema_version != API_EVIDENCE_SCHEMA_VERSION:
        return (
            f"{API_EVIDENCE_FILENAME} declares schema_version "
            f"{api_evidence.schema_version!r}, but this build reads "
            f"{API_EVIDENCE_SCHEMA_VERSION!r}; refusing to read fields under a "
            "contract that may have changed meaning"
        )
    if api_evidence.run_id is None:
        return (
            f"{API_EVIDENCE_FILENAME} records no run_id; a schema "
            f"{API_EVIDENCE_SCHEMA_VERSION} sidecar always carries one, so "
            "this file is not one this build can identify"
        )
    if api_evidence.run_id != record.run_id:
        return (
            f"{API_EVIDENCE_FILENAME} run_id ({api_evidence.run_id!r}) does "
            f"not match final_record.run_id ({record.run_id!r}); the sidecar "
            "describes a different run than the record beside it"
        )
    for field, sidecar_value, record_value in (
        ("plan.model_id", api_evidence.model_id, record.model.model_id),
        (
            "plan.workload_hash",
            api_evidence.workload_hash,
            record.command.workload_hash,
        ),
        ("plan.config_hash", api_evidence.config_hash, record.command.config_hash),
    ):
        if sidecar_value is None:
            return (
                f"{API_EVIDENCE_FILENAME} records no {field}; a schema "
                f"{API_EVIDENCE_SCHEMA_VERSION} sidecar always carries it, so "
                "its absence means this file cannot be checked against the "
                "record it sits beside"
            )
        if record_value is not None and sidecar_value != record_value:
            return (
                f"{API_EVIDENCE_FILENAME} {field} ({sidecar_value!r}) does not "
                f"match the final record ({record_value!r})"
            )
    # Nullability has to agree as well as value. A sidecar that names a
    # revision the record does not, or the reverse, describes a different
    # thing from the record even though no direct comparison fails.
    if (api_evidence.model_revision is None) != (record.model.model_revision is None):
        return (
            f"{API_EVIDENCE_FILENAME} plan.model_revision "
            f"({api_evidence.model_revision!r}) and "
            f"final_record.model.model_revision "
            f"({record.model.model_revision!r}) disagree about whether a model "
            "revision was recorded at all"
        )
    if (
        api_evidence.model_revision is not None
        and record.model.model_revision is not None
        and api_evidence.model_revision != record.model.model_revision
    ):
        return (
            f"{API_EVIDENCE_FILENAME} plan.model_revision "
            f"({api_evidence.model_revision!r}) does not match "
            f"final_record.model.model_revision "
            f"({record.model.model_revision!r})"
        )
    if record.runtime.provider is None:
        # The sidecar's mere existence is the proof: only a hosted-API
        # collector writes one, and it carries provider-reported usage and a
        # client-observed first-token offset. Meanwhile ``SystemKey`` reads
        # locality from the record alone, so accepting the pair would label a
        # hosted run local, publish local-only peak memory for it, and exempt
        # it from every provider-keyed pricing lookup. Keying this on the
        # sidecar's own ``plan.provider`` instead would leave the case where
        # the sidecar never names its provider wide open, which is weaker
        # than the evidence already on disk.
        return (
            f"this run carries {API_EVIDENCE_FILENAME}, which only a hosted "
            "API collector writes, but final_record.runtime.provider is null; "
            "the record claims this run was local and locality is read from "
            "the record, so the pair cannot label a system"
        )
    if (
        api_evidence.provider is not None
        and api_evidence.provider != record.runtime.provider
    ):
        return (
            f"{API_EVIDENCE_FILENAME} plan.provider "
            f"({api_evidence.provider!r}) does not match final_record.runtime."
            f"provider ({record.runtime.provider!r})"
        )
    return None


def _run_seal_error(evidence: RunEvidence) -> str | None:
    """Verify the run-level seal that ``workloads run-api`` writes.

    ``artifact_set_is_complete`` covers only the four files the collector
    writes into the collection directory. ``run-api`` additionally seals the
    whole run directory, hashing the collector's marker together with
    ``final_record.json`` and ``verification.json`` -- the record carrying
    the graded outcome and the summary this loader actually reads. Checking
    it here extends integrity to exactly the two artifacts a comparison
    trusts most and that the collector's own marker never covered.

    Required for hosted runs, optional for local ones. ``workloads run-api``
    writes this seal for every row that produced a full artifact set, so a
    hosted run without one is not something the pipeline produces: it is an
    unsealed directory, and accepting it would let the record and the
    verification -- the graded outcome and the summary this loader reads --
    be edited with nothing to notice. ``workloads run`` writes no seal at
    all, so requiring one there would exclude every local run; the local
    exception is exactly that and nothing wider.
    """
    run_dir = evidence.verification_path.parent
    marker_path = run_dir / RUN_MANIFEST_NAME
    if not marker_path.is_file():
        if evidence.final_record.runtime.provider is not None:
            return (
                f"final_record.runtime.provider is "
                f"{evidence.final_record.runtime.provider!r}, so this run was "
                f"executed by a hosted API and ``workloads run-api`` seals "
                f"every such run directory, but no {RUN_MANIFEST_NAME} is "
                "present; the record and the verification for it are covered "
                "by no integrity marker at all"
            )
        return None
    # The shared verifier reads this marker with an unbounded ``read_text``
    # and follows symlinks. Every other integrity marker this module consumes
    # goes through the collector's bounded no-follow reader first, and this
    # one is no different: check it is a regular file of a sane size, and
    # that it parses, before handing it to a parser that would otherwise
    # read a device node or a file large enough to be a denial of service.
    raw_seal = _read_bounded_regular_file(marker_path, _MAX_ARTIFACT_MANIFEST_BYTES)
    if raw_seal is None:
        return (
            f"{RUN_MANIFEST_NAME} is not a readable regular file within the "
            f"{_MAX_ARTIFACT_MANIFEST_BYTES} byte limit the run pipeline "
            "writes it under"
        )
    try:
        json.loads(raw_seal.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        return f"could not parse {RUN_MANIFEST_NAME}: {exc}"
    try:
        sealed = run_artifacts_are_complete(
            run_dir, expected_run_id=evidence.final_record.run_id
        )
    except OSError as exc:  # pragma: no cover - defensive
        return f"could not verify the run-level seal: {exc}"
    if sealed:
        return None
    return (
        f"the run directory does not match its own {RUN_MANIFEST_NAME} seal; "
        "the verification, the final record or the collector's own marker "
        "was modified after the run landed, so none of them can be trusted"
    )


def _to_system_run(
    evidence: RunEvidence,
) -> tuple[SystemRun | None, ExcludedRun | None]:
    # The artifact-set and sidecar checks run before the run-level seal so
    # the *specific* cause is what gets reported: a missing collector marker
    # or a mismatched sidecar necessarily breaks the seal too, and "the run
    # directory does not match its seal" tells an operator far less than
    # naming the file that is wrong. Nothing is trusted any earlier for it --
    # the seal still has to pass before this run becomes evidence, and every
    # read on the way is bounded, no-follow and hash-checked.
    api_evidence, sidecar_error = _api_evidence_for(evidence)
    if sidecar_error is not None:
        return None, ExcludedRun(
            run_id=evidence.run_id,
            source_results_dir=evidence.source_results_dir,
            reason=sidecar_error,
        )

    if api_evidence is not None:
        disagreement = _provider_agreement_error(evidence, api_evidence)
        if disagreement is not None:
            return None, ExcludedRun(
                run_id=evidence.run_id,
                source_results_dir=evidence.source_results_dir,
                reason=disagreement,
            )

    record = evidence.final_record
    verification = evidence.verification
    if api_evidence is not None:
        settings = api_evidence.decode_settings
    else:
        settings = decode_settings_from_argv(record.command.argv)

    prompt_hash = verification.verified_prompt_hash
    if prompt_hash is None:
        return None, ExcludedRun(
            run_id=evidence.run_id,
            source_results_dir=evidence.source_results_dir,
            reason=(
                "verification.json records no verified_prompt_hash, so this run "
                "cannot be proved to have executed the same prompt as anything "
                "it would be compared against"
            ),
        )

    unit_key = ComparableUnitKey(
        workload_id=verification.workload_id,
        workload_version=verification.workload_version,
        workload_prompt_hash=prompt_hash,
        context_tier=verification.context_tier,
        quality_metric=record.outcome.quality_metric,
        max_output_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        # A local run records no message structure, and neither does an API
        # run whose request was the bare prompt, so both normalize to None
        # and stay comparable. A system prompt or a prepended conversation
        # produces a digest instead, which separates the stratum.
        request_shape=(
            None
            if api_evidence is None
            else request_shape_for(
                api_evidence.messages, workload_prompt_hash=prompt_hash
            )
        ),
    )
    system_key = SystemKey(
        model_id=record.model.model_id,
        model_revision=record.model.model_revision,
        provider=record.runtime.provider,
        runtime_name=record.runtime.name,
        runtime_backend=record.runtime.backend,
        accelerator=record.platform.accelerator,
        quantization=record.model.quantization,
        reasoning_effort=(
            None if api_evidence is None else api_evidence.reasoning_effort
        ),
        decode_mode=verification.decode_mode,
        endpoint=None if api_evidence is None else api_evidence.endpoint,
        thinking_type=None if api_evidence is None else api_evidence.thinking_type,
        # The collector's own configuration identity, carried verbatim. It
        # covers endpoint, model, sampling, provider extensions, finish-reason
        # vocabulary, timeout and system prompt, so a configuration
        # difference this module does not model by name still separates two
        # systems instead of pooling them.
        execution_config_hash=record.command.config_hash,
    )

    seal_error = _run_seal_error(evidence)
    if seal_error is not None:
        return None, ExcludedRun(
            run_id=evidence.run_id,
            source_results_dir=evidence.source_results_dir,
            reason=seal_error,
        )

    try:
        # The explicit checks above name the offending field, which is what
        # an operator needs. This catches whatever they do not.
        #
        # It must hash the dataclasses, not their ``sort_key()`` tuples.
        # ``compare()`` uses the objects themselves as dict keys, and
        # ``sort_key()`` folds every optional field through ``x or default``:
        # a value that is both falsy and unhashable, such as ``[]`` or
        # ``{}``, is replaced by a hashable fallback inside ``sort_key()``
        # while staying unhashable in the dataclass. Probing the tuples would
        # therefore pass and the dict insertion would still raise. Hashing
        # the dataclasses covers both, since a value that hashes here also
        # survives the ``or`` coercion.
        hash((unit_key, system_key))
    except TypeError as exc:
        return None, ExcludedRun(
            run_id=evidence.run_id,
            source_results_dir=evidence.source_results_dir,
            reason=(
                "this run's identity cannot be used as a comparison key: " f"{exc}"
            ),
        )

    return (
        SystemRun(
            run_id=evidence.run_id,
            source_results_dir=evidence.source_results_dir,
            unit_key=unit_key,
            system_key=system_key,
            verification=verification,
            verification_path=evidence.verification_path,
            record=record,
            record_path=evidence.final_record_path,
            decode_settings=settings,
            api_evidence=api_evidence,
        ),
        None,
    )


def _raw_collector_directory_error(results_dir: Path) -> str | None:
    """Explain a directory that is raw collector output, not a results tree.

    ``collect-api`` and ``collect-mlx`` write a flat artifact set into
    whatever ``--output-dir`` they are given. Pointing ``compare --results``
    at one of those is an easy and reasonable mistake, and without this the
    loader simply finds no ``runs/`` tree, reports no comparable units, and
    leaves the operator to work out why.
    """
    if (results_dir / "runs").is_dir():
        return None
    markers = [
        name
        for name in (API_EVIDENCE_FILENAME, "record.json", ARTIFACT_MANIFEST_NAME)
        if (results_dir / name).is_file()
    ]
    if not markers:
        return None
    return (
        f"{results_dir} looks like raw collector output ({', '.join(markers)}) "
        "rather than a results directory. compare reads a "
        "`workloads run --output-dir` tree, which holds runs/<run_id>/"
        "verification.json alongside the final_record.json it points at. A "
        "collector directory on its own has no verification.json, which means "
        "nothing has evaluated the response, so there is no pass rate and no "
        "quality score to compare. Execute the workload matrix and point "
        "--results at that output directory instead."
    )


def load_comparison_evidence(
    results_dirs: tuple[Path, ...],
) -> LoadedComparisonEvidence:
    """Load every comparable run under every given results directory.

    Each directory is loaded independently and merged under a key scoped to
    its own resolved path, rather than under ``run_id`` alone. A matrix
    ``run_id`` names the workload, tier and decode mode and nothing about the
    system, so running one matrix against a local model and against a hosted
    API yields two directories whose run ids collide entirely. Treating that
    as a conflict would reject the exact comparison this command exists for.

    Repeated *references* to one artifact are de-duplicated, so passing the
    same directory twice, or two paths that resolve to it, stays harmless.
    Repeated *measurements* are kept: two results trees holding the same
    matrix row are two repetitions, which is how a policy's
    ``min_measured_repetitions`` is satisfied in the first place.

    Raises ``CompareEvidenceError`` when the input set itself is
    untrustworthy.
    """
    if not results_dirs:
        raise CompareEvidenceError(
            "at least one results directory is required; there is nothing to "
            "compare otherwise"
        )

    for results_dir in results_dirs:
        mistake = _raw_collector_directory_error(results_dir)
        if mistake is not None:
            raise CompareEvidenceError(mistake)

    runs: list[SystemRun] = []
    excluded: list[ExcludedRun] = []
    seen_sources: set[Path] = set()

    for results_dir in results_dirs:
        resolved = results_dir.expanduser().resolve(strict=False)
        if resolved in seen_sources:
            # The same tree named twice. Loading it again would double every
            # measurement in it, which silently inflates evidence counts and
            # narrows every variance figure derived from them.
            continue
        seen_sources.add(resolved)

        try:
            loaded: LoadedEvidence = load_evidence((results_dir,))
        except (ValueError, RecursionError) as exc:
            raise CompareEvidenceError(
                f"could not load evidence from {results_dir}: {exc}"
            ) from exc

        excluded.extend(loaded.excluded)
        for evidence in loaded.usable:
            run, excluded_run = _to_system_run(evidence)
            if run is None:
                if excluded_run is not None:
                    excluded.append(excluded_run)
                continue
            runs.append(run)

    return LoadedComparisonEvidence(
        runs=tuple(_deduplicate(runs)),
        excluded=tuple(excluded),
    )


def _run_fingerprint(run: SystemRun) -> tuple[str, str, str]:
    """A content fingerprint for one run, independent of where it was read.

    The API sidecar is part of the fingerprint, not just the verification and
    the record. Time-to-first-token, cached prompt tokens and reasoning
    tokens are read only from the sidecar and never appear in either of the
    other two artifacts, so two runs that agree on both of those but differ
    on any of that evidence are genuinely different measurements. Comparing
    only the first two would call them identical duplicates and silently drop
    one, taking its usage and its timing with it.
    """
    return (
        json.dumps(run.verification.to_dict(), sort_keys=True),
        json.dumps(run.record.to_dict(), sort_keys=True),
        (
            ""
            if run.api_evidence is None
            else json.dumps(run.api_evidence.fingerprint(), sort_keys=True)
        ),
    )


def _artifact_source(run: SystemRun) -> str:
    """The physical artifact this run was read from, canonicalized.

    Two entries that resolve to the same file are the same evidence reached
    twice; two that resolve to different files are two pieces of evidence,
    whatever their ids say.
    """
    try:
        return str(run.verification_path.resolve(strict=False))
    except OSError:  # pragma: no cover - defensive
        return str(run.verification_path)


def _deduplicate(runs: Sequence[SystemRun]) -> list[SystemRun]:
    """Drop repeated references to one artifact, keep genuine repetitions.

    Repetitions are the ordinary case and must survive. Executing one matrix
    twice into two results directories is exactly how a policy's
    ``min_measured_repetitions`` is satisfied, and every run in the second
    tree shares its ``run_id`` and its whole system identity with the first,
    because a matrix ``run_id`` names the task and nothing about the system
    or the attempt. Treating a repeated ``(system, run_id)`` as a conflict
    therefore rejected the normal way of collecting evidence, and made a
    repetition count above one impossible to reach.

    So identity is not what de-duplicates here; the artifact is. A run is
    dropped only when it is the *same file* reached twice (the same
    directory passed twice, or two paths that resolve to it), or when its
    content is byte-identical to one already kept. Byte-identical includes
    ``started_at``/``ended_at``, which two genuinely separate executions
    cannot share, so that case is a copied tree rather than a repetition and
    counting it twice would inflate the evidence count and understate every
    variance derived from it.
    """
    kept: list[SystemRun] = []
    seen_sources: set[str] = set()
    seen_content: set[tuple[Any, str, tuple[str, str, str]]] = set()
    for run in runs:
        source = _artifact_source(run)
        if source in seen_sources:
            continue
        content = (run.system_key.sort_key(), run.run_id, _run_fingerprint(run))
        if content in seen_content:
            continue
        seen_sources.add(source)
        seen_content.add(content)
        kept.append(run)
    return kept
