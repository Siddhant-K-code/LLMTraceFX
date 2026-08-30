"""Executing the workload matrix against an OpenAI-compatible HTTP API.

This is the remote counterpart to ``workloads.verify``. It answers the
same question that pipeline answers for a local MLX checkpoint -- *did
this model actually solve the task, and how long did it take* -- for a
model that is reachable only over an OpenAI-compatible streaming
chat-completions endpoint such as OpenRouter or Z.ai.

What it reuses rather than re-implements
----------------------------------------
Everything that touches the network, the wire protocol, or a secret
belongs to ``collectors.openai_api`` and is used here unmodified:

* ``collect_openai_stream`` performs the request, decodes the SSE
  stream, classifies provider failures, redacts the credential out of
  every persisted string, and publishes its own artifact set;
* ``build_request_plan`` produces the credential-free plan that
  ``--dry-run`` prints and that this module hashes for resume;
* ``artifact_set_is_complete`` is the *only* thing trusted to say a
  prior collection directory is whole;
* ``redact_text_for_dry_run`` scrubs any diagnostic before it reaches
  ``verification.json``.

Everything about *which rows to run and what the answer was worth*
belongs to the existing workload pipeline and is likewise reused: the
matrix manifest and its hashed prompts, the workload catalog and its
pinned versions, the deterministic evaluators, the ``RowStatus``
vocabulary, and the canonical ``ExperimentRecord``.

What this module adds is the binding between the two, and nothing else.

Per-row data flow
-----------------
1. **Reject unsupported rows explicitly.** A ``native-mtp`` row is never
   executed. Native multi-token prediction is a decoding mechanism
   inside a local runtime; a hosted API's reasoning or "thinking"
   settings are a different thing entirely, and quietly running such a
   row with ``--reasoning-effort`` set would publish evidence labelled
   ``native-mtp`` that measured something else. The row is recorded
   ``UNSUPPORTED`` with that reason.
2. **Verify the prompt.** The prompt text is read from the entry's
   ``prompt_path`` and its sha256 compared with the manifest. A
   mismatch fails the row rather than sending a different prompt than
   the matrix planned.
3. **Verify the workload catalog binding.** Catalog version drift fails
   the row rather than grading against a different spec.
4. **Resume only on complete, hash-verified evidence.** A prior row is
   trusted only when its status is completed/skipped, it was produced
   by *this* backend, and its prompt hash, workload version and API
   binding hash all still match, *and* the collector's own artifact
   marker still verifies every file in the collection directory. A
   stale, partial or tampered artifact set reruns.
5. **Execute through the unmodified collector**, with the matrix row's
   ``max_tokens`` as ``max_output_tokens``. Tripping the event cap fails
   the row as truncated even when a terminal ``finish_reason`` had already
   arrived: this pipeline stopped reading, so what it holds is a prefix of
   the answer.
6. **Evaluate the final answer only.** ``response_text`` is the
   assembled content, never the reasoning stream, so a model that
   reasons its way to the answer and then states something else is
   graded on what it stated. A provider failure short-circuits
   evaluation entirely and is persisted as-is, so a passing evaluator
   can never paper over a request that did not succeed. An evaluator
   that raises leaves the row ``INCONCLUSIVE`` with the measurement
   evidence preserved and ``quality_score`` unset.

Both the planning and the execution path run the collector's public
``assert_credential_not_embedded`` pre-flight *before* building a request
plan. That ordering is load-bearing rather than tidy: an endpoint query
value is folded into the plan's ``config_hash`` as its sha256, that hash
reaches ``verification.json`` through the binding hash, and no redactor
can undo a hash once it has been written down.

No pricing, cost or cross-provider comparison is computed anywhere here.
The provider's own usage counters are persisted by the collector as
reported; turning them into money or into a ranking is a separate
concern with its own correctness burden.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import (
    atomic_write_text,
    config_hash,
    sha256_bytes,
    sha256_text,
)
from ..collectors.openai_api import (
    ARTIFACT_MANIFEST_NAME,
    FAILURE_STREAM_TRUNCATED,
    APICollectionConfig,
    FinishReasonVocabulary,
    HTTPRequest,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    RequestPlan,
    StreamingResponse,
    StreamingTransport,
    UrllibStreamingTransport,
    _contains_credential,
    artifact_set_is_complete,
    assert_credential_not_embedded,
    build_request_plan,
    collect_openai_stream,
    redact_text_for_dry_run,
)
from ..collectors.sse import SSEDecodeError, SSEDecoder
from ..schema import ExperimentRecord, OutcomeInfo, SchemaValidationError, utc_now_iso
from .catalog import workload_by_id
from .evaluators import evaluate_workload
from .matrix import DECODE_MODE_NATIVE_MTP, MatrixEntry, MatrixManifest
from .schema import Workload, WorkloadCategory, WorkloadSchemaError
from .verify import (
    BACKEND_OPENAI_API,
    VERIFICATION_SCHEMA_VERSION,
    RowSelection,
    RowStatus,
    RowVerification,
    VerifyError,
    _record_is_safe_to_resume,
    select_entries,
)

API_BINDING_SCHEMA_VERSION = "1"

#: Default ceiling on dispatched SSE events per row. A streaming endpoint
#: that never stops is bounded by the request timeout, but a *chatty* one
#: can stay under that timeout while emitting far more events than any
#: answer needs, and every event costs a timing row in memory. The cap is
#: part of the binding because changing it changes what a run is willing
#: to observe, which makes two runs under different caps different
#: measurement configurations.
DEFAULT_MAX_STREAM_EVENTS = 10_000

#: The terminal sentinel an OpenAI-compatible stream closes with. Matched
#: here only to stop charging events past it; the collector remains the
#: authority on what the sentinel means.
_DONE_SENTINEL = "[DONE]"

#: Name of the run-level completion marker, written last.
RUN_MANIFEST_NAME = "run.json"
RUN_MANIFEST_SCHEMA_VERSION = "1"

#: Stand-in for a value this module refuses to repeat.
_REJECTED = "[REJECTED]"

#: Exactly what the run-level marker seals, as relative names under the run
#: directory. Fixed here rather than taken from the marker, because the
#: marker is a file and its contents are not authority over which paths
#: this process will open.
_SEALED_COLLECTION_MARKER = f"collection/{ARTIFACT_MANIFEST_NAME}"
_SEALED_ARTIFACT_NAMES = frozenset(
    {_SEALED_COLLECTION_MARKER, "final_record.json", "verification.json"}
)
_SEALED_ARTIFACT_LIMITS = {
    _SEALED_COLLECTION_MARKER: MAX_METADATA_ARTIFACT_BYTES,
    "final_record.json": MAX_EVIDENCE_ARTIFACT_BYTES,
    "verification.json": MAX_METADATA_ARTIFACT_BYTES,
}


def _run_marker_payload(
    *,
    run_id: str,
    collection_dir: Path,
    final_record_path: Path,
    verification_path: Path,
) -> dict[str, Any]:
    """Seal the whole run directory, not just the collector's own set.

    ``artifact_set_is_complete`` covers the four files the collector
    writes and nothing else, so ``final_record.json`` and
    ``verification.json`` sat outside every integrity check: the record
    carrying the graded outcome, and the summary resume actually reads,
    could both be edited and would still be trusted. This marker closes
    that by hashing all three together, including the collector's own
    marker. It is tamper evidence rather than tamper proofing: an
    unsigned marker can itself be rewritten, but a partial write or an
    edit made without regenerating it both stop resume from trusting the
    directory.
    """
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "artifacts": [
            {
                "name": name,
                "sha256": sha256_bytes(
                    read_bounded_regular_bytes(path, _SEALED_ARTIFACT_LIMITS[name])
                ),
            }
            for name, path in (
                (_SEALED_COLLECTION_MARKER, collection_dir / ARTIFACT_MANIFEST_NAME),
                ("final_record.json", final_record_path),
                ("verification.json", verification_path),
            )
        ],
    }


def run_artifacts_are_complete(run_dir: Path, *, expected_run_id: str) -> bool:
    """True when the run directory is whole and unmodified since it landed.

    The marker is removed before the run's files are written and written
    last, so any interruption leaves a directory this rejects rather than
    one that reads as trustworthy.

    Tamper evidence, not tamper proofing: nothing here is signed, so
    anyone able to edit the artifacts can rewrite the marker too. What it
    catches is the partial write, the crash between files, and the edit
    made without regenerating the marker.
    """
    try:
        marker_text = read_bounded_regular_text(
            run_dir / RUN_MANIFEST_NAME, MAX_METADATA_ARTIFACT_BYTES
        )
        marker = json.loads(marker_text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ArtifactReadError, ValueError, RecursionError):
        return False
    if not isinstance(marker, dict):
        return False
    if marker.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        return False
    # A run directory copied from another row carries that row's evidence
    # under this row's name. Every hash still checks out, because nothing
    # in it was edited, so identity has to be asserted rather than
    # inferred from integrity.
    if marker.get("run_id") != expected_run_id:
        return False
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False

    sealed: dict[str, str] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            return False
        name, digest = entry.get("name"), entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            return False
        # The name comes back out of a file, so it is never joined onto the
        # run directory as given. An absolute path, or one containing "..",
        # would otherwise send this check off to hash some unrelated file,
        # and a marker that verifies against a file nobody touched is worse
        # than no marker at all.
        if name not in _SEALED_ARTIFACT_NAMES:
            return False
        sealed[name] = digest

    # Every expected artifact must be named exactly once. A marker that
    # simply omitted an entry would otherwise vouch for a file it never
    # covered.
    if set(sealed) != _SEALED_ARTIFACT_NAMES:
        return False

    for name, digest in sealed.items():
        try:
            raw = read_bounded_regular_bytes(
                run_dir / name, _SEALED_ARTIFACT_LIMITS[name]
            )
        except (OSError, ArtifactReadError):
            return False
        if sha256_bytes(raw) != digest:
            return False
    return True


_TRUSTABLE_RESUME_STATUSES = (RowStatus.COMPLETED, RowStatus.SKIPPED)

_UNSUPPORTED_NATIVE_MTP_REASON = (
    "native-mtp rows are not executable through an OpenAI-compatible API. "
    "Native multi-token prediction is a decoding mechanism inside a local "
    "runtime; a hosted API exposes no such control, and its reasoning or "
    "thinking settings are a different mechanism measuring something else. "
    "This row is rejected rather than silently re-labelled as API reasoning."
)

#: Workload categories whose evaluator executes the model's answer.
#:
#: ``evaluate_code_completion`` writes the candidate to disk and runs it
#: with this interpreter. Locally that is a considered trade: the answer
#: came from a checkpoint on this machine, and the evaluator bounds it
#: with a minimal environment, a new session and POSIX resource limits.
#:
#: None of that is a sandbox. There is no network namespace, no filesystem
#: confinement and no seccomp policy, so the code may open sockets and
#: read anything the invoking user can read. Over an API the answer is
#: produced by a remote party, which turns "grade this workload" into
#: "execute whatever that endpoint returned", and a compromised or hostile
#: endpoint gets arbitrary local code execution out of a benchmark run.
#:
#: So the API backend fails closed. There is deliberately no opt-in flag:
#: a flag would make this a default rather than a boundary, and the honest
#: time to add one is when a real sandbox exists to put behind it.
_EXECUTING_EVALUATOR_CATEGORIES = frozenset({WorkloadCategory.CODE_COMPLETION.value})

_UNSUPPORTED_EXECUTING_EVALUATOR_REASON = (
    "this workload is graded by executing the model's answer as a local "
    "program, which is not something to do with output from a remote "
    "endpoint. The evaluator bounds the process with a minimal environment "
    "and resource limits, but it has no network or filesystem sandbox, so "
    "running an API provider's answer would hand that provider local code "
    "execution. The row is rejected rather than executed; no request is "
    "sent and the evaluator is never invoked. Run this category locally "
    "with `workloads run`, where the answer comes from a checkpoint on "
    "this machine."
)


def _executes_the_answer(workload: Workload) -> bool:
    """True when grading this workload runs the model's answer as a program.

    Asked of the catalog workload rather than of the matrix row, because
    the catalog is what ``evaluate_workload`` dispatches on. A manifest is
    a file on disk, and one that labels a code-completion row as
    ``structured_json`` would otherwise satisfy a check that trusted the
    row and then be graded by executing the answer anyway. The row's own
    category is still checked first, but only as a cheap short circuit;
    the authority is here.
    """
    return workload.category.value in _EXECUTING_EVALUATOR_CATEGORIES


def _unsupported_reason(entry: MatrixEntry) -> str | None:
    """Why this row cannot be run through an API, or ``None`` if it can.

    Checked before anything is planned, sent or graded, so an unsupported
    row never reaches a transport or an evaluator. This looks only at the
    row, so it cannot be the last word on a workload's evaluator: see
    ``_executes_the_answer``, which is re-checked against the catalog once
    the workload has been resolved.
    """
    if entry.decode_mode == DECODE_MODE_NATIVE_MTP:
        return _UNSUPPORTED_NATIVE_MTP_REASON
    if entry.category in _EXECUTING_EVALUATOR_CATEGORIES:
        return _UNSUPPORTED_EXECUTING_EVALUATOR_REASON
    if not entry.runnable:
        return entry.unsupported_reason or _UNSUPPORTED_NATIVE_MTP_REASON
    return None


class APIVerifyError(VerifyError):
    """Raised for invalid API-verify configuration (never a row outcome)."""


# --- Event cap ---------------------------------------------------------------


class _EventCappedResponse:
    """A streaming response that stops once ``limit`` events are exceeded.

    The byte chunks are handed to the collector exactly as the transport
    produced them. Counting is done by feeding a second copy of each
    chunk to the shared :class:`~..collectors.sse.SSEDecoder`, so the
    framing rules used to count are literally the framing rules used to
    parse, and this class contains no parsing of its own.

    Counting never raises. A stream this decoder cannot decode is a
    stream the collector's own decoder is about to reject with the
    authoritative diagnostic, so a decode error here only stops counting
    and lets the bytes through untouched.

    The cap trips only when the stream actually dispatched more events
    than the limit allows. Reaching the limit is not exceeding it, and
    conflating the two condemns clean measurements: a provider may close
    with a terminal ``finish_reason`` and no ``[DONE]`` sentinel, which
    this project supports, and the collector keeps pulling after the final
    chunk.

    The decision is made from the event count alone, never from whether
    another chunk happens to exist. Those are different quantities: a
    chunk may carry several events, or none at all. A keepalive comment, a
    stray blank line and a sentinel arriving in its own socket read are all
    chunks that dispatch nothing, so treating "another chunk arrived" as
    "the cap was exceeded" fails streams that never exceeded it, and makes
    the verdict depend on how the network happened to segment the body.
    Two runs over identical bytes must not disagree because one of them
    was read in smaller pieces.

    Counting only what the shared decoder dispatches keeps the outcome a
    property of the stream rather than of the network. The count is taken
    before the chunk is handed on, so the decision cannot depend on
    whether the collector happened to stop reading first: a body that
    arrives in one large read is capped exactly as the same body split
    across many small ones.

    Counting stops at the ``[DONE]`` sentinel, because the collector stops
    there too. Anything a gateway appends after it -- a duplicated
    sentinel, a stray trailing frame -- is never read by the collector and
    so must never be charged. Charging it would reintroduce exactly the
    dependence this class exists to remove, since those frames are only
    ever seen when they happen to share a socket read with the sentinel.

    The sentinel itself is a dispatched event and is charged like any
    other, so ``limit`` admits ``limit - 1`` content events. That is the
    only reading under which the count is a property of the bytes alone.
    """

    def __init__(self, inner: StreamingResponse, *, limit: int) -> None:
        self._inner = inner
        self._limit = limit
        self._decoder = SSEDecoder()
        self._counting = True
        self.events_seen = 0
        self.cap_tripped = False

    @property
    def status_code(self) -> int:
        return self._inner.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return self._inner.headers

    def _charge(self, chunk: bytes) -> bool:
        """Charge ``chunk``'s events. True when the budget is now exceeded.

        Stops charging at the sentinel and reports no excess for anything
        past it, so a frame the collector will never read cannot decide
        this row's verdict.
        """
        try:
            events = list(self._decoder.feed(chunk))
        except SSEDecodeError:
            # The collector's own decoder is about to reject this with the
            # authoritative diagnostic, so counting stops and the bytes go
            # through untouched.
            self._counting = False
            return False

        for event in events:
            self.events_seen += 1
            if self.events_seen > self._limit:
                return True
            if event.data.strip() == _DONE_SENTINEL:
                # The collector returns here, so nothing after this frame
                # is ever read and nothing after it is ever chargeable.
                self._counting = False
                return False
        return False

    def iter_bytes(self) -> Iterator[bytes]:
        for chunk in self._inner.iter_bytes():
            if self._counting and self._charge(chunk):
                self.cap_tripped = True
                return
            yield chunk

    def close(self) -> None:
        self._inner.close()


class _EventCappedTransport:
    """Wrap a transport so every response it opens is event-capped."""

    def __init__(self, inner: StreamingTransport, *, limit: int) -> None:
        self._inner = inner
        self._limit = limit
        self.last_response: _EventCappedResponse | None = None

    def open_stream(self, request: HTTPRequest) -> StreamingResponse:
        capped = _EventCappedResponse(
            self._inner.open_stream(request), limit=self._limit
        )
        self.last_response = capped
        return capped


# --- Binding -----------------------------------------------------------------


@dataclass(frozen=True)
class APIBinding:
    """Everything about *where and how* rows are sent, minus the secret.

    Validation of the endpoint, provider label, credential variable name
    and request parameters is delegated to ``APICollectionConfig``, which
    already owns those rules; this class only adds the one field the
    collector does not have.
    """

    provider: str
    endpoint: str
    model_id: str
    credential_env_var: str
    model_revision: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    request_timeout_seconds: float = 120.0
    max_stream_events: int = DEFAULT_MAX_STREAM_EVENTS
    extensions: ProviderExtensions = field(default_factory=ProviderExtensions)
    finish_reasons: FinishReasonVocabulary = field(
        default_factory=FinishReasonVocabulary
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_stream_events, bool)
            or not isinstance(self.max_stream_events, int)
            or self.max_stream_events < 1
        ):
            raise APIVerifyError("max_stream_events must be a positive integer")

    def collection_config(
        self,
        entry: MatrixEntry,
        *,
        prompt: str,
        output_dir: Path,
        command_argv: tuple[str, ...],
    ) -> APICollectionConfig:
        """Build the collector config for one matrix row.

        Every validation rule lives in ``APICollectionConfig.__post_init__``
        and is reached from here, so an invalid binding is reported by the
        component that defines what valid means.
        """
        return APICollectionConfig(
            run_id=entry.run_id,
            provider=self.provider,
            endpoint=self.endpoint,
            model_id=self.model_id,
            prompt=prompt,
            output_dir=output_dir,
            command_argv=command_argv,
            credential_env_var=self.credential_env_var,
            system_prompt=self.system_prompt,
            max_output_tokens=entry.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            request_timeout_seconds=self.request_timeout_seconds,
            extensions=self.extensions,
            finish_reasons=self.finish_reasons,
            model_revision=self.model_revision,
        )

    def validate(self) -> None:
        """Raise ``APIVerifyError`` if this binding could never be used.

        Builds one throwaway ``APICollectionConfig`` against placeholder
        row values so that a malformed endpoint, provider label, credential
        variable name or request parameter is reported once, up front,
        instead of once per selected row. Nothing is written and no network
        call is made: ``APICollectionConfig`` validates purely in memory,
        and the placeholders below are only there to satisfy the fields
        this binding does not own.
        """
        placeholder = "binding-validation-probe"
        try:
            APICollectionConfig(
                run_id=placeholder,
                provider=self.provider,
                endpoint=self.endpoint,
                model_id=self.model_id,
                prompt=placeholder,
                output_dir=Path(placeholder),
                command_argv=("llmtracefx-optimizer", "workloads", "run-api"),
                credential_env_var=self.credential_env_var,
                system_prompt=self.system_prompt,
                max_output_tokens=1,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=self.seed,
                request_timeout_seconds=self.request_timeout_seconds,
                extensions=self.extensions,
                finish_reasons=self.finish_reasons,
                model_revision=self.model_revision,
            )
        except OpenAIStreamCollectorError as exc:
            raise APIVerifyError(str(exc)) from exc

    def binding_hash(
        self, *, request_plan: RequestPlan, workload_id: str, workload_version: str
    ) -> str:
        """Identity of everything that affects the request or its grading.

        Built on the collector's own ``config_hash``, which already covers
        the sanitized endpoint identity (origin, path, query keys and
        hashed query values), provider label, model ID and revision, the
        portable request parameters including this row's ``max_tokens``,
        the provider extensions and reasoning settings, the finish-reason
        vocabulary, the request timeout, and the system prompt's hash.
        Added here are the event cap and the evaluation binding, because
        a row graded by a different workload version is not the same
        measurement even when the request is byte-identical.

        The credential environment *variable name* is deliberately not
        hashed, for the same reason the collector excludes it: two runs
        differing only in which variable held the key issue byte-identical
        requests and are graded identically, so it affects neither the
        request nor the evaluation nor resume -- and a caller who pastes a
        key into that slot by mistake must not have a derivation of it
        written into an artifact. The name itself is still recorded, in
        the masked form the request plan chooses.
        """
        return config_hash(
            {
                "binding_schema_version": API_BINDING_SCHEMA_VERSION,
                # Namespaced by backend so an API binding hash can never
                # collide with an MLX run-binding hash written by
                # ``verify.RunBinding`` into the same field.
                "backend": BACKEND_OPENAI_API,
                "request_identity": request_plan.config_hash,
                "max_stream_events": self.max_stream_events,
                "workload_id": workload_id,
                "workload_version": workload_version,
            }
        )


# --- Planning ----------------------------------------------------------------


#: A run_id becomes a directory name, so it has to be one path component
#: and nothing more. The generated ids are ``<workload>-<tier>-<mode>``
#: with an optional ``-depth<n>``, so this is the shape they already have.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _unsafe_run_id_reason(run_id: str) -> str | None:
    """Why ``run_id`` cannot be used as a directory name, or ``None``.

    A run_id is read straight out of the matrix manifest, which is a file
    on disk, and is then joined onto the output directory. An absolute
    value replaces the output directory outright; ``..`` climbs out of it;
    a separator plants artifacts somewhere nobody asked for. None of that
    needs an attacker, either: a hand-edited manifest is enough to have a
    run quietly write outside the directory the caller named.

    The value is not echoed. It failed this check, which is exactly the
    circumstance in which repeating it is a bad idea.
    """
    if not isinstance(run_id, str) or not run_id:
        return "run_id is empty"
    if run_id in (".", ".."):
        return "run_id is a relative directory reference"
    if os.path.isabs(run_id) or os.path.splitdrive(run_id)[0]:
        return "run_id is an absolute path"
    if "/" in run_id or "\\" in run_id or os.sep in run_id:
        return "run_id contains a path separator"
    if "\x00" in run_id:
        return "run_id contains a null byte"
    # ``fullmatch`` rather than ``match``: ``$`` also matches just
    # before a trailing newline, so ``match`` would accept "row\\n" and
    # create a directory whose name the refusal text says is impossible.
    if not _SAFE_RUN_ID.fullmatch(run_id):
        return (
            "run_id is not a single safe path component matching "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return None


def _run_dir(entry: MatrixEntry, *, output_dir: Path) -> Path:
    return output_dir / "runs" / entry.run_id


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    """Resolve a manifest-relative path against the manifest's directory."""
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


def _command_argv(
    entry: MatrixEntry, *, binding: APIBinding, matrix_path: Path
) -> tuple[str, ...]:
    """The credential-free invocation recorded in the row's evidence.

    Reconstructed rather than copied from ``sys.argv`` so that resolved
    defaults are stated explicitly and nothing the caller typed can reach
    an artifact. The collector sanitizes this further before persisting
    it: query values are stripped from the endpoint and the credential
    variable name is masked unless the environment actually defines it.
    """
    argv = [
        "llmtracefx-optimizer",
        "workloads",
        "run-api",
        "--matrix",
        str(matrix_path),
        "--run-id",
        entry.run_id,
        "--provider",
        binding.provider,
        "--endpoint",
        binding.endpoint,
        "--model-id",
        binding.model_id,
        "--api-key-env",
        binding.credential_env_var,
        "--request-timeout",
        str(binding.request_timeout_seconds),
        "--max-stream-events",
        str(binding.max_stream_events),
    ]
    for flag, value in (
        ("--model-revision", binding.model_revision),
        ("--temperature", binding.temperature),
        ("--top-p", binding.top_p),
        ("--seed", binding.seed),
        ("--reasoning-effort", binding.extensions.reasoning_effort),
        ("--thinking", binding.extensions.thinking_type),
        ("--provider-request-id", binding.extensions.provider_request_id),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    if binding.extensions.clear_thinking is not None:
        argv.extend(
            (
                "--clear-thinking",
                "true" if binding.extensions.clear_thinking else "false",
            )
        )
    return tuple(argv)


@dataclass(frozen=True)
class APIRowPlan:
    """A dry-run description of one selected row: what would be sent."""

    entry: MatrixEntry
    unsupported: bool
    unsupported_reason: str | None
    ready: bool
    blockers: tuple[str, ...]
    prompt_path: Path
    collection_dir: Path
    final_record_path: Path
    verification_path: Path
    request_plan: RequestPlan | None
    binding_hash: str | None
    credential_env_var_present: bool
    path_rejected: bool = False
    """True when this row was refused for an unsafe or credential-bearing
    artifact path, in which case its ``run_id`` is withheld from the
    rendered plan along with the paths derived from it."""

    def to_dict(self) -> dict[str, Any]:
        """A secret-safe rendering of this plan.

        The request plan is the collector's own credential-free document:
        it carries message *digests* rather than prompt text, endpoint
        query *keys* rather than values, and header *names* rather than
        header values. Whether the credential variable is defined is
        reported as a boolean, never as its contents.
        """
        return {
            # Withheld on the refusal path for the same reason the paths
            # are: the value that was refused may be the credential.
            "run_id": _REJECTED if self.path_rejected else self.entry.run_id,
            "workload_id": self.entry.workload_id,
            "workload_version": self.entry.workload_version,
            "category": self.entry.category,
            "context_tier": self.entry.context_tier,
            "decode_mode": self.entry.decode_mode,
            "max_tokens": self.entry.max_tokens,
            "status": (
                "unsupported"
                if self.unsupported
                else ("ready" if self.ready else "blocked")
            ),
            "unsupported_reason": self.unsupported_reason,
            "blockers": list(self.blockers),
            "prompt_path": str(self.prompt_path),
            "collection_dir": str(self.collection_dir),
            "final_record_path": str(self.final_record_path),
            "verification_path": str(self.verification_path),
            "api_binding_hash": self.binding_hash,
            "credential_env_var_present": self.credential_env_var_present,
            "request_plan": (
                None if self.request_plan is None else self.request_plan.to_dict()
            ),
        }


def plan_api_row(
    entry: MatrixEntry,
    *,
    manifest_dir: Path,
    matrix_path: Path,
    output_dir: Path,
    binding: APIBinding,
    environ: Mapping[str, str],
) -> APIRowPlan:
    """Describe what running ``entry`` would send, without any network call."""
    run_dir = _run_dir(entry, output_dir=output_dir)
    prompt_path = _resolve_path(entry.prompt_path, base_dir=manifest_dir)
    collection_dir = run_dir / "collection"
    credential = environ.get(binding.credential_env_var, "").strip()
    credential_present = bool(credential)

    # The same refusal the execution path applies, so a dry run reports
    # what a real run would do rather than printing a plan for a row that
    # would then be rejected. The path is not echoed either way: it is
    # rejected precisely because of what it may contain.
    refusal = _unsafe_run_id_reason(entry.run_id)
    if (
        refusal is None
        and credential
        and _contains_credential(str(run_dir), credential)
    ):
        refusal = (
            "the value named by --api-key-env appears in this row's "
            "artifact path; refusing because creating it would write that "
            "value into the filesystem"
        )
    if refusal is not None:
        placeholder = Path(_REJECTED)
        return APIRowPlan(
            entry=entry,
            unsupported=False,
            unsupported_reason=None,
            ready=False,
            blockers=(f"unsafe artifact path: {refusal}",),
            prompt_path=placeholder,
            collection_dir=placeholder,
            final_record_path=placeholder,
            verification_path=placeholder,
            request_plan=None,
            binding_hash=None,
            credential_env_var_present=credential_present,
            path_rejected=True,
        )

    unsupported_reason = _unsupported_reason(entry)
    if unsupported_reason is not None:
        return APIRowPlan(
            entry=entry,
            unsupported=True,
            unsupported_reason=unsupported_reason,
            ready=False,
            blockers=(),
            prompt_path=prompt_path,
            collection_dir=collection_dir,
            final_record_path=run_dir / "final_record.json",
            verification_path=run_dir / "verification.json",
            request_plan=None,
            binding_hash=None,
            credential_env_var_present=credential_present,
        )

    blockers: list[str] = []
    prompt_text: str | None = None
    if not prompt_path.exists():
        blockers.append(f"prompt file missing: {prompt_path}")
    else:
        try:
            prompt_text = prompt_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            blockers.append(f"prompt file unreadable: {exc}")
        else:
            verified_hash = sha256_text(prompt_text)
            if verified_hash != entry.prompt.prompt_hash:
                prompt_text = None
                blockers.append(
                    "prompt hash mismatch: matrix metadata records "
                    f"{entry.prompt.prompt_hash} but {prompt_path} hashes to "
                    f"{verified_hash}; regenerate the matrix or restore the "
                    "original prompt file"
                )

    workload_version: str | None = None
    try:
        workload = workload_by_id(entry.workload_id)
    except KeyError:
        blockers.append(f"unknown workload_id in catalog: {entry.workload_id!r}")
    else:
        # Same authority as the execution path: the catalog decides which
        # evaluator runs, so a manifest that mislabels a code workload
        # cannot present it here as runnable either.
        if _executes_the_answer(workload):
            return APIRowPlan(
                entry=entry,
                unsupported=True,
                unsupported_reason=_UNSUPPORTED_EXECUTING_EVALUATOR_REASON,
                ready=False,
                blockers=(),
                prompt_path=prompt_path,
                collection_dir=collection_dir,
                final_record_path=run_dir / "final_record.json",
                verification_path=run_dir / "verification.json",
                request_plan=None,
                binding_hash=None,
                credential_env_var_present=credential_present,
            )
        workload_version = workload.version
        if workload.version != entry.workload_version:
            workload_version = None
            blockers.append(
                f"workload '{entry.workload_id}' version drift: matrix pinned "
                f"v{entry.workload_version}, catalog has v{workload.version}"
            )

    if not credential_present:
        # The variable name is deliberately not repeated. A name the
        # environment does not define was never proven to be a name, and
        # the likeliest reason it is absent is that the caller put the
        # credential in the name slot; echoing it here would write a
        # secret into the very plan document that promises not to hold
        # one. This mirrors ``_persistable_env_var`` in the collector.
        blockers.append(
            "the environment variable named by --api-key-env is not set or "
            "is empty; export the API key there before running without "
            "--dry-run (it is never accepted as a command argument and never "
            "written to any artifact)"
        )

    request_plan: RequestPlan | None = None
    binding_hash: str | None = None
    if prompt_text is not None:
        try:
            config = binding.collection_config(
                entry,
                prompt=prompt_text,
                output_dir=collection_dir,
                command_argv=_command_argv(
                    entry, binding=binding, matrix_path=matrix_path
                ),
            )
            # The same pre-flight the real run enforces, and it has to come
            # before the plan is built rather than after. A credential
            # sitting in an endpoint query value is folded into
            # ``config_hash`` as ``sha256(value)``, and no redactor can
            # remove a hash once it has been written down. Refusing here
            # keeps a derivation of the secret out of the plan document and
            # out of the binding hash, and stops a dry run green-lighting a
            # configuration the real run would refuse.
            assert_credential_not_embedded(config, environ)
            request_plan = build_request_plan(config, environ=environ)
        except OpenAIStreamCollectorError as exc:
            blockers.append(f"invalid API collector configuration: {exc}")
        else:
            if workload_version is not None:
                binding_hash = binding.binding_hash(
                    request_plan=request_plan,
                    workload_id=entry.workload_id,
                    workload_version=workload_version,
                )

    return APIRowPlan(
        entry=entry,
        unsupported=False,
        unsupported_reason=None,
        ready=not blockers,
        blockers=tuple(blockers),
        prompt_path=prompt_path,
        collection_dir=collection_dir,
        final_record_path=run_dir / "final_record.json",
        verification_path=run_dir / "verification.json",
        request_plan=request_plan,
        binding_hash=binding_hash,
        credential_env_var_present=credential_present,
    )


def plan_selected_api_rows(
    manifest: MatrixManifest,
    *,
    manifest_dir: Path,
    matrix_path: Path,
    output_dir: Path,
    selection: RowSelection,
    binding: APIBinding,
    environ: Mapping[str, str],
) -> tuple[APIRowPlan, ...]:
    """Dry-run: describe every selected row without touching the network."""
    return tuple(
        plan_api_row(
            entry,
            manifest_dir=manifest_dir,
            matrix_path=matrix_path,
            output_dir=output_dir,
            binding=binding,
            environ=environ,
        )
        for entry in select_entries(manifest, selection)
    )


def render_plan_document(
    plans: tuple[APIRowPlan, ...], *, binding: APIBinding, environ: Mapping[str, str]
) -> str:
    """Render the dry-run plan as secret-safe JSON.

    Scrubbed as a whole document as well as field by field. The
    individual fields are already credential-free by construction, but a
    caller who pasted the key into ``--endpoint`` would otherwise have it
    echoed back from a field that is normally harmless, so the assembled
    text gets the collector's redactor passed over it too.
    """
    credential = environ.get(binding.credential_env_var, "").strip() or None
    payload = {
        "dry_run": True,
        "network_request_performed": False,
        "backend": BACKEND_OPENAI_API,
        "credential_env_var_present": credential is not None,
        "rows": [plan.to_dict() for plan in plans],
    }
    return redact_text_for_dry_run(
        json.dumps(payload, indent=2, allow_nan=False), credential
    )


# --- Execution ---------------------------------------------------------------


@dataclass(frozen=True)
class APIRowResult:
    """Outcome of planning/executing one selected matrix row over the API."""

    entry: MatrixEntry
    verification: RowVerification
    final_record: ExperimentRecord | None


def _load_prior_verification(path: Path) -> RowVerification | None:
    if not path.exists():
        return None
    try:
        return RowVerification.read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, VerifyError):
        # A corrupt or partial artifact from an interrupted run must never
        # be mistaken for a trustworthy completed result.
        return None


def _total_ms(record: ExperimentRecord | None) -> float | None:
    if record is None or record.timing.total is None:
        return None
    return record.timing.total.value


def execute_api_row(
    entry: MatrixEntry,
    *,
    manifest_dir: Path,
    matrix_path: Path,
    output_dir: Path,
    binding: APIBinding,
    resume: bool,
    transport_factory: Callable[[], StreamingTransport],
    environ: Mapping[str, str] | None = None,
) -> APIRowResult:
    """Verify, (maybe) execute, and evaluate one matrix row over the API.

    ``transport_factory`` is invoked only immediately before a request is
    actually made, so a batch of unsupported, blocked or resumed rows
    never constructs a transport and therefore never opens a socket.
    """
    resolved_environ = os.environ if environ is None else environ
    credential = resolved_environ.get(binding.credential_env_var, "").strip() or None
    started_at = utc_now_iso()

    # Before a path is derived, let alone created. Everything below joins
    # run_id onto the output directory and then writes there, including
    # the rejection paths, so an unsafe id would have its refusal written
    # to the very place it should not reach. Nothing is written here for
    # the same reason: there is no directory this row may safely touch.
    refusal = _unsafe_run_id_reason(entry.run_id)
    if refusal is None and credential is not None:
        candidate = _run_dir(entry, output_dir=output_dir)
        if _contains_credential(str(candidate), credential):
            refusal = (
                "the value named by --api-key-env appears in this row's "
                "artifact path; refusing because creating it would write "
                "that value into the filesystem"
            )
    if refusal is not None:
        return APIRowResult(
            entry=entry,
            verification=RowVerification(
                schema_version=VERIFICATION_SCHEMA_VERSION,
                run_id=_REJECTED,
                workload_id=entry.workload_id,
                workload_version=entry.workload_version,
                category=entry.category,
                context_tier=entry.context_tier,
                decode_mode=entry.decode_mode,
                status=RowStatus.FAILED,
                reason=f"unsafe artifact path: {refusal}",
                recorded_prompt_hash=entry.prompt.prompt_hash,
                verified_prompt_hash=None,
                run_binding_hash=None,
                resumed=False,
                outcome_success=None,
                quality_score=None,
                total_ms=None,
                started_at=started_at,
                ended_at=utc_now_iso(),
                final_record_path=None,
                collection_dir=None,
                backend=BACKEND_OPENAI_API,
                provider=None,
                api_model_id=None,
                artifacts_verified=None,
            ),
            final_record=None,
        )

    run_dir = _run_dir(entry, output_dir=output_dir)
    verification_path = run_dir / "verification.json"
    collection_dir = run_dir / "collection"
    final_record_path = run_dir / "final_record.json"

    def _scrub(text: str) -> str:
        return redact_text_for_dry_run(text, credential)

    def _finish(
        status: RowStatus,
        reason: str | None,
        *,
        final_record: ExperimentRecord | None = None,
        verified_hash: str | None = None,
        binding_hash: str | None = None,
        resumed: bool = False,
        wrote_collection: bool = False,
        artifacts_verified: bool | None = None,
    ) -> APIRowResult:
        verification = RowVerification(
            schema_version=VERIFICATION_SCHEMA_VERSION,
            run_id=entry.run_id,
            workload_id=entry.workload_id,
            workload_version=entry.workload_version,
            category=entry.category,
            context_tier=entry.context_tier,
            decode_mode=entry.decode_mode,
            status=status,
            reason=None if reason is None else _scrub(reason),
            recorded_prompt_hash=entry.prompt.prompt_hash,
            verified_prompt_hash=verified_hash,
            run_binding_hash=binding_hash,
            resumed=resumed,
            outcome_success=(
                final_record.outcome.success if final_record is not None else None
            ),
            quality_score=(
                final_record.outcome.quality_score if final_record is not None else None
            ),
            total_ms=_total_ms(final_record),
            started_at=started_at,
            ended_at=utc_now_iso(),
            final_record_path=(
                str(final_record_path) if final_record is not None else None
            ),
            collection_dir=(
                str(collection_dir)
                if final_record is not None and wrote_collection
                else None
            ),
            backend=BACKEND_OPENAI_API,
            # Scrubbed like every other persisted string. The pre-flight
            # above refuses a run whose provider label or model ID contains
            # the credential, but this artifact is written on paths that
            # never reach a request at all, so it cannot rely on that
            # refusal having happened. Both fields are raw caller input and
            # both are shapes a real key can take.
            provider=_scrub(binding.provider),
            api_model_id=_scrub(binding.model_id),
            artifacts_verified=artifacts_verified,
        )
        atomic_write_text(verification_path, verification.to_json())

        # The run-level marker is written last and removed first, so a
        # crash anywhere above leaves a directory `run_artifacts_are_
        # complete` rejects rather than one that reads as trustworthy. It
        # is only meaningful for a row that produced a full artifact set;
        # a rejected or failed-before-execution row has nothing to seal.
        if final_record is not None and wrote_collection and artifacts_verified:
            try:
                atomic_write_text(
                    run_dir / RUN_MANIFEST_NAME,
                    json.dumps(
                        _run_marker_payload(
                            run_id=entry.run_id,
                            collection_dir=collection_dir,
                            final_record_path=final_record_path,
                            verification_path=verification_path,
                        ),
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                )
            except OSError:
                # A marker that cannot be written simply is not written.
                # The row's evidence still stands on its own; only resume
                # declines to trust it, which is the safe direction.
                pass

        return APIRowResult(
            entry=entry, verification=verification, final_record=final_record
        )

    # 1. Unsupported rows are rejected first, before a transport is built
    #    or an evaluator is chosen. Native-MTP is never remapped onto API
    #    reasoning, and a workload graded by executing the answer is never
    #    handed output from a remote endpoint.
    unsupported_reason = _unsupported_reason(entry)
    if unsupported_reason is not None:
        return _finish(RowStatus.UNSUPPORTED, unsupported_reason)

    # 2. Verify the fully materialized prompt against the matrix metadata.
    prompt_path = _resolve_path(entry.prompt_path, base_dir=manifest_dir)
    if not prompt_path.exists():
        return _finish(RowStatus.FAILED, f"prompt file missing: {prompt_path}")

    try:
        prompt_text = prompt_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return _finish(RowStatus.FAILED, f"prompt file unreadable: {exc}")

    verified_hash = sha256_text(prompt_text)
    if verified_hash != entry.prompt.prompt_hash:
        return _finish(
            RowStatus.FAILED,
            "prompt hash mismatch: matrix metadata records "
            f"{entry.prompt.prompt_hash} but {prompt_path} hashes to "
            f"{verified_hash}; regenerate the matrix or restore the original "
            "prompt file before running",
            verified_hash=verified_hash,
        )

    # 3. Verify the workload catalog binding.
    try:
        workload = workload_by_id(entry.workload_id)
    except KeyError:
        return _finish(
            RowStatus.FAILED,
            f"unknown workload_id in catalog: {entry.workload_id!r}",
            verified_hash=verified_hash,
        )
    if workload.version != entry.workload_version:
        return _finish(
            RowStatus.FAILED,
            f"workload '{entry.workload_id}' version drift: matrix pinned "
            f"v{entry.workload_version}, catalog has v{workload.version}; "
            "regenerate the matrix",
            verified_hash=verified_hash,
        )

    # The catalog decides which evaluator runs, so the catalog decides
    # whether this row is safe to run at all. Checking only the manifest's
    # category left a mislabelled row executing a remote answer.
    if _executes_the_answer(workload):
        return _finish(
            RowStatus.UNSUPPORTED,
            _UNSUPPORTED_EXECUTING_EVALUATOR_REASON,
            verified_hash=verified_hash,
        )

    # 4. Build the collector config and derive this row's binding hash.
    try:
        config = binding.collection_config(
            entry,
            prompt=prompt_text,
            output_dir=collection_dir,
            command_argv=_command_argv(entry, binding=binding, matrix_path=matrix_path),
        )
        # Before the plan, not after. ``build_request_plan`` folds an
        # endpoint query *value* into ``config_hash`` as its sha256, and
        # that hash is persisted in this row's ``run_binding_hash``. A
        # credential parked there would leave a derivation of itself in an
        # artifact even though the request is about to be refused, and no
        # redactor can undo a hash.
        assert_credential_not_embedded(config, resolved_environ)
        request_plan = build_request_plan(config, environ=resolved_environ)
    except OpenAIStreamCollectorError as exc:
        return _finish(
            RowStatus.FAILED,
            f"invalid API collector configuration: {exc}",
            verified_hash=verified_hash,
        )

    binding_hash = binding.binding_hash(
        request_plan=request_plan,
        workload_id=entry.workload_id,
        workload_version=workload.version,
    )

    # 5. Resume only on a complete, hash-verified artifact set from this
    #    backend. Every condition below has to hold: a matching hash over
    #    an incomplete directory, or a complete directory whose files were
    #    edited after the marker was written, both rerun.
    if resume:
        prior = _load_prior_verification(verification_path)
        if (
            prior is not None
            and prior.status in _TRUSTABLE_RESUME_STATUSES
            and prior.backend == BACKEND_OPENAI_API
            and prior.verified_prompt_hash == verified_hash
            and prior.run_binding_hash == binding_hash
            and prior.workload_version == workload.version
            # The summary must be this row's, not one copied in from
            # another. Integrity says the bytes are unedited; it says
            # nothing about which row they describe.
            and prior.run_id == entry.run_id
            and artifact_set_is_complete(collection_dir)
            # Seals final_record.json and verification.json too, which the
            # collector's own marker does not cover, and asserts the same
            # identity over the directory as a whole.
            and run_artifacts_are_complete(run_dir, expected_run_id=entry.run_id)
        ):
            try:
                trusted_record = ExperimentRecord.read_json(final_record_path)
            except (OSError, UnicodeError, SchemaValidationError):
                trusted_record = None
            if trusted_record is not None and not _record_is_safe_to_resume(
                trusted_record,
                expected_run_id=entry.run_id,
                expected_model_id=binding.model_id,
            ):
                trusted_record = None
            if trusted_record is not None:
                return _finish(
                    RowStatus.SKIPPED,
                    "trusted prior completed artifact set (hash-verified and "
                    "complete); not re-executed",
                    final_record=trusted_record,
                    verified_hash=verified_hash,
                    binding_hash=binding_hash,
                    resumed=True,
                    wrote_collection=True,
                    artifacts_verified=True,
                )

    # 6. Execute through the unmodified collector. Resume has declined to
    #    trust whatever was here, so the marker vouching for it is dropped
    #    before the first byte is overwritten and rewritten only once the
    #    replacement set is complete.
    (run_dir / RUN_MANIFEST_NAME).unlink(missing_ok=True)

    capped = _EventCappedTransport(transport_factory(), limit=binding.max_stream_events)
    try:
        collection_result = collect_openai_stream(
            config, transport=capped, environ=resolved_environ
        )
    except (OSError, OpenAIStreamCollectorError) as exc:
        # A missing credential, an unusable environment or a failed
        # artifact write never describes a request whose result could be
        # graded, so it fails the row rather than producing evidence.
        return _finish(
            RowStatus.FAILED,
            f"API collection could not be attempted: {exc}",
            verified_hash=verified_hash,
            binding_hash=binding_hash,
        )

    artifacts_verified = artifact_set_is_complete(collection_dir)
    collected_record = collection_result.record
    cap_tripped = capped.last_response is not None and capped.last_response.cap_tripped
    cap_note = (
        f"the stream was abandoned after the configured "
        f"{binding.max_stream_events}-event cap, so the answer cannot be "
        "known to be whole"
    )

    if not collected_record.outcome.success:
        # A provider failure is evidence in its own right and is never
        # overwritten by an evaluator verdict; the evaluator is not run.
        failure = collection_result.evidence.failure
        detail = (
            "collector reported failure without an error detail"
            if failure is None
            else f"{failure.category}: {failure.message}"
        )
        if cap_tripped:
            detail += f" ({cap_note})"
        collected_record.write_json(final_record_path)
        return _finish(
            RowStatus.FAILED,
            detail,
            final_record=collected_record,
            verified_hash=verified_hash,
            binding_hash=binding_hash,
            wrote_collection=True,
            artifacts_verified=artifacts_verified,
        )

    if cap_tripped:
        # Cutting the stream short can land after a terminal finish_reason,
        # which is enough for the collector to call the stream cleanly
        # ended. It is not enough for us: we are the ones who stopped
        # reading, so what arrived is a prefix of the answer and grading it
        # would publish a truncation as a verdict. The collected outcome is
        # replaced rather than the record being dropped, so the timing and
        # usage evidence survives while the outcome stops claiming success.
        capped_record = dataclasses.replace(
            collected_record,
            outcome=OutcomeInfo(
                success=False,
                quality_score=None,
                quality_metric=None,
                notes=cap_note,
            ),
        )
        capped_record.write_json(final_record_path)
        return _finish(
            RowStatus.FAILED,
            f"{FAILURE_STREAM_TRUNCATED}: {cap_note}",
            final_record=capped_record,
            verified_hash=verified_hash,
            binding_hash=binding_hash,
            wrote_collection=True,
            artifacts_verified=artifacts_verified,
        )

    # 7. Grade the final answer only. ``response_text`` is the assembled
    #    content stream; reasoning deltas are not part of it.
    try:
        evaluated_outcome = evaluate_workload(workload, collection_result.response_text)
    except (WorkloadSchemaError, OSError, RuntimeError) as exc:
        inconclusive_record = dataclasses.replace(
            collected_record,
            outcome=OutcomeInfo(
                success=collected_record.outcome.success,
                quality_score=None,
                quality_metric=None,
                notes=_scrub(f"evaluation inconclusive: {exc}"),
            ),
        )
        inconclusive_record.write_json(final_record_path)
        return _finish(
            RowStatus.INCONCLUSIVE,
            f"evaluator raised an unexpected error: {exc}",
            final_record=inconclusive_record,
            verified_hash=verified_hash,
            binding_hash=binding_hash,
            wrote_collection=True,
            artifacts_verified=artifacts_verified,
        )

    final_record = dataclasses.replace(collected_record, outcome=evaluated_outcome)
    final_record.write_json(final_record_path)
    return _finish(
        RowStatus.COMPLETED,
        None,
        final_record=final_record,
        verified_hash=verified_hash,
        binding_hash=binding_hash,
        wrote_collection=True,
        artifacts_verified=artifacts_verified,
    )


def run_selected_api_rows(
    manifest: MatrixManifest,
    *,
    manifest_dir: Path,
    matrix_path: Path,
    output_dir: Path,
    selection: RowSelection,
    binding: APIBinding,
    resume: bool,
    transport_factory: Callable[[], StreamingTransport] = UrllibStreamingTransport,
    environ: Mapping[str, str] | None = None,
) -> tuple[APIRowResult, ...]:
    """Execute and evaluate every selected matrix row in manifest order."""
    return tuple(
        execute_api_row(
            entry,
            manifest_dir=manifest_dir,
            matrix_path=matrix_path,
            output_dir=output_dir,
            binding=binding,
            resume=resume,
            transport_factory=transport_factory,
            environ=environ,
        )
        for entry in select_entries(manifest, selection)
    )
