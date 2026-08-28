"""Executable, resumable, quality-aware verification pipeline.

This module turns the dry-run matrix produced by ``matrix.generate_matrix``
into real, evaluated evidence. It is the foundation the future ``tune``
command will build on: a user with an existing local MLX model (and,
optionally, an existing local draft model) selects rows from a matrix
manifest and this module executes each one through the existing MLX-LM
collector (``collectors.mlx.collect_mlx``, added in PR #4), evaluates the
response with the deterministic PR #5 evaluators, and persists a canonical
``ExperimentRecord`` whose ``OutcomeInfo`` reflects task quality.

Data flow per selected row
--------------------------
1. **Reject unsupported rows explicitly.** Any row whose ``decode_mode`` is
   ``native-mtp`` (or that the matrix already marked ``runnable=False``) is
   never executed and never silently downgraded to generic draft-model
   speculation; it is recorded as ``RowStatus.UNSUPPORTED`` with the exact
   reason. Native-MTP support does not exist in this project (see
   ``collectors.native_mtp``'s capability detection) and this pipeline does
   not add one.
2. **Verify the prompt.** The exact fully materialized prompt text is read
   from the entry's ``prompt_path`` (never from an in-memory copy) and its
   sha256 hash is compared against the hash recorded in the matrix
   manifest. A mismatch means the prompt file was edited (or the manifest
   is stale) after matrix generation, so the row fails clearly instead of
   silently executing a different prompt than the one the matrix planned.
3. **Verify the workload catalog binding.** The workload is looked up by
   ``workload_id`` and its ``version`` must match the version pinned in the
   manifest; catalog drift fails the row clearly rather than evaluating
   against a different (possibly incompatible) spec.
4. **Resume by trusting only hash-matching completed artifacts.** If a
   prior ``verification.json`` for this run_id exists, has a status of
   ``completed``/``skipped``, and its recorded prompt hash / run-binding
   hash (target+draft model paths, seed, max_tokens, num_draft_tokens) /
   workload version all still match, the prior ``final_record.json`` is
   trusted and reused (``RowStatus.SKIPPED``) without re-executing.
   Otherwise the row is (re-)executed.
5. **Execute via the existing MLX-LM collector.** Only ``collect_mlx`` is
   used; models are never downloaded (``RunBinding`` requires the target
   and any draft path to already exist on disk). Supplying
   ``--draft-model-path`` enables generic external draft-model speculative
   decoding (labeled ``"draft-model"`` by the collector, per PR #5's
   native-MTP-vs-draft-model distinction) -- it is never applied to
   native-MTP rows.
6. **Evaluate without ever overwriting a runtime failure.** If the
   collector reports a runtime failure (``outcome.success is False``), the
   collected record is persisted unchanged as the final record
   (``RowStatus.FAILED``); the evaluator is not invoked. If collection
   succeeds, the deterministic evaluator's ``OutcomeInfo`` replaces the
   collector's placeholder outcome in the final record
   (``RowStatus.COMPLETED``). If the evaluator itself raises an unexpected
   error despite a successful collection, the row is recorded as
   ``RowStatus.INCONCLUSIVE`` with measurement evidence preserved and
   ``quality_score`` left ``None`` (never guessed).
7. **Persist atomically.** ``collect_mlx`` already atomically writes
   ``record.json``/``response.txt``/``environment.json`` under the row's
   ``collection/`` directory; this module additionally writes
   ``final_record.json`` (the evaluated canonical record) and
   ``verification.json`` (a machine-readable summary with statuses,
   hashes, and artifact paths) using the same atomic write-then-rename
   helper used throughout this project.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text, config_hash, sha256_text
from ..collectors.mlx import (
    MLXCollectionConfig,
    MLXCollectorError,
    MLXRuntime,
    collect_mlx,
)
from ..schema import ExperimentRecord, OutcomeInfo, SchemaValidationError, utc_now_iso
from .catalog import workload_by_id
from .evaluators import evaluate_workload
from .matrix import DECODE_MODE_NATIVE_MTP, MatrixEntry, MatrixManifest
from .schema import WorkloadSchemaError

VERIFICATION_SCHEMA_VERSION = "1"


class VerifyError(ValueError):
    """Raised for invalid verify-pipeline configuration (not row outcomes)."""


class RowStatus(str, Enum):
    """Explicit, machine-readable outcome for one selected matrix row."""

    COMPLETED = "completed"
    """Executed and evaluated; ``outcome`` in the final record reflects
    task quality (which may itself be a pass or a fail)."""

    FAILED = "failed"
    """A runtime/collector failure, a stale prompt/catalog mismatch, or an
    invalid collector configuration prevented producing evidence."""

    UNSUPPORTED = "unsupported"
    """The row's decode mode is not runnable (native-MTP); rejected
    explicitly rather than silently downgraded to generic speculation."""

    SKIPPED = "skipped"
    """A prior hash-matching completed artifact was trusted and reused
    instead of re-executing (resume)."""

    INCONCLUSIVE = "inconclusive"
    """Collection succeeded but the deterministic evaluator could not
    render a verdict (an unexpected, non-quality-related error); timing
    evidence is preserved but ``quality_score`` is left unset."""


_TRUSTABLE_RESUME_STATUSES = (RowStatus.COMPLETED, RowStatus.SKIPPED)


@dataclass(frozen=True)
class RowSelection:
    """Row selection filters for ``workloads run``.

    Each field is either ``None`` (no filter on that axis) or a
    ``frozenset`` of allowed values; a row must match every non-``None``
    filter to be selected. An empty ``RowSelection()`` selects every row
    in the manifest.
    """

    run_ids: frozenset[str] | None = None
    categories: frozenset[str] | None = None
    context_tiers: frozenset[str] | None = None
    decode_modes: frozenset[str] | None = None

    def matches(self, entry: MatrixEntry) -> bool:
        if self.run_ids is not None and entry.run_id not in self.run_ids:
            return False
        if self.categories is not None and entry.category not in self.categories:
            return False
        if (
            self.context_tiers is not None
            and entry.context_tier not in self.context_tiers
        ):
            return False
        if self.decode_modes is not None and entry.decode_mode not in self.decode_modes:
            return False
        return True


def select_entries(
    manifest: MatrixManifest, selection: RowSelection
) -> tuple[MatrixEntry, ...]:
    """Return the manifest entries matching ``selection``, in manifest order."""
    return tuple(entry for entry in manifest.entries if selection.matches(entry))


def _resolve_path(raw: str, *, base_dir: Path) -> Path:
    """Resolve a manifest-relative path against the manifest's directory.

    Mirrors ``RunnerConfig.from_file``'s handling of relative
    ``results_dir``: paths recorded in the manifest are relative to the
    directory the manifest itself was generated into, not the current
    process's working directory, so a manifest can be consumed correctly
    regardless of where ``workloads run`` happens to be invoked from.
    """
    path = Path(raw)
    return path if path.is_absolute() else base_dir / path


@dataclass(frozen=True)
class RunBinding:
    """Explicit, no-download local path binding for one verify run.

    Requires the target model path (and, if given, the draft model path)
    to already exist on disk. A model identifier is never turned into an
    implicit download by this pipeline.
    """

    target_model_path: Path
    draft_model_path: Path | None = None
    seed: int = 0
    num_draft_tokens: int = 2

    def __post_init__(self) -> None:
        if not self.target_model_path.exists():
            raise VerifyError(
                f"target model path does not exist: {self.target_model_path}. "
                "Provide an existing local model path; models are never "
                "downloaded by this pipeline."
            )
        if self.draft_model_path is not None and not self.draft_model_path.exists():
            raise VerifyError(
                f"draft model path does not exist: {self.draft_model_path}. "
                "Provide an existing local model path; models are never "
                "downloaded by this pipeline."
            )

    def hash_payload(self, *, max_tokens: int) -> dict[str, Any]:
        return {
            "target_model_path": str(self.target_model_path),
            "draft_model_path": (
                str(self.draft_model_path)
                if self.draft_model_path is not None
                else None
            ),
            "seed": self.seed,
            "num_draft_tokens": self.num_draft_tokens,
            "max_tokens": max_tokens,
        }

    def run_binding_hash(self, *, max_tokens: int) -> str:
        return config_hash(self.hash_payload(max_tokens=max_tokens))


@dataclass(frozen=True)
class RowVerification:
    """Machine-readable per-row verification summary (``verification.json``)."""

    schema_version: str
    run_id: str
    workload_id: str
    workload_version: str
    category: str
    context_tier: str
    decode_mode: str
    status: RowStatus
    reason: str | None
    recorded_prompt_hash: str | None
    verified_prompt_hash: str | None
    run_binding_hash: str | None
    resumed: bool
    outcome_success: bool | None
    quality_score: float | None
    total_ms: float | None
    started_at: str
    ended_at: str
    final_record_path: str | None
    collection_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "category": self.category,
            "context_tier": self.context_tier,
            "decode_mode": self.decode_mode,
            "status": self.status.value,
            "reason": self.reason,
            "recorded_prompt_hash": self.recorded_prompt_hash,
            "verified_prompt_hash": self.verified_prompt_hash,
            "run_binding_hash": self.run_binding_hash,
            "resumed": self.resumed,
            "outcome_success": self.outcome_success,
            "quality_score": self.quality_score,
            "total_ms": self.total_ms,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "final_record_path": self.final_record_path,
            "collection_dir": self.collection_dir,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RowVerification:
        try:
            return cls(
                schema_version=str(
                    data.get("schema_version", VERIFICATION_SCHEMA_VERSION)
                ),
                run_id=data["run_id"],
                workload_id=data["workload_id"],
                workload_version=data["workload_version"],
                category=data["category"],
                context_tier=data["context_tier"],
                decode_mode=data["decode_mode"],
                status=RowStatus(data["status"]),
                reason=data.get("reason"),
                recorded_prompt_hash=data.get("recorded_prompt_hash"),
                verified_prompt_hash=data.get("verified_prompt_hash"),
                run_binding_hash=data.get("run_binding_hash"),
                resumed=bool(data.get("resumed", False)),
                outcome_success=data.get("outcome_success"),
                quality_score=data.get("quality_score"),
                total_ms=data.get("total_ms"),
                started_at=data["started_at"],
                ended_at=data["ended_at"],
                final_record_path=data.get("final_record_path"),
                collection_dir=data.get("collection_dir"),
            )
        except (KeyError, ValueError) as exc:
            raise VerifyError(f"invalid verification.json: {exc}") from exc

    @classmethod
    def read_json(cls, path: Path) -> RowVerification:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load_prior_verification(path: Path) -> RowVerification | None:
    if not path.exists():
        return None
    try:
        return RowVerification.read_json(path)
    except (OSError, json.JSONDecodeError, VerifyError):
        # A corrupt/partial artifact from an interrupted previous run must
        # never be mistaken for a trustworthy completed result.
        return None


def _total_ms(record: ExperimentRecord | None) -> float | None:
    if record is None or record.timing.total is None:
        return None
    return record.timing.total.value


@dataclass(frozen=True)
class RowResult:
    """Outcome of planning/executing one selected matrix row."""

    entry: MatrixEntry
    verification: RowVerification
    final_record: ExperimentRecord | None


@dataclass(frozen=True)
class RowPlan:
    """A dry-run description of one selected row: what would happen."""

    entry: MatrixEntry
    unsupported: bool
    unsupported_reason: str | None
    ready: bool
    blockers: tuple[str, ...]
    prompt_path: Path
    collection_dir: Path
    final_record_path: Path
    verification_path: Path


def _run_dir(entry: MatrixEntry, *, output_dir: Path) -> Path:
    return output_dir / "runs" / entry.run_id


def plan_row(
    entry: MatrixEntry,
    *,
    manifest_dir: Path,
    output_dir: Path,
    binding: RunBinding | None,
) -> RowPlan:
    """Describe what running ``entry`` would do, without loading a model."""
    run_dir = _run_dir(entry, output_dir=output_dir)
    prompt_path = _resolve_path(entry.prompt_path, base_dir=manifest_dir)

    if entry.decode_mode == DECODE_MODE_NATIVE_MTP or not entry.runnable:
        reason = entry.unsupported_reason or (
            "native-mtp execution is not implemented by this pipeline; no "
            "capable runtime is wired into `workloads run`"
        )
        return RowPlan(
            entry=entry,
            unsupported=True,
            unsupported_reason=reason,
            ready=False,
            blockers=(),
            prompt_path=prompt_path,
            collection_dir=run_dir / "collection",
            final_record_path=run_dir / "final_record.json",
            verification_path=run_dir / "verification.json",
        )

    blockers: list[str] = []
    if binding is None:
        blockers.append("no --model-path binding was provided")

    if not prompt_path.exists():
        blockers.append(f"prompt file missing: {prompt_path}")
    else:
        verified_hash = sha256_text(prompt_path.read_text(encoding="utf-8"))
        if verified_hash != entry.prompt.prompt_hash:
            blockers.append(
                "prompt hash mismatch: matrix metadata records "
                f"{entry.prompt.prompt_hash} but {prompt_path} hashes to "
                f"{verified_hash}; regenerate the matrix or restore the "
                "original prompt file"
            )

    try:
        workload = workload_by_id(entry.workload_id)
    except KeyError:
        blockers.append(f"unknown workload_id in catalog: {entry.workload_id!r}")
    else:
        if workload.version != entry.workload_version:
            blockers.append(
                f"workload '{entry.workload_id}' version drift: matrix "
                f"pinned v{entry.workload_version}, catalog has "
                f"v{workload.version}"
            )

    return RowPlan(
        entry=entry,
        unsupported=False,
        unsupported_reason=None,
        ready=not blockers,
        blockers=tuple(blockers),
        prompt_path=prompt_path,
        collection_dir=run_dir / "collection",
        final_record_path=run_dir / "final_record.json",
        verification_path=run_dir / "verification.json",
    )


def plan_selected_rows(
    manifest: MatrixManifest,
    *,
    manifest_dir: Path,
    output_dir: Path,
    selection: RowSelection,
    binding: RunBinding | None,
) -> tuple[RowPlan, ...]:
    """Dry-run: describe every selected row without loading any model."""
    return tuple(
        plan_row(
            entry, manifest_dir=manifest_dir, output_dir=output_dir, binding=binding
        )
        for entry in select_entries(manifest, selection)
    )


def _build_command_argv(
    entry: MatrixEntry, *, binding: RunBinding, model_id: str
) -> tuple[str, ...]:
    argv = [
        "llmtracefx-optimizer",
        "collect-mlx",
        "--run-id",
        entry.run_id,
        "--model-path",
        str(binding.target_model_path),
        "--model-id",
        model_id,
        "--max-tokens",
        str(entry.max_tokens),
        "--seed",
        str(binding.seed),
    ]
    if binding.draft_model_path is not None:
        argv.extend(
            (
                "--draft-model-path",
                str(binding.draft_model_path),
                "--num-draft-tokens",
                str(binding.num_draft_tokens),
            )
        )
    return tuple(argv)


def execute_row(
    entry: MatrixEntry,
    *,
    manifest_dir: Path,
    output_dir: Path,
    model_id: str,
    binding: RunBinding,
    resume: bool,
    runtime_factory: Callable[[], MLXRuntime],
) -> RowResult:
    """Verify, (maybe) execute, and evaluate one selected matrix row.

    ``runtime_factory`` is only invoked immediately before the collector
    actually needs to run (never for unsupported/rejected rows, resume-
    trusted rows, or rows that fail verification before execution), so
    selecting only unsupported native-MTP rows never constructs (and
    therefore never imports or platform-checks) an MLX runtime.
    """
    started_at = utc_now_iso()
    run_dir = _run_dir(entry, output_dir=output_dir)
    verification_path = run_dir / "verification.json"
    collection_dir = run_dir / "collection"
    final_record_path = run_dir / "final_record.json"

    def _finish(
        status: RowStatus,
        reason: str | None,
        *,
        final_record: ExperimentRecord | None = None,
        verified_hash: str | None = None,
        binding_hash: str | None = None,
        resumed: bool = False,
        wrote_collection: bool = False,
    ) -> RowResult:
        verification = RowVerification(
            schema_version=VERIFICATION_SCHEMA_VERSION,
            run_id=entry.run_id,
            workload_id=entry.workload_id,
            workload_version=entry.workload_version,
            category=entry.category,
            context_tier=entry.context_tier,
            decode_mode=entry.decode_mode,
            status=status,
            reason=reason,
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
        )
        atomic_write_text(verification_path, verification.to_json())
        return RowResult(
            entry=entry, verification=verification, final_record=final_record
        )

    # 1. Native-MTP / unsupported rows: reject explicitly, never execute,
    #    never silently downgrade to generic draft-model speculation.
    if entry.decode_mode == DECODE_MODE_NATIVE_MTP or not entry.runnable:
        reason = entry.unsupported_reason or (
            "native-mtp execution is not implemented by this pipeline; no "
            "capable runtime is wired into `workloads run`, so this row is "
            "rejected rather than silently downgraded to generic "
            "draft-model speculation"
        )
        return _finish(RowStatus.UNSUPPORTED, reason)

    # 2. Verify the fully materialized prompt against the matrix metadata.
    prompt_path = _resolve_path(entry.prompt_path, base_dir=manifest_dir)
    if not prompt_path.exists():
        return _finish(RowStatus.FAILED, f"prompt file missing: {prompt_path}")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    verified_hash = sha256_text(prompt_text)
    if verified_hash != entry.prompt.prompt_hash:
        return _finish(
            RowStatus.FAILED,
            "prompt hash mismatch: matrix metadata records "
            f"{entry.prompt.prompt_hash} but {prompt_path} hashes to "
            f"{verified_hash}; regenerate the matrix or restore the "
            "original prompt file before running",
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

    binding_hash = binding.run_binding_hash(max_tokens=entry.max_tokens)

    # 4. Resume: trust only a prior completed/skipped artifact whose
    #    hashes still match every current input.
    if resume:
        prior = _load_prior_verification(verification_path)
        if (
            prior is not None
            and prior.status in _TRUSTABLE_RESUME_STATUSES
            and prior.verified_prompt_hash == verified_hash
            and prior.run_binding_hash == binding_hash
            and prior.workload_version == workload.version
        ):
            try:
                trusted_record = ExperimentRecord.read_json(final_record_path)
            except (OSError, SchemaValidationError):
                trusted_record = None
            if trusted_record is not None:
                return _finish(
                    RowStatus.SKIPPED,
                    "trusted prior completed artifact (hash-matching); not "
                    "re-executed",
                    final_record=trusted_record,
                    verified_hash=verified_hash,
                    binding_hash=binding_hash,
                    resumed=True,
                    wrote_collection=True,
                )

    # 5. Execute via the existing MLX-LM collector.
    try:
        config = MLXCollectionConfig(
            run_id=entry.run_id,
            model_path=binding.target_model_path,
            model_id=model_id,
            prompt=prompt_text,
            output_dir=collection_dir,
            command_argv=_build_command_argv(entry, binding=binding, model_id=model_id),
            max_tokens=entry.max_tokens,
            seed=binding.seed,
            draft_model_path=binding.draft_model_path,
            num_draft_tokens=binding.num_draft_tokens,
        )
    except MLXCollectorError as exc:
        return _finish(
            RowStatus.FAILED,
            f"invalid collector configuration: {exc}",
            verified_hash=verified_hash,
            binding_hash=binding_hash,
        )

    try:
        runtime = runtime_factory()
    except MLXCollectorError as exc:
        return _finish(
            RowStatus.FAILED,
            f"MLX runtime is unavailable in this environment: {exc}",
            verified_hash=verified_hash,
            binding_hash=binding_hash,
        )

    collection_result = collect_mlx(config, runtime=runtime)
    collected_record = collection_result.record

    if not collected_record.outcome.success:
        # A runtime failure is never overwritten by an evaluator result.
        collected_record.write_json(final_record_path)
        return _finish(
            RowStatus.FAILED,
            (
                collected_record.error.message
                if collected_record.error is not None
                else "collector reported failure without an error detail"
            ),
            final_record=collected_record,
            verified_hash=verified_hash,
            binding_hash=binding_hash,
            wrote_collection=True,
        )

    try:
        evaluated_outcome = evaluate_workload(workload, collection_result.response_text)
    except (WorkloadSchemaError, OSError, RuntimeError) as exc:
        inconclusive_record = dataclasses.replace(
            collected_record,
            outcome=OutcomeInfo(
                success=collected_record.outcome.success,
                quality_score=None,
                quality_metric=None,
                notes=f"evaluation inconclusive: {exc}",
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
    )


def run_selected_rows(
    manifest: MatrixManifest,
    *,
    manifest_dir: Path,
    output_dir: Path,
    selection: RowSelection,
    binding: RunBinding,
    resume: bool,
    runtime_factory: Callable[[], MLXRuntime],
) -> tuple[RowResult, ...]:
    """Verify, execute, and evaluate every selected matrix row in order.

    ``runtime_factory`` is passed through to ``execute_row`` and invoked
    lazily, so a batch consisting only of unsupported/resumed rows never
    constructs an MLX runtime.
    """
    return tuple(
        execute_row(
            entry,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            model_id=manifest.model_id,
            binding=binding,
            resume=resume,
            runtime_factory=runtime_factory,
        )
        for entry in select_entries(manifest, selection)
    )


def iter_result_statuses(results: Iterable[RowResult]) -> Iterable[RowStatus]:
    return (result.verification.status for result in results)
