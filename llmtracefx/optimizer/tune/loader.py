"""Load and re-validate verified evidence from one or more results directories.

Consumes exactly the artifacts PR #6's verification pipeline produces:
``<results_dir>/runs/<run_id>/verification.json`` and the
``final_record.json`` it references. Never trusts a ``verification.json``
summary at face value -- the referenced ``final_record.json`` is always
re-read and structurally validated, and run/workload/hash identity between the two
artifacts is re-checked here. A row whose identity does not check out is
excluded entirely (not merely rejected as a candidate) because nothing
about it can be trusted enough to even place it in a comparable group.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from ..schema import ExperimentRecord, SchemaValidationError
from ..workloads.verify import RowStatus, RowVerification, VerifyError


class TuneInputError(ValueError):
    """Raised for unrecoverable problems with the tuning input artifacts.

    Unlike a single excluded run (a per-row problem that is reported and
    skipped), this is raised when the *set* of inputs itself cannot be
    trusted, e.g. two results directories disagree about what run_id
    ``X`` means.
    """


@dataclass(frozen=True)
class ExcludedRun:
    """A run that could not be used at all (not even as a rejected candidate)."""

    run_id: str
    source_results_dir: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "source_results_dir": self.source_results_dir,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: object) -> ExcludedRun:
        if not isinstance(data, dict):
            raise TuneInputError("excluded run entry must be a JSON object")
        for key in ("run_id", "source_results_dir", "reason"):
            if key not in data:
                raise TuneInputError(
                    f"excluded run entry is missing required field: {key!r}"
                )
            if not isinstance(data[key], str):
                raise TuneInputError(
                    f"excluded run entry.{key} must be a string, got {data[key]!r}"
                )
        return cls(
            run_id=data["run_id"],
            source_results_dir=data["source_results_dir"],
            reason=data["reason"],
        )


@dataclass(frozen=True)
class RunEvidence:
    """One fully loaded, identity-checked (verification, final_record) pair."""

    run_id: str
    source_results_dir: str
    verification: RowVerification
    verification_path: Path
    final_record: ExperimentRecord
    final_record_path: Path


@dataclass(frozen=True)
class LoadedEvidence:
    """Everything loaded from the requested results directories."""

    usable: tuple[RunEvidence, ...]
    excluded: tuple[ExcludedRun, ...]


def _resolve_artifact_path(raw: str | None, *, results_dir: Path, run_id: str) -> Path:
    if raw is None:
        return results_dir / "runs" / run_id / "final_record.json"
    path = Path(raw)
    return path if path.is_absolute() else results_dir / raw


def _record_identity_error(
    verification: RowVerification, record: ExperimentRecord
) -> str | None:
    """Return a reason string if ``record`` does not match ``verification``.

    Checks the identity facts a hand-edited or stale artifact pair could
    disagree on: which run produced it, and whether the prompt that was
    actually hashed at verify-time is the same prompt the collector hashed
    while producing the final record.
    """
    if record.run_id != verification.run_id:
        return (
            f"final_record.run_id ({record.run_id!r}) does not match "
            f"verification.run_id ({verification.run_id!r})"
        )
    if (
        verification.recorded_prompt_hash is not None
        and verification.verified_prompt_hash is not None
        and verification.recorded_prompt_hash != verification.verified_prompt_hash
    ):
        return (
            "verification.json itself records a recorded/verified prompt hash "
            "mismatch (recorded="
            f"{verification.recorded_prompt_hash!r}, verified="
            f"{verification.verified_prompt_hash!r}); this artifact predates a "
            "resolved drift and cannot be trusted"
        )
    if (
        verification.verified_prompt_hash is not None
        and record.command.workload_hash is not None
        and verification.verified_prompt_hash != record.command.workload_hash
    ):
        return (
            "final_record.command.workload_hash "
            f"({record.command.workload_hash!r}) does not match "
            f"verification.verified_prompt_hash ({verification.verified_prompt_hash!r})"
        )
    return None


def _load_one_run(
    verification_path: Path, *, results_dir: Path
) -> tuple[RunEvidence | None, ExcludedRun | None]:
    run_id = verification_path.parent.name
    try:
        verification = RowVerification.read_json(verification_path)
    except (OSError, VerifyError) as exc:
        return None, ExcludedRun(
            run_id=run_id,
            source_results_dir=str(results_dir),
            reason=f"could not read/parse verification.json: {exc}",
        )

    if verification.status == RowStatus.UNSUPPORTED:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=(
                verification.reason
                or "row is unsupported (e.g. native-mtp); no final_record.json "
                "was ever produced for it"
            ),
        )

    if verification.final_record_path is None:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=(
                "verification.json has no final_record_path recorded (no "
                f"canonical evidence was produced); row status was "
                f"'{verification.status.value}'"
            ),
        )

    final_record_path = _resolve_artifact_path(
        verification.final_record_path, results_dir=results_dir, run_id=run_id
    )
    try:
        # Preserve legacy tune diagnostics: the tuner classifies non-finite
        # measurements with field-specific rejection reasons.
        record = ExperimentRecord.read_json(final_record_path, allow_non_finite=True)
    except OSError as exc:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=f"could not read final_record.json ({final_record_path}): {exc}",
        )
    except SchemaValidationError as exc:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=(
                f"final_record.json ({final_record_path}) failed schema "
                f"validation: {exc}"
            ),
        )

    identity_error = _record_identity_error(verification, record)
    if identity_error is not None:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=identity_error,
        )

    return (
        RunEvidence(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            verification=verification,
            verification_path=verification_path,
            final_record=record,
            final_record_path=final_record_path,
        ),
        None,
    )


def load_evidence(
    results_dirs: tuple[Path, ...],
    *,
    primary_run_ids: Set[str] | None = None,
) -> LoadedEvidence:
    """Load and identity-check every run under every given results directory.

    Raises ``TuneInputError`` if the same ``run_id`` appears more than once
    with materially different content (different verification or final
    record payloads) -- this project never silently picks one of two
    conflicting artifacts. An exact duplicate (identical content, e.g. the
    same results directory passed twice or an overlapping directory tree)
    is harmless and is de-duplicated.
    """
    usable: dict[str, RunEvidence] = {}
    excluded: list[ExcludedRun] = []

    for index, results_dir in enumerate(results_dirs):
        runs_dir = results_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for verification_path in sorted(runs_dir.glob("*/verification.json")):
            if (
                index == 0
                and primary_run_ids is not None
                and verification_path.parent.name not in primary_run_ids
            ):
                continue
            evidence, excluded_run = _load_one_run(
                verification_path, results_dir=results_dir
            )
            if evidence is None:
                if excluded_run is not None:
                    excluded.append(excluded_run)
                continue

            prior = usable.get(evidence.run_id)
            if prior is None:
                usable[evidence.run_id] = evidence
                continue

            if (
                prior.verification.to_dict() == evidence.verification.to_dict()
                and prior.final_record.to_dict() == evidence.final_record.to_dict()
            ):
                # Identical duplicate (e.g. an overlapping/reused results
                # directory): harmless, keep the first copy.
                continue

            raise TuneInputError(
                f"duplicate run_id {evidence.run_id!r} found in both "
                f"{prior.source_results_dir!r} and "
                f"{evidence.source_results_dir!r} with conflicting artifact "
                "content; refusing to silently pick one. Remove or reconcile "
                "the stale/duplicate results directory before tuning."
            )

    return LoadedEvidence(
        usable=tuple(usable[run_id] for run_id in sorted(usable)),
        excluded=tuple(excluded),
    )
