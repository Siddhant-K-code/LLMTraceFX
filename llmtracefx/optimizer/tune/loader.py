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


def _contained_regular_file(candidate: Path, *, root: Path) -> Path | None:
    """The resolved ``candidate`` when it is a regular file inside ``root``.

    Resolution happens before the containment test, so a symlink pointing
    out of the tree fails it rather than being followed out. ``is_file`` on
    the resolved path additionally rejects a directory, a FIFO or a device
    node, which would otherwise be opened and read.
    """
    try:
        resolved = candidate.resolve(strict=True)
        anchor = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_relative_to(anchor):
        return None
    if not resolved.is_file():
        return None
    return resolved


def _resolve_artifact_path(
    raw: str | None, *, results_dir: Path, run_id: str
) -> Path | None:
    """Resolve a recorded artifact path inside the results directory named.

    ``verify.execute_row`` writes the record at
    ``<output_dir>/runs/<run_id>/final_record.json`` unconditionally, so that
    location is correct by construction for every recorded form and is tried
    first. The recorded string is corroboration, not authority.

    Anchoring on the results directory rather than on the recorded string
    matters in both directions. A relative ``--output-dir`` records a path
    relative to the *working directory* of that run, so reading the recorded
    string literally would resolve it against whatever tree this process
    happens to be standing in: run ``compare --results B`` from inside tree
    ``A`` and you would silently get A's measurements labelled as B's, with
    every identity check passing because a matrix ``run_id`` names the task
    and not the system. Joining the recorded string under the results
    directory instead doubles the prefix
    (``artifacts/run/artifacts/run/...``) and finds nothing. The canonical
    location avoids both.

    Nothing outside ``results_dir`` is ever returned. The recorded string is
    data from the artifact being validated, so an absolute path, a ``..``
    segment or a symlink in it is under the control of whoever wrote that
    artifact; honouring any of them would let one results tree serve another
    tree's measurements while every identity check passed. When the recorded
    string does not resolve to a regular file inside the tree, ``None`` is
    returned and the caller reports the record as unreadable rather than
    reading it from wherever the string pointed.
    """
    canonical = results_dir / "runs" / run_id / "final_record.json"
    contained = _contained_regular_file(canonical, root=results_dir)
    if contained is not None:
        return contained
    if raw is None or not isinstance(raw, str):
        # Not a string means nothing usable to resolve. ``RowVerification``
        # does not type-check this field, and reaching ``Path()`` with a list
        # raises ``TypeError`` out of the middle of the loader.
        return None
    candidate = Path(raw)
    # An absolute recorded path is believed only when it names something
    # inside the tree that was actually asked for.
    probe = candidate if candidate.is_absolute() else results_dir / raw
    return _contained_regular_file(probe, root=results_dir)


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
        # ``read_json`` is bounded, regular-file-only and does not follow
        # symlinks, so this needs no separate guard here.
        verification = RowVerification.read_json(verification_path)
    except (OSError, VerifyError, TypeError, ArithmeticError) as exc:
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
    if final_record_path is None:
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=(
                f"could not read final_record.json for run "
                f"{verification.run_id!r}: it is not a regular file inside the "
                "results directory. A record reached through an absolute "
                "path, a parent-directory segment or a symlink leaving the "
                "tree is not this tree's evidence"
            ),
        )
    try:
        # Preserve legacy tune diagnostics: the tuner classifies non-finite
        # measurements with field-specific rejection reasons. ``read_json``
        # is bounded, regular-file-only and does not follow symlinks.
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
    except (ArithmeticError, ValueError, TypeError, RecursionError) as exc:
        # A record is untrusted input, and JSON's own limits are not reported
        # as schema failures. A numeric literal too large for a float raises
        # ``OverflowError``, which is an ``ArithmeticError`` and so is caught
        # by nothing above; deep nesting raises ``RecursionError``; a value
        # of the wrong type reaches an arithmetic or a hash as ``TypeError``.
        # None of those are conditions the caller can act on as themselves,
        # so they become the same exclusion every other malformed record
        # produces rather than escaping as a stack trace.
        return None, ExcludedRun(
            run_id=verification.run_id,
            source_results_dir=str(results_dir),
            reason=(
                f"final_record.json ({final_record_path}) could not be parsed: "
                f"{type(exc).__name__}: {exc}"
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
