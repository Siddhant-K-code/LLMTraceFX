"""Local orchestrator for one authorized Modal L4 crossover execution.

This module is the only place that may cause provider spend, and it is
written so that every path to spend passes a gate first and every path
away from it passes teardown.

The Modal SDK is imported lazily, inside ``execute``, after the
environment has been checked for credential and routing overrides, the
authorization receipt has been verified against an operator signature,
the official rates have been refreshed and adjudicated, account headroom
has been established, and the application ledger has been opened. A
plan, a verification, or a dry run therefore never loads the SDK at all.

Calls are strictly sequential and follow the sealed call sequence:
staging, staging verification, the eager canary, the compiled canary,
and then the thirty-two sealed cells only if both canaries passed, and
finally the analysis inventory. Every call reserves its lifecycle in the
ledger first, is dispatched once, and is never replaced or retried. A
second attempt, a crash, a preemption, a timeout, or a missing terminal
receipt stops the run where it stands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text
from .modal_l4_crossover import (
    COMPUTE_PLANNED_SECONDS,
    FUNCTION_SPEC_BY_KEY,
    HARD_CAP_USD,
    LIFECYCLE_BY_ID,
    PROFILE_AUTHENTICATION_FIELDS,
    PROFILE_AUTHENTICATION_GATE,
    PROFILE_AUTHENTICATION_MECHANISM,
    PROFILE_AUTHENTICATION_SCHEMA_VERSION,
    PROTOCOL_ID,
    RUNTIME_IMAGE_SPEC_COMMITMENT,
    STATISTICAL_PUBLICATION,
    TESTED_MODAL_VERSION,
    UNCONTROLLED_CACHE_LIMITATIONS,
    UNSUPPORTED_PROVIDER_CONTROLS,
    ModalApplicationLedger,
    ModalL4ContractError,
    ModalL4Plan,
    build_default_plan,
    call_sequence,
    evaluate_attempt_receipts,
    evaluate_memory_gate,
    evaluate_teardown_receipt,
    require_credential_exposure_cleared,
    require_local_profile_authentication,
    run_scoped_names,
    runtime_image_identity,
    verify_profile_authentication,
    verify_sdk_capabilities,
)
from .modal_l4_rates import (
    HEADROOM_SIGNATURE_NAMESPACE,
    OFFICIAL_SOURCE_URLS,
    RateRefreshError,
    account_headroom,
    read_structured_receipt,
    refresh_official_rates,
)
from .vllm_compile import BASE_IMAGE_REFERENCE, canonical_decimal, canonical_json

AUTHORIZATION_SCHEMA_VERSION = "1"
ORCHESTRATION_SCHEMA_VERSION = "1"
AUTHORIZATION_SIGNER_IDENTITY = "modal-l4-crossover-coordinator"
AUTHORIZATION_SIGNATURE_NAMESPACE = "llmtracefx-modal-l4-authorization-v1"
HEADROOM_SIGNER_IDENTITY = "modal-l4-headroom-coordinator"
SIGNATURE_COMMAND_TIMEOUT_SECONDS = 10
GIT_COMMAND_TIMEOUT_SECONDS = 15
# The longest a signed approval's execution window may last. The sealed compute
# envelope reserves ``COMPUTE_PLANNED_SECONDS`` (4h14m) of provider lifecycles;
# a real run also spends a few minutes in preflight -- the official-rate fetch,
# the signature and git-checkout checks, and the headroom probe -- before the
# first spend, and a short teardown after the last. The window may therefore
# last at most the full compute envelope plus a generous one-hour operational
# allowance (about five and a quarter hours). That is comfortably longer than
# any real run, yet far from indefinite, so a stale or replayed approval can
# never authorize a run once its bounded window has closed.
MAX_EXECUTION_WINDOW = timedelta(seconds=COMPUTE_PLANNED_SECONDS + 3600)
_SAFE_EXECUTION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_NONCE = re.compile(r"^[0-9a-f]{32,64}$")
# The repository root of this checkout. The source-checkout gate reads its git
# state so a run can only start from source committed exactly at the signed
# head.
REPO_ROOT = Path(__file__).resolve().parents[4]
# The only untracked path the source-checkout gate tolerates: an out-of-band
# trace directory that no execution module imports. Everything else untracked,
# and any tracked modification, blocks the run.
IGNORED_UNTRACKED_PREFIX = ".agent-traces/"
IGNORED_UNTRACKED_EXACT = ".agent-traces"

APP_NAME_VAR = "LLMTRACEFX_MODAL_L4_APP_NAME"
VOLUME_NAME_VAR = "LLMTRACEFX_MODAL_L4_VOLUME_NAME"
NONCE_VAR = "LLMTRACEFX_MODAL_L4_NONCE"
PLAN_SHA256_VAR = "LLMTRACEFX_MODAL_L4_PLAN_SHA256"

# Provider failures that must be classified rather than swallowed. Names are
# matched textually so a fake SDK in tests, and a real SDK whose exception
# tree moves, are both handled without importing the provider package here.
TIMEOUT_EXCEPTION_NAMES = ("FunctionTimeoutError", "ExecTimeoutError", "TimeoutError")
PREEMPTION_EXCEPTION_NAMES = ("SandboxTerminatedError", "ResourceExhaustedError")


class ModalExecutionError(ModalL4ContractError):
    """Raised when an authorized execution cannot proceed safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    """Parse a strict UTC ISO-8601 timestamp, refusing anything ambiguous.

    A bounded execution window is only meaningful if its edges are unambiguous
    instants. A naive timestamp, a non-UTC offset, or an unparseable string is
    refused so a window can never be silently widened or reinterpreted in a
    local timezone. Only UTC (``+00:00`` or a trailing ``Z``) is accepted.
    """

    if not isinstance(value, str) or not value:
        raise ModalExecutionError(f"authorization {field} must be a timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ModalExecutionError(
            f"authorization {field} is not an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ModalExecutionError(
            f"authorization {field} must be an explicit UTC timestamp"
        )
    return parsed.astimezone(timezone.utc)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ceil_seconds(elapsed_ns: int) -> int:
    """Return a conservative whole-second ceiling of an elapsed duration.

    Rounding up never under-reports the observed duration; a call that took
    any nonzero time bills at least one second.
    """

    if elapsed_ns <= 0:
        return 0
    return math.ceil(elapsed_ns / 1_000_000_000)


def _sha256_json(value: Any) -> str:
    return _sha256_text(canonical_json(value))


def _sha256_file(path: Path) -> str:
    return _sha256_text(read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES))


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
    )


@dataclass(frozen=True)
class ModalExecutionAuthorization:
    """Explicit, signed authority for one future Modal L4 execution."""

    plan_sha256: str
    source_head: str
    experiment_nonce: str
    workspace_sha256: str
    rate_receipt_sha256: str
    credential_exposure_attestation_sha256: str
    authorized_at: str
    not_before: str
    expires_at: str
    authorization_sha256: str

    @staticmethod
    def content(
        *,
        plan_sha256: str,
        source_head: str,
        experiment_nonce: str,
        workspace_sha256: str,
        rate_receipt_sha256: str,
        credential_exposure_attestation_sha256: str,
        authorized_at: str,
        not_before: str,
        expires_at: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "provider": "Modal",
            "approved": True,
            "plan_sha256": plan_sha256,
            "source_head": source_head,
            "experiment_nonce": experiment_nonce,
            "run_names": run_scoped_names(experiment_nonce),
            "base_image_reference": BASE_IMAGE_REFERENCE,
            "runtime_image_spec_commitment": RUNTIME_IMAGE_SPEC_COMMITMENT,
            "runtime_image_run_commitment": runtime_image_identity(
                source_head=source_head
            )["runtime_image_run_commitment"],
            "workspace_sha256": workspace_sha256,
            "rate_receipt_sha256": rate_receipt_sha256,
            "credential_exposure_attestation_sha256": (
                credential_exposure_attestation_sha256
            ),
            "credential_exposure_gate_cleared": True,
            "authorized_at": authorized_at,
            # A signed, bounded execution window. The approval is valid only
            # between these two UTC instants, so a stale or replayed approval
            # cannot authorize a run indefinitely: preflight refuses before or
            # after the window, before any network fetch or SDK import.
            "not_before": not_before,
            "expires_at": expires_at,
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "automatic_retries": 0,
            "provider_sdk_tested_version": TESTED_MODAL_VERSION,
            "accepts_modal_crash_reschedule_residual": True,
            "authentication": "standard_local_modal_profile_only",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.content(
                plan_sha256=self.plan_sha256,
                source_head=self.source_head,
                experiment_nonce=self.experiment_nonce,
                workspace_sha256=self.workspace_sha256,
                rate_receipt_sha256=self.rate_receipt_sha256,
                credential_exposure_attestation_sha256=(
                    self.credential_exposure_attestation_sha256
                ),
                authorized_at=self.authorized_at,
                not_before=self.not_before,
                expires_at=self.expires_at,
            ),
            "authorization_sha256": self.authorization_sha256,
        }

    @classmethod
    def from_dict(cls, data: Any, *, plan: ModalL4Plan) -> ModalExecutionAuthorization:
        if not isinstance(data, Mapping):
            raise ModalExecutionError("authorization must be an object")
        for field in (
            "plan_sha256",
            "workspace_sha256",
            "rate_receipt_sha256",
            "credential_exposure_attestation_sha256",
        ):
            value = data.get(field)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ModalExecutionError(f"authorization {field} is invalid")
        head = data.get("source_head")
        if not isinstance(head, str) or _GIT_HEAD.fullmatch(head) is None:
            raise ModalExecutionError("authorization source head is invalid")
        nonce = data.get("experiment_nonce")
        if not isinstance(nonce, str) or _NONCE.fullmatch(nonce) is None:
            raise ModalExecutionError("authorization nonce is invalid")
        authorized_at = data.get("authorized_at")
        if not isinstance(authorized_at, str) or not authorized_at:
            raise ModalExecutionError("authorization timestamp is invalid")
        # The bounded window edges are parsed strictly (UTC only) and must form
        # a real, ordered interval no longer than the documented maximum before
        # the content hash is even recomputed. A window that is empty, inverted,
        # or wider than a full compute envelope plus preflight is refused so a
        # single approval can never authorize an effectively unbounded run.
        not_before = _parse_utc_timestamp(data.get("not_before"), field="not_before")
        expires_at = _parse_utc_timestamp(data.get("expires_at"), field="expires_at")
        if not_before >= expires_at:
            raise ModalExecutionError(
                "authorization execution window is empty or inverted"
            )
        if expires_at - not_before > MAX_EXECUTION_WINDOW:
            raise ModalExecutionError(
                "authorization execution window exceeds the maximum bounded "
                f"duration of {int(MAX_EXECUTION_WINDOW.total_seconds())} seconds"
            )
        expected = cls.content(
            plan_sha256=data["plan_sha256"],
            source_head=head,
            experiment_nonce=nonce,
            workspace_sha256=data["workspace_sha256"],
            rate_receipt_sha256=data["rate_receipt_sha256"],
            credential_exposure_attestation_sha256=data[
                "credential_exposure_attestation_sha256"
            ],
            authorized_at=authorized_at,
            not_before=data["not_before"],
            expires_at=data["expires_at"],
        )
        if {
            key: value for key, value in data.items() if key != "authorization_sha256"
        } != expected:
            raise ModalExecutionError(
                "authorization does not match the approved execution envelope"
            )
        if data.get("authorization_sha256") != _sha256_json(expected):
            raise ModalExecutionError("authorization content hash does not verify")
        if data["plan_sha256"] != plan.content_sha256:
            raise ModalExecutionError("authorization is bound to a different plan")
        return cls(
            plan_sha256=data["plan_sha256"],
            source_head=head,
            experiment_nonce=nonce,
            workspace_sha256=data["workspace_sha256"],
            rate_receipt_sha256=data["rate_receipt_sha256"],
            credential_exposure_attestation_sha256=data[
                "credential_exposure_attestation_sha256"
            ],
            authorized_at=authorized_at,
            not_before=data["not_before"],
            expires_at=data["expires_at"],
            authorization_sha256=data["authorization_sha256"],
        )

    def execution_window(self) -> tuple[datetime, datetime]:
        """Return the parsed (not_before, expires_at) UTC window."""

        return (
            _parse_utc_timestamp(self.not_before, field="not_before"),
            _parse_utc_timestamp(self.expires_at, field="expires_at"),
        )

    @classmethod
    def read(cls, path: Path, *, plan: ModalL4Plan) -> ModalExecutionAuthorization:
        try:
            payload = json.loads(
                read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES),
                parse_constant=reject_non_finite_json_constant,
            )
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise ModalExecutionError(
                f"authorization could not be read safely: {exc}"
            ) from exc
        return cls.from_dict(payload, plan=plan)


def _default_signature_runner(command: Sequence[str], input_text: str) -> int:
    completed = subprocess.run(
        list(command),
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
        timeout=SIGNATURE_COMMAND_TIMEOUT_SECONDS,
        env={"PATH": _SAFE_EXECUTION_PATH, "LANG": "C", "LC_ALL": "C"},
    )
    return completed.returncode


def _verify_detached_signature(
    message: str,
    *,
    signature_path: Path,
    authorized_signers_path: Path,
    signer_identity: str,
    namespace: str,
    label: str,
    runner: Callable[[Sequence[str], str], int] | None = None,
) -> dict[str, Any]:
    """Verify an OpenSSH detached signature over ``message``.

    This is the same fail-closed mechanism the CloudRift execution path uses:
    ``ssh-keygen -Y verify`` against an out-of-band authorized-signers file
    under a fixed namespace and signer identity. A receipt this process could
    have written for itself proves nothing, so the signature must be made by a
    key the operator listed separately.
    """

    for path, path_label in (
        (signature_path, f"{label} signature"),
        (authorized_signers_path, f"{label} authorized signers"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ModalExecutionError(
                f"{path_label} must be a non-symlink regular file"
            )
    argv = (
        "ssh-keygen",
        "-Y",
        "verify",
        "-f",
        str(authorized_signers_path.resolve()),
        "-I",
        signer_identity,
        "-n",
        namespace,
        "-s",
        str(signature_path.resolve()),
    )
    run = runner or _default_signature_runner
    if run(argv, message) != 0:
        raise ModalExecutionError(f"{label} signature did not verify")
    return {
        "mechanism": "openssh_detached_signature",
        "namespace": namespace,
        "signer_identity": signer_identity,
        "signature_sha256": _sha256_file(signature_path),
        "authorized_signers_sha256": _sha256_file(authorized_signers_path),
        "verified": True,
    }


def verify_authorization_signature(
    authorization: ModalExecutionAuthorization,
    *,
    signature_path: Path,
    authorized_signers_path: Path,
    runner: Callable[[Sequence[str], str], int] | None = None,
) -> dict[str, Any]:
    """Verify an OpenSSH detached signature over the authorization."""

    return _verify_detached_signature(
        canonical_json(authorization.to_dict()),
        signature_path=signature_path,
        authorized_signers_path=authorized_signers_path,
        signer_identity=AUTHORIZATION_SIGNER_IDENTITY,
        namespace=AUTHORIZATION_SIGNATURE_NAMESPACE,
        label="authorization",
        runner=runner,
    )


def build_headroom_signature_verifier(
    *,
    signature_path: Path,
    authorized_signers_path: Path,
    runner: Callable[[Sequence[str], str], int] | None = None,
) -> Callable[[Mapping[str, Any]], None]:
    """Return a real OpenSSH verifier for a signed operator headroom receipt.

    The account has no pre-run spend authority, so headroom can only come from
    a receipt an operator signed out of band. This wires the production
    verifier the account-headroom gate calls: it canonicalises the exact
    receipt and checks a detached signature made under the fixed headroom
    namespace by a listed key. Tests inject their own verifier instead; the
    production path is real and refuses unsigned or self-authored headroom.
    """

    def verify(receipt: Mapping[str, Any]) -> None:
        _verify_detached_signature(
            canonical_json(dict(receipt)),
            signature_path=signature_path,
            authorized_signers_path=authorized_signers_path,
            signer_identity=HEADROOM_SIGNER_IDENTITY,
            namespace=HEADROOM_SIGNATURE_NAMESPACE,
            label="headroom",
            runner=runner,
        )

    return verify


class SourceCheckoutProbe(Protocol):
    def __call__(self) -> Mapping[str, Any]: ...


def _default_source_checkout_probe(repo_root: Path) -> dict[str, Any]:
    """Read the real git checkout state with a fixed PATH and no shell.

    Only ``git rev-parse HEAD`` and ``git status --porcelain`` are run, each
    with a closed environment, no stdin, and a bounded timeout. The raw
    porcelain is returned for adjudication; nothing is interpreted here so the
    parsing and the refusal policy stay testable with an injected fake.
    """

    def _git(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True,
                text=True,
                check=False,
                shell=False,
                timeout=GIT_COMMAND_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
                env={
                    "PATH": _SAFE_EXECUTION_PATH,
                    "LANG": "C",
                    "LC_ALL": "C",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ModalExecutionError(
                f"source checkout state could not be read: {type(exc).__name__}"
            ) from exc
        if completed.returncode != 0:
            raise ModalExecutionError(
                "source checkout state could not be read from git"
            )
        return completed.stdout

    return {
        "head": _git("rev-parse", "HEAD").strip(),
        "status_porcelain": _git("status", "--porcelain", "--untracked-files=all"),
    }


def _is_ignored_untracked(path: str) -> bool:
    normalized = path.strip().strip('"')
    return normalized == IGNORED_UNTRACKED_EXACT or normalized.startswith(
        IGNORED_UNTRACKED_PREFIX
    )


def verify_source_checkout(
    *,
    source_head: str,
    repo_root: Path = REPO_ROOT,
    probe: SourceCheckoutProbe | None = None,
) -> dict[str, Any]:
    """Bind execution to a clean git checkout at the authorized source head.

    This gate runs before any network fetch, SDK import, or provider call, so a
    run can only start from source committed exactly at the signed head. The
    known-untracked ``.agent-traces/`` directory is tolerated; any other
    untracked path -- especially untracked Python or package source that
    execution imports -- and any tracked modification is a refusal. The probe
    is injectable so offline tests exercise the policy with a fake checkout
    state; production reads real git and refuses until the implementation is
    committed exactly.
    """

    if not isinstance(source_head, str) or _GIT_HEAD.fullmatch(source_head) is None:
        raise ModalExecutionError("authorization source head is not a commit")
    observed = (probe or (lambda: _default_source_checkout_probe(repo_root)))()
    if not isinstance(observed, Mapping):
        raise ModalExecutionError("source checkout probe returned no state")
    head = observed.get("head")
    if not isinstance(head, str) or _GIT_HEAD.fullmatch(head) is None:
        raise ModalExecutionError("source checkout HEAD is not a commit")
    if head != source_head:
        raise ModalExecutionError(
            "source checkout HEAD does not match the authorized source head"
        )
    porcelain = observed.get("status_porcelain")
    if not isinstance(porcelain, str):
        raise ModalExecutionError("source checkout status is unavailable")
    dirty_tracked = 0
    disallowed_untracked = 0
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        if len(line) < 4 or line[2] != " ":
            raise ModalExecutionError("source checkout status line is malformed")
        code, path = line[:2], line[3:]
        if code == "??":
            if not _is_ignored_untracked(path):
                disallowed_untracked += 1
        else:
            # Any staged, modified, deleted, or renamed tracked path. Paths are
            # counted, never echoed, so a private path cannot leak into a
            # refusal message or a persisted receipt.
            dirty_tracked += 1
    if dirty_tracked or disallowed_untracked:
        raise ModalExecutionError(
            "source checkout is not clean at the authorized head; refusing to "
            "run from uncommitted or dirty source"
        )
    return {
        "verified": True,
        "source_head": source_head,
        "tracked_workspace_clean": True,
        "ignored_untracked_prefix": IGNORED_UNTRACKED_PREFIX,
    }


def _classify_provider_failure(exc: BaseException) -> str:
    name = type(exc).__name__
    if name in TIMEOUT_EXCEPTION_NAMES:
        return "timeout"
    if name in PREEMPTION_EXCEPTION_NAMES:
        return "preemption"
    return "crash"


def verify_execution_window(
    authorization: ModalExecutionAuthorization,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Refuse a stale, premature, or replayed approval outside its window.

    The signed approval carries a bounded ``[not_before, expires_at)`` UTC
    window. This is checked before any network fetch or SDK import, so an
    approval that is not yet valid, or has expired, can never authorize a run --
    an indefinitely valid approval cannot be replayed later. ``now`` must be a
    timezone-aware instant; a naive clock is refused rather than guessed.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ModalExecutionError("current time must be timezone-aware")
    now_utc = now.astimezone(timezone.utc)
    not_before, expires_at = authorization.execution_window()
    if now_utc < not_before:
        raise ModalExecutionError(
            "authorization is not yet valid; current time precedes its "
            "execution window"
        )
    if now_utc >= expires_at:
        raise ModalExecutionError(
            "authorization has expired; refusing a stale or replayed approval "
            "outside its execution window"
        )
    return {
        "verified": True,
        "not_before": authorization.not_before,
        "expires_at": authorization.expires_at,
        "checked_within_window": True,
    }


PROFILE_VALIDATION_TIMEOUT_SECONDS = 20
# The probe is the running interpreter's own ``modal`` module, not a bare
# ``modal`` looked up on PATH. ``sys.executable -m modal`` is guaranteed to be
# the exact package the loaded SDK was imported from -- same interpreter, same
# site-packages, same version -- so the probed CLI cannot silently be a
# different SDK than the one this run adjudicated, and the pinned uv venv is
# used rather than whatever ``modal`` a fixed PATH happens to resolve. The
# recorded verdict schema (version, mechanism, fields) is the shared closed
# schema in modal_l4_crossover, validated identically on both sides.
PROFILE_PROBE_MODULE_ARGS = ("-m", "modal", "token", "info")
# Environment variables that would redirect which profile, config file, or
# account the probe reads. None of these may be forwarded into the probe, or a
# routing override could make an unauthenticated run look authenticated.
_PROFILE_CONFIG_OVERRIDE_ENV = (
    "MODAL_PROFILE",
    "MODAL_CONFIG_PATH",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_TOKEN",
    "MODAL_SERVER_URL",
    "MODAL_ENVIRONMENT",
    "MODAL_WORKSPACE",
)
_PROFILE_RESULT_KEYS = frozenset(PROFILE_AUTHENTICATION_FIELDS)


class LocalProfileValidator(Protocol):
    def __call__(self, *, sdk_version: str) -> Mapping[str, Any]: ...


def profile_probe_command() -> tuple[str, ...]:
    """Return the exact read-only profile probe for the running interpreter.

    ``sys.executable -m modal token info`` invokes the same installed ``modal``
    package the SDK is loaded from, so the probed CLI version is the loaded SDK
    version by construction rather than by hope.
    """

    return (sys.executable, *PROFILE_PROBE_MODULE_ARGS)


def _default_profile_command_runner(command: Sequence[str]) -> int:
    """Run a read-only Modal CLI probe, discarding every stream.

    stdin is closed and stdout/stderr are sent to the void, so a profile,
    workspace, account, or token identifier the CLI would print can never enter
    this process. Only the exit status is observed. No shell is used, and the
    environment is built from scratch: it carries just enough for the standard
    profile to be discovered -- ``HOME`` (and ``USERPROFILE`` on Windows) so
    ``~/.modal.toml`` is found -- while every profile, config-path, token, or
    routing override is deliberately dropped so the probe cannot be redirected
    to a different profile than the standard local one. The real ``HOME`` value
    is passed to the child but never returned or recorded.
    """

    env = {
        "PATH": _SAFE_EXECUTION_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "MODAL_TERMINAL_PROMPT": "0",
        "NO_COLOR": "1",
    }
    for name in ("HOME", "USERPROFILE"):
        value = os.environ.get(name)
        if value:
            env[name] = value
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        shell=False,
        timeout=PROFILE_VALIDATION_TIMEOUT_SECONDS,
        env=env,
    )
    return completed.returncode


def validate_local_profile(
    *,
    sdk_version: str,
    runner: Callable[[Sequence[str]], int] | None = None,
    clock: Callable[[], str] = _now,
) -> dict[str, Any]:
    """Confirm a standard authenticated local Modal profile before any spend.

    Runs ``sys.executable -m modal token info`` -- the strongest read-only
    profile probe in the pinned CLI, executed through the *running
    interpreter's own* ``modal`` module rather than a bare ``modal`` resolved on
    PATH -- purely for its exit status. Because it is the same package the SDK
    is loaded from, the probed CLI version is verified to equal the loaded SDK
    version by construction (same interpreter, same install), which is recorded
    as ``cli_version == sdk_version``. Success (status 0) records only booleans,
    that equal version, the same-module mechanism, and a timestamp; the
    profile, account, workspace, and token identifiers the command would print
    are discarded by the runner and never retained, and neither ``HOME`` nor any
    path is recorded. A non-zero status, or any inability to run the probe,
    refuses the run before the app or image is imported.
    """

    if not isinstance(sdk_version, str) or not sdk_version:
        raise ModalExecutionError("profile validation requires the probed SDK version")
    run = runner or _default_profile_command_runner
    try:
        code = run(profile_probe_command())
    except (OSError, subprocess.SubprocessError) as exc:
        raise ModalExecutionError(
            f"local Modal profile could not be validated: {type(exc).__name__}"
        ) from exc
    if isinstance(code, bool) or not isinstance(code, int):
        raise ModalExecutionError("profile validation returned no exit status")
    if code != 0:
        raise ModalExecutionError(
            "no authenticated standard local Modal profile; refusing before any "
            "provider import or spend"
        )
    return {
        "schema_version": PROFILE_AUTHENTICATION_SCHEMA_VERSION,
        "gate": PROFILE_AUTHENTICATION_GATE,
        "authenticated": True,
        "mechanism": PROFILE_AUTHENTICATION_MECHANISM,
        # Equal by construction: the probe is the running interpreter's own
        # modal module, which is the package the SDK version was read from.
        "cli_version": sdk_version,
        "sdk_version": sdk_version,
        "records_profile_identity": False,
        "checked_at": clock(),
    }


def _sanitize_profile_authentication(result: Any) -> dict[str, Any]:
    """Refuse a profile-validation result that is unsafe or over-scoped.

    Delegates to the shared closed-schema validator so the execution path that
    produces the verdict and the result path that later consumes it enforce the
    exact same schema: the pinned version, the same-interpreter module
    mechanism, a probed CLI version that equals the loaded SDK version, no
    retained profile identity, and a timestamp. Failures are surfaced as an
    execution error so the run refuses before the app or image is imported.
    """

    try:
        return verify_profile_authentication(result)
    except ModalL4ContractError as exc:
        raise ModalExecutionError(str(exc)) from exc


@dataclass
class _CallOutcome:
    step: Mapping[str, Any]
    receipt: dict[str, Any] | None
    failure: str | None
    seconds: int
    elapsed_ns: int


class ModalOrchestrator:
    """Run the sealed call sequence once, then always tear down."""

    def __init__(
        self,
        *,
        plan: ModalL4Plan,
        authorization: ModalExecutionAuthorization,
        workspace: Path,
        ledger: ModalApplicationLedger,
        credential_exposure: Mapping[str, Any],
        rate_receipt: Mapping[str, Any],
        sdk_loader: Callable[[], Any],
        app_loader: Callable[[], Any],
        rate_refresh: Mapping[str, Any] | None = None,
        source_checkout: Mapping[str, Any] | None = None,
        profile_validator: LocalProfileValidator | None = None,
        clock: Callable[[], str] = _now,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.plan = plan
        self.authorization = authorization
        self.workspace = workspace
        self.ledger = ledger
        self.credential_exposure = dict(credential_exposure)
        self.rate_receipt = dict(rate_receipt)
        # The fresh capture+verification of the official rate documents, and
        # the source-checkout receipt, are embedded in the orchestration so a
        # later result bundle can project and cross-bind them without a fresh
        # fetch or a second git probe.
        self.rate_refresh = dict(rate_refresh) if rate_refresh is not None else None
        self.source_checkout = (
            dict(source_checkout) if source_checkout is not None else None
        )
        self._sdk_loader = sdk_loader
        self._app_loader = app_loader
        self._profile_validator = profile_validator
        self._clock = clock
        self._monotonic_ns = monotonic_ns
        self.names = run_scoped_names(authorization.experiment_nonce)
        self.attempt_receipts: list[dict[str, Any]] = []
        self.cell_receipts: dict[str, dict[str, Any]] = {}
        self.memory_gate: list[dict[str, Any]] = []
        self.container_identities: dict[str, str] = {}
        self.profile_authentication: dict[str, Any] | None = None
        self.outstanding_call: Any = None
        self.status = "not_started"
        self.failure: str | None = None
        self.teardown: dict[str, Any] | None = None

    # -- provider plumbing ------------------------------------------------

    def _dispatch(self, function: Any, step: Mapping[str, Any]) -> _CallOutcome:
        spec = FUNCTION_SPEC_BY_KEY[str(step["function_key"])]
        lifecycle = LIFECYCLE_BY_ID[str(step["lifecycle_id"])]
        arguments: tuple[Any, ...] = ()
        if step["cell_id"] is not None:
            arguments = (str(step["cell_id"]), self.authorization.experiment_nonce)
        elif step["function_key"] in ("eager_canary", "compiled_canary"):
            arguments = (self.authorization.experiment_nonce,)
        elif step["function_key"] == "analysis":
            arguments = (sorted(self.cell_receipts),)
        started_ns = self._monotonic_ns()
        try:
            call = function.spawn(*arguments)
            self.outstanding_call = call
            receipt = call.get(timeout=spec.timeout_seconds)
        except BaseException as exc:  # noqa: BLE001 - classified, then torn down
            # The handle is deliberately retained: a call that raised may
            # still have a live or reschedulable container, so teardown must
            # be free to cancel it with terminate_containers=True.
            elapsed_ns = max(0, self._monotonic_ns() - started_ns)
            return _CallOutcome(
                step=step,
                receipt=None,
                failure=_classify_provider_failure(exc),
                seconds=_ceil_seconds(elapsed_ns),
                elapsed_ns=elapsed_ns,
            )
        elapsed_ns = max(0, self._monotonic_ns() - started_ns)
        observed_seconds = _ceil_seconds(elapsed_ns)
        if not isinstance(receipt, dict) or receipt.get("terminal") is not True:
            # A non-terminal return is not a proven-idle call; keep the handle
            # so teardown can cancel any container it may have left running.
            return _CallOutcome(
                step=step,
                receipt=receipt if isinstance(receipt, dict) else None,
                failure="missing_terminal_receipt",
                seconds=observed_seconds,
                elapsed_ns=elapsed_ns,
            )
        # Terminal success: this call is settled, so its handle can be released.
        self.outstanding_call = None
        failure = None
        if observed_seconds > lifecycle.planned_seconds:
            # The client observed a duration past the preregistered lifecycle
            # ceiling. Never cap it to look compliant; invalidate the run.
            failure = "lifecycle_ceiling_exceeded"
        return _CallOutcome(
            step=step,
            receipt=receipt,
            failure=failure,
            seconds=observed_seconds,
            elapsed_ns=elapsed_ns,
        )

    def _record_attempt(self, outcome: _CallOutcome) -> dict[str, Any]:
        receipt = outcome.receipt or {}
        identity = receipt.get("container_identity_sha256")
        lifecycle_id = str(outcome.step["lifecycle_id"])
        reused = False
        if isinstance(identity, str) and identity:
            reused = identity in self.container_identities.values()
            self.container_identities[lifecycle_id] = identity
        attempt = {
            "lifecycle_id": lifecycle_id,
            "attempt": 2 if reused else 1,
            "crashed": outcome.failure == "crash",
            "preempted": outcome.failure == "preemption",
            "timed_out": outcome.failure == "timeout",
            "terminal_receipt": (
                outcome.failure is None and receipt.get("status") == "completed"
            ),
        }
        self.attempt_receipts.append(attempt)
        return attempt

    def _reserve(self, step: Mapping[str, Any]) -> str:
        lifecycle_id = str(step["lifecycle_id"])
        if lifecycle_id not in LIFECYCLE_BY_ID:
            raise ModalExecutionError("call sequence names an unplanned lifecycle")
        call_id = f"call-{int(step['step']):03d}"
        self.ledger.reserve(
            call_id, lifecycle_id=lifecycle_id, reserved_at=self._clock()
        )
        return call_id

    def _settle(self, call_id: str, outcome: _CallOutcome) -> None:
        lifecycle = LIFECYCLE_BY_ID[str(outcome.step["lifecycle_id"])]
        if outcome.failure is None:
            # The observed duration is a client-side upper bound on billed
            # compute; it is recorded as-is (never capped to the ceiling) and
            # is guaranteed not to exceed planned_seconds because a breach is
            # classified as a failure and aborted below instead.
            self.ledger.complete(
                call_id,
                completed_at=self._clock(),
                actual_seconds=outcome.seconds,
                duration_provenance="client_observed_monotonic_ceiling_seconds",
            )
            return
        detail = f"call {outcome.failure}"
        if outcome.failure == "lifecycle_ceiling_exceeded":
            detail = (
                f"call exceeded its {lifecycle.planned_seconds}s lifecycle "
                f"ceiling with {outcome.seconds}s observed"
            )
        self.ledger.abort(call_id, aborted_at=self._clock(), reason=detail)

    # -- run --------------------------------------------------------------

    def execute(self) -> dict[str, Any]:
        if self.credential_exposure.get("cleared") is not True:
            raise ModalExecutionError(
                "provider execution is blocked by the credential-exposure gate"
            )
        modal_module = self._sdk_loader()
        capabilities = verify_sdk_capabilities(modal_module)
        # Standard local profile validation: after every offline gate and the
        # SDK capability probe, but before the app or image is imported and
        # before any spend. It is a read-only ``modal token info`` probe whose
        # output is discarded; only a boolean, the version, and a timestamp are
        # kept. A missing or unauthenticated profile refuses here.
        validator = self._profile_validator or (
            lambda *, sdk_version: validate_local_profile(sdk_version=sdk_version)
        )
        self.profile_authentication = _sanitize_profile_authentication(
            validator(sdk_version=str(capabilities["version"]))
        )
        os.environ[NONCE_VAR] = self.authorization.experiment_nonce
        os.environ[APP_NAME_VAR] = self.names["app_name"]
        os.environ[VOLUME_NAME_VAR] = self.names["volume_name"]
        os.environ[PLAN_SHA256_VAR] = self.plan.content_sha256
        app_module = self._app_loader()
        if getattr(app_module.app, "registered_web_endpoints", ()):
            raise ModalExecutionError("the provider app must expose no web endpoint")
        functions = app_module.FUNCTIONS
        self.status = "running"
        try:
            with app_module.app.run():
                app_module.app.set_tags(app_module.APP_TAGS)
                self._run_sequence(functions)
        except ModalExecutionError:
            raise
        except BaseException as exc:  # noqa: BLE001 - teardown owns every exit
            self.status = "failed"
            self.failure = f"{type(exc).__name__}"
            raise
        finally:
            self.teardown = self._tear_down(modal_module, app_module)
        return self._assemble(capabilities)

    def _run_sequence(self, functions: Mapping[str, Any]) -> None:
        canaries_passed = 0
        for step in call_sequence():
            key = str(step["function_key"])
            if step["cell_id"] is not None and canaries_passed != 2:
                self.status = "refused"
                self.failure = "memory gate did not pass; no cell was dispatched"
                return
            call_id = self._reserve(step)
            outcome = self._dispatch(functions[key], step)
            self._settle(call_id, outcome)
            attempt = self._record_attempt(outcome)
            if outcome.failure is not None or not attempt["terminal_receipt"]:
                self.status = "invalidated"
                self.failure = (
                    f"{key} returned {outcome.failure or 'a non-terminal receipt'}"
                )
                return
            receipt = outcome.receipt or {}
            if key in ("eager_canary", "compiled_canary"):
                verdict = evaluate_memory_gate(receipt.get("observation"))
                self.memory_gate.append({**verdict, "receipt": receipt})
                if not verdict["passed"]:
                    self.status = "refused"
                    self.failure = (
                        f"memory gate failed for the {verdict['mode']} canary"
                    )
                    return
                canaries_passed += 1
            if step["cell_id"] is not None:
                self.cell_receipts[str(step["cell_id"])] = receipt
        self.status = "complete"

    # -- teardown ---------------------------------------------------------

    def _tear_down(self, modal_module: Any, app_module: Any) -> dict[str, Any]:
        failures: list[str] = []
        cancelled = True
        if self.outstanding_call is not None:
            try:
                self.outstanding_call.cancel(terminate_containers=True)
            except BaseException:  # noqa: BLE001 - teardown records, never raises
                cancelled = False
                failures.append("outstanding_call_cancel_failed")
            finally:
                self.outstanding_call = None
        scale_zero = True
        for key, function in getattr(app_module, "FUNCTIONS", {}).items():
            try:
                stats = function.get_current_stats()
            except BaseException:  # noqa: BLE001 - absence is recorded, not inferred
                scale_zero = False
                failures.append(f"scale_zero_unverified:{key}")
                continue
            if (
                getattr(stats, "num_total_runners", 1) != 0
                or getattr(stats, "backlog", 1) != 0
            ):
                scale_zero = False
                failures.append(f"runners_remaining:{key}")
        volume_deleted = True
        try:
            modal_module.Volume.objects.delete(
                self.names["volume_name"], allow_missing=True
            )
        except BaseException:  # noqa: BLE001 - teardown records, never raises
            volume_deleted = False
            failures.append("volume_delete_failed")
        listing_available = True
        live: list[str] = []
        try:
            for volume in modal_module.Volume.objects.list():
                name = getattr(volume, "name", "")
                if name == self.names["volume_name"]:
                    live.append(name)
        except BaseException:  # noqa: BLE001 - ambiguity fails closed below
            listing_available = False
            failures.append("named_resource_listing_unavailable")
        receipt = {
            "outstanding_calls_cancelled": cancelled,
            # Exiting the ephemeral app.run() context is a local SDK action,
            # not provider-verified app deletion; it is labelled as such and
            # the provider-side deletion stays explicitly unverifiable.
            "app_context_exited": True,
            "app_stop_mechanism": "ephemeral_app_run_context_exit",
            "app_deletion_provider_verified": None,
            "app_deletion_null_reason": UNSUPPORTED_PROVIDER_CONTROLS[
                "explicit_app_stop_method"
            ],
            # Functions are observable only through autoscaler scale-to-zero,
            # never as a stop or delete.
            "functions_scaled_to_zero": scale_zero,
            "function_inventory_observability": "control_plane_scale_to_zero_only",
            "scale_zero_verified_via_control_plane": scale_zero,
            # No per-container inventory or delete exists.
            "container_inventory_observable": False,
            "container_inventory_null_reason": UNSUPPORTED_PROVIDER_CONTROLS[
                "individual_container_deletion"
            ],
            "individual_container_deletion": None,
            "individual_container_deletion_null_reason": UNSUPPORTED_PROVIDER_CONTROLS[
                "individual_container_deletion"
            ],
            # The run-scoped volume is the only named resource this run can
            # enumerate; the empty-listing claim is scoped to volumes so it is
            # never read as covering apps, functions, or containers.
            "volume_deleted": volume_deleted,
            "named_resource_listing_scope": "volumes_only",
            "named_volume_listing_available": listing_available,
            "live_named_volumes": live,
            "run_created_noncredential_secrets_deleted": True,
            "run_created_secret_count": 0,
            "credential_secret_created": False,
            "sanitized_receipts_retained": True,
            "provider_reported_spend_usd": None,
            "teardown_failures": failures,
            "observed_at": self._clock(),
        }
        return {**receipt, "adjudication": evaluate_teardown_receipt(receipt)}

    # -- output -----------------------------------------------------------

    def _assemble(self, capabilities: Mapping[str, Any]) -> dict[str, Any]:
        attempts = (
            evaluate_attempt_receipts(self.attempt_receipts)
            if (self.status == "complete")
            else {
                "valid": False,
                "findings": [],
                "teardown_required": True,
                "action": "invalidate_and_tear_down",
            }
        )
        # A completed, valid run still cannot publish a result unless teardown
        # is adjudicated complete: an incomplete teardown (a live volume, an
        # unverified scale-to-zero, an ambiguous listing) leaves an
        # unaccounted-for resource, so the run is a refusal.
        teardown_complete = bool(
            isinstance(self.teardown, Mapping)
            and isinstance(self.teardown.get("adjudication"), Mapping)
            and self.teardown["adjudication"].get("complete") is True
        )
        published = (
            self.status == "complete" and attempts["valid"] and teardown_complete
        )
        document = {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "kind": (
                "llmtracefx.modal_l4_crossover.result"
                if published
                else "llmtracefx.modal_l4_crossover.refusal"
            ),
            "published": published,
            "status": self.status,
            "failure": self.failure,
            "plan_sha256": self.plan.content_sha256,
            "source_head": self.authorization.source_head,
            "experiment_nonce": self.authorization.experiment_nonce,
            "authorization_sha256": self.authorization.authorization_sha256,
            "run_names": self.names,
            "base_image_reference": BASE_IMAGE_REFERENCE,
            "runtime_image": runtime_image_identity(
                source_head=self.authorization.source_head
            ),
            "provider_sdk": dict(capabilities),
            "profile_authentication": self.profile_authentication,
            # The reason prose stays out of the orchestration; the standalone
            # credential-exposure envelope is bound to this verdict on its
            # non-reason projection.
            "credential_exposure": {
                key: value
                for key, value in self.credential_exposure.items()
                if key != "reason"
            },
            "rate_receipt": dict(self.rate_receipt),
            # The fresh capture and its verification of the official rate
            # documents are embedded alongside the structured receipt so a
            # result bundle can prove list-rate provenance -- the exact hashed
            # documents and the receipt-to-capture binding -- without a fetch.
            # This is not a claim that HTML was parsed; rates remain a manual
            # structured receipt whose provenance is these hashed captures.
            "rate_refresh": (
                dict(self.rate_refresh) if self.rate_refresh is not None else None
            ),
            # The source-checkout receipt binds the run to a clean git checkout
            # at the authorized head. It is booleans plus the head and the one
            # tolerated untracked prefix, so it is safe to embed and project.
            "source_checkout": (
                dict(self.source_checkout) if self.source_checkout is not None else None
            ),
            "call_sequence_executed": [
                {
                    "lifecycle_id": item["lifecycle_id"],
                    "attempt": item["attempt"],
                    "terminal_receipt": item["terminal_receipt"],
                }
                for item in self.attempt_receipts
            ],
            "attempt_receipts": [dict(item) for item in self.attempt_receipts],
            "attempt_adjudication": attempts,
            # The full sealed canary receipts are embedded so a later adjudicator
            # verifies the seal, mode, observation, and no-tuning verdict from the
            # orchestration alone; the standalone memory-gate envelope is bound to
            # exactly these observations.
            "memory_gate": {
                "tuning_applied": False,
                "canaries": [dict(item) for item in self.memory_gate],
            },
            "completed_cell_ids": sorted(self.cell_receipts),
            "ledger": self.ledger.snapshot(),
            "teardown": self.teardown,
            "statistical_publication": dict(STATISTICAL_PUBLICATION),
            "uncontrolled_limitations": list(UNCONTROLLED_CACHE_LIMITATIONS),
            "provider_reported_spend_usd": None,
            "provider_reported_spend_null_reason": (
                "provider spend is external, sanitized, and never inferred"
            ),
            "observed_at": self._clock(),
        }
        document["orchestration_sha256"] = _sha256_json(document)
        # Result evidence is written only for a published result. A non-complete
        # status, or a complete run refused because teardown was incomplete,
        # never writes an orchestration receipt, a cell receipt, or a canary
        # receipt, so a refusal can never be mistaken for, or replayed as, a
        # result.
        if published:
            _write_json(self.workspace / "orchestration-receipt.json", document)
            for cell_id, receipt in self.cell_receipts.items():
                _write_json(self.workspace / "cells" / f"{cell_id}.json", receipt)
            for index, item in enumerate(self.memory_gate, start=1):
                _write_json(
                    self.workspace / "memory-gate" / f"canary-{index:02d}.json",
                    item["receipt"],
                )
        return document


def _require_workspace(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ModalExecutionError("workspace must be a non-symlink directory")
    return path.resolve()


# Everything this run creates inside the workspace. None of these may already
# exist when a run starts: a stale receipt or ledger from an earlier attempt
# must never be able to mix with, or be replayed into, a new run's evidence.
RUN_OUTPUT_FILES = (
    "credential-exposure.json",
    "source-checkout.json",
    "rate-refresh.json",
    "headroom.json",
    "authorization-authentication.json",
    "execution-window.json",
    "application-ledger.json",
    ".application-ledger.json.lock",
    "orchestration-receipt.json",
)
RUN_OUTPUT_DIRS = ("cells", "memory-gate")


def _require_clean_output_workspace(
    root: Path, *, allowlist: Sequence[str] = ()
) -> None:
    """Refuse to write into a workspace that already holds run artifacts.

    A clean or new workspace is required before any ledger or receipt is
    created. Callers that legitimately pre-stage inputs in the workspace may
    pass an exact ``allowlist`` of entry names; anything matching a
    run-produced output name is always refused so a stale receipt cannot be
    mixed into, or replayed as, this run's evidence.
    """

    owned = set(RUN_OUTPUT_FILES) | set(RUN_OUTPUT_DIRS)
    permitted = set(allowlist)
    conflicts = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.name in owned or entry.name not in permitted
    )
    if conflicts:
        raise ModalExecutionError(
            "execution workspace is not clean; refusing to reuse it so stale "
            "receipts cannot mix or replay: " + ", ".join(conflicts)
        )


def preflight(
    *,
    authorization_path: Path,
    signature_path: Path,
    authorized_signers_path: Path,
    rate_receipt_path: Path,
    workspace: Path,
    credential_exposure_attestation_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    fetcher: Any = None,
    headroom_probe: Any = None,
    signed_headroom: Mapping[str, Any] | None = None,
    headroom_signature_path: Path | None = None,
    headroom_authorized_signers_path: Path | None = None,
    signature_verifier: Any = None,
    signature_runner: Callable[[Sequence[str], str], int] | None = None,
    source_checkout_probe: SourceCheckoutProbe | None = None,
    repo_root: Path = REPO_ROOT,
    urls: Sequence[str] = OFFICIAL_SOURCE_URLS,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Run every gate that must pass before the provider SDK is imported.

    The credential-exposure gate runs first. A standard-profile credential was
    exposed outside this system, so until a coordinator confirms it was revoked
    and that a fresh local profile was created and never shared, no path here
    authenticates, imports the provider SDK, or spends anything. The
    attestation carries booleans and prose only; no token, hash, prefix, or
    screenshot metadata is read or stored.

    After the signed authorization is parsed and verified, its bounded UTC
    execution window is checked against the current time, and before any
    network fetch or provider import the source-checkout gate binds the run to
    a clean git checkout at the authorized source head, so a run can never
    start from uncommitted or dirty execution source, nor from a stale or
    replayed approval outside its window.
    """

    attestation = (
        read_structured_receipt(credential_exposure_attestation_path)
        if credential_exposure_attestation_path is not None
        else None
    )
    exposure = require_credential_exposure_cleared(attestation)
    require_local_profile_authentication(os.environ if environ is None else environ)
    plan = build_default_plan()
    root = _require_workspace(workspace)
    authorization = ModalExecutionAuthorization.read(authorization_path, plan=plan)
    if authorization.workspace_sha256 != _sha256_text(str(root)):
        raise ModalExecutionError("authorization is bound to a different workspace")
    authentication = verify_authorization_signature(
        authorization,
        signature_path=signature_path,
        authorized_signers_path=authorized_signers_path,
        runner=signature_runner,
    )
    # Bounded execution window: after the signature is verified, but before any
    # network fetch or SDK import, refuse an approval that is not yet valid or
    # has expired so a stale or replayed approval cannot start a run.
    execution_window = verify_execution_window(
        authorization,
        now=now_utc if now_utc is not None else datetime.now(timezone.utc),
    )
    if (
        _sha256_json(attestation)
        != authorization.credential_exposure_attestation_sha256
    ):
        raise ModalExecutionError(
            "the supplied credential-exposure attestation is not the authorized one"
        )
    structured = read_structured_receipt(rate_receipt_path)
    if _sha256_json(structured) != authorization.rate_receipt_sha256:
        raise ModalExecutionError(
            "the supplied rate receipt is not the authorized rate receipt"
        )
    # Source-checkout gate: after credential clearance, environment rejection,
    # signed-authorization verification, and the execution-window check, but
    # before the rate fetch, the SDK import, and any provider call.
    source_checkout = verify_source_checkout(
        source_head=authorization.source_head,
        repo_root=repo_root,
        probe=source_checkout_probe,
    )
    rates = refresh_official_rates(
        structured_receipt=structured,
        observed_at=_now(),
        fetcher=fetcher,
        urls=urls,
    )
    # A signed operator headroom receipt is only trustworthy with a real
    # verifier. Tests inject one directly; the production path builds the
    # OpenSSH detached-signature verifier from the two operator-supplied
    # paths, which must be present together with the receipt itself.
    verifier = signature_verifier
    if verifier is None and signed_headroom is not None and headroom_probe is None:
        if headroom_signature_path is None or headroom_authorized_signers_path is None:
            raise ModalExecutionError(
                "a signed headroom receipt requires both --headroom-signature "
                "and --headroom-authorized-signers"
            )
        verifier = build_headroom_signature_verifier(
            signature_path=headroom_signature_path,
            authorized_signers_path=headroom_authorized_signers_path,
            runner=signature_runner,
        )
    headroom = account_headroom(
        control_plane_probe=headroom_probe,
        signed_receipt=signed_headroom,
        signature_verifier=verifier,
    )
    return {
        "authorization": authorization,
        "authorization_authentication": authentication,
        "execution_window": execution_window,
        "credential_exposure": exposure,
        "source_checkout": source_checkout,
        "rate_receipt": structured,
        "rates": rates,
        "headroom": headroom,
        "plan": plan,
        "workspace": root,
    }


def execute(
    *,
    authorization_path: Path,
    signature_path: Path,
    authorized_signers_path: Path,
    rate_receipt_path: Path,
    workspace: Path,
    credential_exposure_attestation_path: Path | None = None,
    sdk_loader: Callable[[], Any] | None = None,
    app_loader: Callable[[], Any] | None = None,
    profile_validator: LocalProfileValidator | None = None,
    workspace_allowlist: Sequence[str] = (),
    **gate_arguments: Any,
) -> dict[str, Any]:
    """Gate, then run, then always tear down. The SDK loads only here."""

    gates = preflight(
        authorization_path=authorization_path,
        signature_path=signature_path,
        authorized_signers_path=authorized_signers_path,
        rate_receipt_path=rate_receipt_path,
        workspace=workspace,
        credential_exposure_attestation_path=credential_exposure_attestation_path,
        **gate_arguments,
    )
    plan: ModalL4Plan = gates["plan"]
    authorization: ModalExecutionAuthorization = gates["authorization"]
    root: Path = gates["workspace"]
    # A clean or explicitly allowlisted workspace is required before any
    # output or ledger is created, so nothing from a prior attempt can mix
    # with or replay into this run's receipts.
    _require_clean_output_workspace(root, allowlist=workspace_allowlist)
    _write_json(root / "credential-exposure.json", gates["credential_exposure"])
    _write_json(root / "source-checkout.json", gates["source_checkout"])
    _write_json(root / "rate-refresh.json", gates["rates"])
    _write_json(root / "headroom.json", gates["headroom"])
    _write_json(
        root / "authorization-authentication.json",
        gates["authorization_authentication"],
    )
    _write_json(root / "execution-window.json", gates["execution_window"])
    ledger = ModalApplicationLedger.initialize(
        root / "application-ledger.json",
        plan=plan,
        git_head=authorization.source_head,
        experiment_nonce=authorization.experiment_nonce,
    )

    def _load_sdk() -> Any:
        import importlib

        return importlib.import_module("modal")

    def _load_app() -> Any:
        import importlib

        return importlib.import_module("llmtracefx.optimizer.lab.qwen3_8b.modal_l4_app")

    orchestrator = ModalOrchestrator(
        plan=plan,
        authorization=authorization,
        workspace=root,
        ledger=ledger,
        credential_exposure=gates["credential_exposure"],
        rate_receipt=gates["rate_receipt"],
        rate_refresh=gates["rates"],
        source_checkout=gates["source_checkout"],
        sdk_loader=sdk_loader or _load_sdk,
        app_loader=app_loader or _load_app,
        profile_validator=profile_validator,
    )
    return orchestrator.execute()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-modal-l4-execute",
        description=(
            "Gate and run one authorized Modal L4 crossover execution. The "
            "provider SDK is imported only by the run action, and only after "
            "every gate has passed."
        ),
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "run"):
        sub = subparsers.add_parser(action, allow_abbrev=False)
        sub.add_argument("--authorization", required=True, type=Path)
        sub.add_argument("--authorization-signature", required=True, type=Path)
        sub.add_argument("--authorized-signers", required=True, type=Path)
        sub.add_argument("--rate-receipt", required=True, type=Path)
        sub.add_argument("--workspace", required=True, type=Path)
        sub.add_argument("--signed-headroom", type=Path)
        sub.add_argument("--headroom-signature", type=Path)
        sub.add_argument("--headroom-authorized-signers", type=Path)
        sub.add_argument("--credential-exposure-attestation", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    headroom_args = (
        args.signed_headroom,
        args.headroom_signature,
        args.headroom_authorized_signers,
    )
    if any(item is not None for item in headroom_args) and not all(
        item is not None for item in headroom_args
    ):
        print(
            "llmtracefx-modal-l4-execute: --signed-headroom, "
            "--headroom-signature, and --headroom-authorized-signers must be "
            "supplied together",
            file=sys.stderr,
        )
        return 1
    signed_headroom = (
        read_structured_receipt(args.signed_headroom)
        if args.signed_headroom is not None
        else None
    )
    try:
        if args.action == "preflight":
            gates = preflight(
                credential_exposure_attestation_path=(
                    args.credential_exposure_attestation
                ),
                authorization_path=args.authorization,
                signature_path=args.authorization_signature,
                authorized_signers_path=args.authorized_signers,
                rate_receipt_path=args.rate_receipt,
                workspace=args.workspace,
                signed_headroom=signed_headroom,
                headroom_signature_path=args.headroom_signature,
                headroom_authorized_signers_path=args.headroom_authorized_signers,
            )
            print(
                json.dumps(
                    {
                        "gates_passed": True,
                        "credential_exposure_cleared": True,
                        "plan_sha256": gates["plan"].content_sha256,
                        "headroom": gates["headroom"],
                        "provider_sdk_imported": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        document = execute(
            credential_exposure_attestation_path=(args.credential_exposure_attestation),
            authorization_path=args.authorization,
            signature_path=args.authorization_signature,
            authorized_signers_path=args.authorized_signers,
            rate_receipt_path=args.rate_receipt,
            workspace=args.workspace,
            signed_headroom=signed_headroom,
            headroom_signature_path=args.headroom_signature,
            headroom_authorized_signers_path=args.headroom_authorized_signers,
        )
        published = bool(document.get("published"))
        print(
            json.dumps(
                {"status": document["status"], "published": published},
                indent=2,
                sort_keys=True,
            )
        )
        # A non-complete status, or a complete run whose result was not
        # published (for example because teardown was incomplete), never
        # returns CLI success.
        return 0 if published else 2
    except (OSError, ValueError, RateRefreshError, ModalL4ContractError) as exc:
        print(f"llmtracefx-modal-l4-execute: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
