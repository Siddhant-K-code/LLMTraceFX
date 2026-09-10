"""Public command-line entry point for offline KV-truth bundle verification.

Remote execution is intentionally absent from this console-script surface.
The separately installed clean bootstrap verifies its trust manifest before
calling :func:`bootstrap_dispatch` for preflight or execution.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import evidence
from .lifecycle import (
    HostOrchestrationError,
    ProtectedExecutionConfig,
    RemoteOrchestrator,
    RunAuthorization,
    SubprocessCommandRunner,
    enroll_tofu_host_key,
    reject_credential_environment,
    verify_authorization_signature,
)

PROG = "llmtracefx-vllm-kv-truth"
CLEAN_ENV_LAUNCHER = "run-vllm-kv-truth-clean-env.py"
_CLEAN_PARENT_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
_PLATFORM_SYNTHESIZED_ENVIRONMENT_NAMES = {"__CF_USER_TEXT_ENCODING"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser(
        "verify-public-bundle",
        help="portably verify a previously exported public-redacted evidence bundle",
    )
    verify_parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help="directory containing bundle.json, report.txt, report.svg, SHA256SUMS",
    )

    return parser


def _build_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"{PROG} (trusted bootstrap)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--execution-config", required=True, type=Path)
    run_parser.add_argument("--authorization", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    enroll_parser = subparsers.add_parser("enroll-tofu")
    enroll_parser.add_argument("--enrollment-request", required=True, type=Path)
    subparsers.add_parser("preflight")
    return parser


def _run(args: argparse.Namespace) -> int:
    _require_clean_parent_environment()
    config = ProtectedExecutionConfig.load(args.execution_config)
    authorization = RunAuthorization.read(args.authorization)
    if not args.output_dir.is_absolute() or ".." in args.output_dir.parts:
        raise HostOrchestrationError(
            "output directory must be an unambiguous absolute path"
        )
    if args.output_dir != config.local_evidence_dir:
        raise HostOrchestrationError(
            "output directory does not match the protected execution config"
        )
    runner = SubprocessCommandRunner()
    if authorization.signature_path is not None:
        assert authorization.authorized_signers_path is not None
        verify_authorization_signature(
            runner,
            authorization,
            signature_path=authorization.signature_path,
            authorized_signers_path=authorization.authorized_signers_path,
        )
    orchestrator = RemoteOrchestrator(
        config=config,
        authorization=authorization,
        runner=runner,
    )
    outcomes = orchestrator.run(local_evidence_bundle_dir=config.local_evidence_dir)
    failed = [outcome for outcome in outcomes if not outcome.ok]
    for outcome in outcomes:
        print(f"{outcome.stage}: {'ok' if outcome.ok else 'failed'}")
    return 1 if failed else 0


def _require_clean_parent_environment() -> None:
    reject_credential_environment(os.environ)
    names = set(os.environ)
    expected_names = set(_CLEAN_PARENT_ENVIRONMENT)
    changed = sorted(
        name
        for name, value in _CLEAN_PARENT_ENVIRONMENT.items()
        if os.environ.get(name) != value
    )
    unexpected = sorted(
        names - expected_names - _PLATFORM_SYNTHESIZED_ENVIRONMENT_NAMES
    )
    if changed or unexpected:
        raise HostOrchestrationError(
            "clean launcher environment contract failed "
            f"(changed_or_missing={changed}, unexpected={unexpected}); "
            f"use {CLEAN_ENV_LAUNCHER}, not manual variable unsets"
        )


def _preflight_clean_environment() -> int:
    _require_clean_parent_environment()
    print("clean environment preflight: ok")
    return 0


def _enroll_tofu(args: argparse.Namespace) -> int:
    _require_clean_parent_environment()
    receipt = enroll_tofu_host_key(args.enrollment_request, SubprocessCommandRunner())
    print(f"TOFU enrollment receipt SHA-256: {receipt.receipt_sha256}")
    print("host identity: tofu_unverified (provider identity not authenticated)")
    return 0


def _verify_public_bundle(args: argparse.Namespace) -> int:
    bundle = evidence.verify_public_bundle_directory(args.bundle_dir)
    payload = bundle.to_dict()
    print(f"protocol: {payload['protocol_id']}")
    print(f"run_mode: {payload['run_mode']}")
    print(f"experiment_nonce: {payload['experiment_nonce']}")
    print("verification: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _verify_public_bundle(args)
    except evidence.EvidenceError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{PROG}: error: {exc.strerror or exc}", file=sys.stderr)
        return 1


def bootstrap_dispatch(argv: list[str]) -> int:
    """Dispatch verified bootstrap-only operations.

    This is an internal separation of command surfaces, not an unforgeable
    Python capability. Same-user arbitrary Python import is outside the launch
    threat model.
    """

    parser = _build_bootstrap_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _preflight_clean_environment()
        if args.command == "enroll-tofu":
            return _enroll_tofu(args)
        return _run(args)
    except HostOrchestrationError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 128 + exc.signal_number if exc.signal_number is not None else 1
    except evidence.EvidenceError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{PROG}: error: {exc.strerror or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
