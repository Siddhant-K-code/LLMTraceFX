"""Command-line entry point for the vLLM KV-truth host orchestrator.

Deliberately accepts only protected local file/directory paths -- never
host/user/key/known-hosts paths as individual flags, and never anything
host-identifying is printed.
``main()`` returns a process exit code and never raises past its own frame;
every error is reduced to a single, non-leaking line on stderr.
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
    reject_credential_environment,
    verify_authorization_signature,
)

PROG = "llmtracefx-vllm-kv-truth"
CLEAN_ENV_LAUNCHER = "run-vllm-kv-truth-clean-env.sh"
_CLEAN_PARENT_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
_PLATFORM_SYNTHESIZED_ENVIRONMENT_NAMES = {"__CF_USER_TEXT_ENCODING"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help=(
            "execute the complete one-attempt remote protocol for one " "authorized run"
        ),
    )
    run_parser.add_argument(
        "--execution-config",
        required=True,
        type=Path,
        help=(
            "path to a protected JSON file containing host/user/key/"
            "known-hosts/remote-path facts; never passed as individual flags"
        ),
    )
    run_parser.add_argument(
        "--authorization",
        required=True,
        type=Path,
        help="path to the explicit, self-sealed run authorization JSON",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "pre-created protected output directory; must exactly match "
            "local_evidence_dir in the execution config"
        ),
    )
    subparsers.add_parser(
        "preflight-clean-environment",
        help=(
            "verify the clean-launcher environment and exit without provider, "
            "SSH, model, image, or GPU activity"
        ),
    )
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


def _run(args: argparse.Namespace) -> int:
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


def _preflight_clean_environment() -> int:
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
    print("clean environment preflight: ok")
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
        if args.command == "run":
            return _run(args)
        if args.command == "preflight-clean-environment":
            return _preflight_clean_environment()
        return _verify_public_bundle(args)
    except HostOrchestrationError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    except evidence.EvidenceError as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{PROG}: error: {exc.strerror or exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
