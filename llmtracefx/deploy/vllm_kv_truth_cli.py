"""Command-line entry point for the vLLM KV-truth host orchestrator.

Deliberately accepts only two file paths -- never host/user/key/known-hosts
paths as individual flags, and never anything host-identifying is printed.
``main()`` returns a process exit code and never raises past its own frame;
every error is reduced to a single, non-leaking line on stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import vllm_kv_truth_evidence as evidence
from .vllm_kv_truth_lifecycle import (
    HostOrchestrationError,
    ProtectedExecutionConfig,
    RemoteOrchestrator,
    RunAuthorization,
    SubprocessCommandRunner,
    verify_authorization_signature,
)

PROG = "llmtracefx-vllm-kv-truth"


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
