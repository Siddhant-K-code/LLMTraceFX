"""Offline CLI for adjudicating the CloudRift GLM validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..optimizer.cli import SecureArgumentParser, _argument_scrub_scope
from ..optimizer.collectors._shared import atomic_write_text
from ..optimizer.collectors.openai_api import redact_text_for_dry_run
from .cloudrift import CloudRiftSnapshot, build_cloudrift_plan
from .errors import DeploymentPlanError
from .model_inventory import load_inventory

PROG = "llmtracefx-cloudrift"


def build_parser() -> argparse.ArgumentParser:
    parser = SecureArgumentParser(
        prog=PROG,
        description=(
            "Adjudicate one CloudRift GLM-5.3-Flash lifecycle offline. "
            "Never authenticates, provisions, or spends."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        raw = json.loads(args.snapshot.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise DeploymentPlanError("CloudRift snapshot root must be an object")
        snapshot = CloudRiftSnapshot.from_dict(raw)
        inventory = load_inventory(args.inventory)
        plan = build_cloudrift_plan(snapshot, inventory)
        rendered = json.dumps(plan.to_dict(), indent=2, allow_nan=False) + "\n"
        rendered = redact_text_for_dry_run(rendered, None)
        if args.output is not None:
            atomic_write_text(args.output, rendered)
        print(rendered, end="")
        if not plan.approved:
            print(
                f"{PROG}: refused; no paid step may run while blockers remain",
                file=sys.stderr,
            )
            return 1
        return 0
    except (DeploymentPlanError, OSError, json.JSONDecodeError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    with _argument_scrub_scope(parser, raw_argv):
        args = parser.parse_args(raw_argv)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
