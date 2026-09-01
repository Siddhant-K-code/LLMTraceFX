"""Command line entrypoint for the pinned local inference lab."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from .core import (
    LabError,
    acquire_model,
    assess_safety,
    model_files_present,
    run_lab,
    verify_catalog,
    verify_evidence,
    verify_model,
    write_reports,
)
from .manifest import LabManifest, LabManifestError

DEFAULT_MANIFEST_RESOURCE = "data/lab-manifest-v1.json"


def _load(args: argparse.Namespace) -> tuple[LabManifest, str, Path, Path]:
    if args.manifest is None:
        resource = resources.files("llmtracefx.optimizer.lab").joinpath(
            DEFAULT_MANIFEST_RESOURCE
        )
        manifest = LabManifest.from_json(resource.read_text(encoding="utf-8"))
        manifest_source = (
            f"package:llmtracefx.optimizer.lab/{DEFAULT_MANIFEST_RESOURCE}"
        )
    else:
        manifest_path = Path(args.manifest)
        manifest = LabManifest.read_json(manifest_path)
        manifest_source = str(manifest_path)
    workspace = (
        Path(args.workspace)
        if args.workspace is not None
        else Path(manifest.artifacts.workspace)
    )
    model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else Path(manifest.artifacts.model_cache)
    )
    verify_catalog(manifest)
    return manifest, manifest_source, workspace, model_path


def _cmd_plan(args: argparse.Namespace) -> int:
    manifest, manifest_source, workspace, model_path = _load(args)
    present = model_files_present(manifest, model_path)
    workspace_decision = assess_safety(manifest, workspace, include_download=False)
    model_decision = assess_safety(
        manifest,
        model_path.parent,
        include_download=not present,
    )
    payload = {
        "action": "plan",
        "no_spend": True,
        "downloads_performed": False,
        "model_present_by_size": present,
        "manifest": manifest_source,
        "workspace": str(workspace),
        "model_path": str(model_path),
        "model": {
            "official_id": manifest.model.official_id,
            "official_revision": manifest.model.official_revision,
            "repository_id": manifest.model.repository_id,
            "revision": manifest.model.revision,
            "license": manifest.model.license,
            "quantization": manifest.model.quantization,
            "expected_download_bytes": manifest.model.expected_download_bytes,
        },
        "tiers": [tier.name for tier in manifest.context_tiers],
        "safety": model_decision.to_dict(),
        "workspace_safety": workspace_decision.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0 if model_decision.safe and workspace_decision.safe else 2


def _cmd_acquire(args: argparse.Namespace) -> int:
    manifest, _, workspace, model_path = _load(args)
    result = acquire_model(manifest, model_path=model_path, workspace=workspace)
    print(json.dumps(result, indent=2, sort_keys=False))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    manifest, _, workspace, model_path = _load(args)
    if not model_files_present(manifest, model_path):
        if not args.acquire:
            raise LabError(
                "pinned model is not present; rerun with --acquire after "
                "reviewing the no-spend plan"
            )
        acquire_model(manifest, model_path=model_path, workspace=workspace)
    else:
        verify_model(manifest, model_path)
    state = run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier=args.max_tier,
        resume=not args.no_resume,
    )
    report = write_reports(
        manifest,
        workspace=workspace,
        shareable_dir=(
            Path(args.shareable_dir)
            if args.shareable_dir
            else Path(manifest.artifacts.shareable_example_dir)
        ),
    )
    print(
        json.dumps(
            {
                "state": state,
                "report": report,
            },
            indent=2,
            sort_keys=False,
        )
    )
    return 0 if not state["stop_reasons"] else 2


def _cmd_report(args: argparse.Namespace) -> int:
    manifest, _, workspace, _ = _load(args)
    report = write_reports(
        manifest,
        workspace=workspace,
        shareable_dir=(
            Path(args.shareable_dir)
            if args.shareable_dir
            else Path(manifest.artifacts.shareable_example_dir)
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest, _, workspace, model_path = _load(args)
    verify_model(manifest, model_path)
    result = verify_evidence(manifest, workspace=workspace)
    print(json.dumps(result, indent=2, sort_keys=False))
    return 0 if result["verified"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-m5-lab",
        description=(
            "Pinned, safety-gated Qwen3.8-27B MLX-VLM evidence lab. "
            "The default action is an offline/no-download plan."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=("plan", "acquire", "run", "report", "verify"),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Optional lab manifest path; defaults to the versioned manifest "
            "packaged with llmtracefx"
        ),
    )
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--max-tier",
        choices=("2k", "8k", "16k"),
        default="2k",
        help="Highest requested tier; every prior tier must pass its safety gate",
    )
    parser.add_argument(
        "--acquire",
        action="store_true",
        help="Allow `run` to download the pinned public model when absent",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore hash-matching completed row artifacts",
    )
    parser.add_argument(
        "--shareable-dir",
        default=None,
        help="Optional destination for sanitized JSON and self-contained HTML",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv[:1] == ["autopsy"]:
        # Routed to the autopsy module's own self-contained argparse
        # parser, mirroring the standalone `llmtracefx-m5-frontier`
        # entrypoint, so the existing plan|acquire|run|report|verify
        # actions and default action below are completely unaffected.
        from . import autopsy

        return autopsy.main(raw_argv[1:])
    args = build_parser().parse_args(argv)
    commands = {
        "plan": _cmd_plan,
        "acquire": _cmd_acquire,
        "run": _cmd_run,
        "report": _cmd_report,
        "verify": _cmd_verify,
    }
    try:
        return commands[args.action](args)
    except (OSError, UnicodeError, LabManifestError, LabError) as exc:
        print(f"M5 lab failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
