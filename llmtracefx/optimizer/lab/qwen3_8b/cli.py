"""Command line entrypoint for the planned Qwen3-8B M5 Pro control.

Packaged as ``llmtracefx-m5-control`` (see ``pyproject.toml``), fully
namespace-separated from ``llmtracefx-m5-lab``/``llmtracefx-m5-frontier``:
every default manifest, cache path, and workspace path here lives under
a ``qwen3-8b`` prefix so nothing this CLI does can read, write, or
overwrite the packaged Qwen3.8-27B lab's artifacts. No conversion or
benchmark has run on any machine yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from ...collectors._shared import atomic_write_text
from ..core import LabError, model_files_present, verify_catalog, verify_model
from ..manifest import LabManifest, LabManifestError
from .benchmark import run_benchmark
from .control_manifest import (
    ControlManifestError,
    ControlManifestTemplate,
    bind_control_manifest,
    build_bound_manifest_payload,
)
from .conversion import (
    ConversionError,
    ConversionReceipt,
    assess_conversion_safety,
    run_conversion,
    source_cache_has_existing_content,
)
from .conversion_manifest import ConversionManifest, ConversionManifestError
from .report import verify_control_evidence, write_control_reports

DEFAULT_CONVERSION_MANIFEST_RESOURCE = "data/qwen3-8b-conversion-manifest-v1.json"
DEFAULT_CONTROL_TEMPLATE_RESOURCE = "data/qwen3-8b-control-manifest-template-v1.json"


def _packaged_text(name: str) -> str:
    resource = resources.files("llmtracefx.optimizer.lab.qwen3_8b").joinpath(name)
    return resource.read_text(encoding="utf-8")


def _load_conversion_manifest(path: str | None) -> tuple[ConversionManifest, str]:
    if path is None:
        payload = _packaged_text(DEFAULT_CONVERSION_MANIFEST_RESOURCE)
        return (
            ConversionManifest.from_json(payload),
            f"package:llmtracefx.optimizer.lab.qwen3_8b/{DEFAULT_CONVERSION_MANIFEST_RESOURCE}",
        )
    return ConversionManifest.read_json(Path(path)), str(path)


def _load_control_template(path: str | None) -> tuple[ControlManifestTemplate, str]:
    if path is None:
        payload = _packaged_text(DEFAULT_CONTROL_TEMPLATE_RESOURCE)
        return (
            ControlManifestTemplate.from_json(payload),
            f"package:llmtracefx.optimizer.lab.qwen3_8b/{DEFAULT_CONTROL_TEMPLATE_RESOURCE}",
        )
    return ControlManifestTemplate.read_json(Path(path)), str(path)


def _cmd_plan(args: argparse.Namespace) -> int:
    conversion, conversion_source = _load_conversion_manifest(args.conversion_manifest)
    source_path = Path(args.source_path or conversion.artifacts.source_cache)
    output_path = Path(args.output_path or conversion.artifacts.output_cache)
    conversion_workspace = Path(
        args.conversion_workspace or conversion.artifacts.workspace
    )
    conversion_decision = assess_conversion_safety(
        conversion,
        conversion_workspace,
        include_source_download=not source_cache_has_existing_content(source_path),
    )
    receipt_path = conversion_workspace / "conversion-receipt.json"
    payload = {
        "action": "plan",
        "no_spend": True,
        "downloads_performed": False,
        "conversion_manifest": conversion_source,
        "conversion_id": conversion.conversion_id,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "conversion_workspace": str(conversion_workspace),
        "source_fully_pinned": conversion.source.fully_pinned,
        "output_already_present": output_path.exists(),
        "conversion_receipt_present": receipt_path.is_file(),
        "model": {
            "official_id": conversion.source.official_id,
            "official_revision": conversion.source.official_revision,
            "expected_source_bytes": conversion.source.expected_source_bytes,
            "expected_output_bytes": conversion.expected_output_bytes,
            "converter": conversion.converter.package,
            "converter_version": conversion.converter.version,
            "converter_git_revision": conversion.converter.git_revision,
        },
        "required_free_disk_bytes": conversion_decision.required_free_disk_bytes,
        "observed_free_disk_bytes": conversion_decision.snapshot.disk_free_bytes,
        "conversion_safety": conversion_decision.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0 if conversion_decision.safe else 2


def _cmd_convert(args: argparse.Namespace) -> int:
    conversion, _ = _load_conversion_manifest(args.conversion_manifest)
    source_path = Path(args.source_path or conversion.artifacts.source_cache)
    output_path = Path(args.output_path or conversion.artifacts.output_cache)
    workspace = Path(args.conversion_workspace or conversion.artifacts.workspace)
    journal = run_conversion(
        conversion,
        source_path=source_path,
        output_path=output_path,
        workspace=workspace,
    )
    print(json.dumps(journal, indent=2, sort_keys=False))
    return 0 if journal["status"] == "completed" else 2


def _cmd_bind(args: argparse.Namespace) -> int:
    if args.receipt is None or args.output is None:
        raise ControlManifestError("bind requires both --receipt and --output")
    conversion, _ = _load_conversion_manifest(args.conversion_manifest)
    template, _ = _load_control_template(args.control_template)
    receipt = ConversionReceipt.read_json(Path(args.receipt))
    payload = build_bound_manifest_payload(
        template, receipt, conversion_manifest=conversion
    )
    bound = LabManifest.from_dict(payload)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_path, json.dumps(payload, indent=2, sort_keys=False) + "\n"
    )
    print(
        json.dumps(
            {
                "action": "bind",
                "manifest_path": str(output_path),
                "lab_id": bound.lab_id,
                "model_binding_fingerprint": bound.model.revision,
                "model_binding_fingerprint_type": (
                    "self_conversion_identity_fingerprint_not_a_git_commit"
                ),
                "output_total_bytes": bound.model.expected_download_bytes,
            },
            indent=2,
            sort_keys=False,
        )
    )
    return 0


def _load_bound_manifest(args: argparse.Namespace) -> tuple[LabManifest, str]:
    if args.manifest is not None:
        return LabManifest.read_json(Path(args.manifest)), str(args.manifest)
    if args.receipt is not None:
        conversion, _ = _load_conversion_manifest(args.conversion_manifest)
        template, _ = _load_control_template(args.control_template)
        receipt = ConversionReceipt.read_json(Path(args.receipt))
        bound = bind_control_manifest(template, receipt, conversion_manifest=conversion)
        return bound, f"bound-from-receipt:{args.receipt}"
    raise LabError(
        "either --manifest (an already-bound control manifest) or --receipt "
        "(a completed conversion receipt to bind on the fly) is required"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    manifest, _ = _load_bound_manifest(args)
    manifest_path = Path(args.manifest) if args.manifest is not None else None
    if manifest_path is None:
        raise LabError(
            "run requires --manifest pointing at a materialized bound "
            "manifest file (each isolated row subprocess re-reads it); "
            "use `bind` first if you only have a receipt"
        )
    workspace = Path(args.workspace or manifest.artifacts.workspace)
    model_path = Path(args.model_path or manifest.artifacts.model_cache)
    verify_catalog(manifest)
    if not model_files_present(manifest, model_path):
        raise LabError(
            "the self-converted model is not present at the expected path; "
            "run `convert` (and `bind`) first"
        )
    verify_model(manifest, model_path)
    state = run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=model_path,
        max_tier=args.max_tier,
        run_mode=args.run_mode,
        clean_boot_confirmed=args.confirm_clean_boot,
        resume=not args.no_resume,
        row_timeout_seconds=args.row_timeout_seconds,
    )
    report = write_control_reports(
        manifest,
        workspace=workspace,
        shareable_dir=(
            Path(args.shareable_dir)
            if args.shareable_dir
            else Path(manifest.artifacts.shareable_example_dir)
        ),
    )
    print(json.dumps({"state": state, "report": report}, indent=2, sort_keys=False))
    return 0 if not state["stop_reasons"] else 2


def _cmd_report(args: argparse.Namespace) -> int:
    manifest, _ = _load_bound_manifest(args)
    workspace = Path(args.workspace or manifest.artifacts.workspace)
    report = write_control_reports(
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
    manifest, _ = _load_bound_manifest(args)
    workspace = Path(args.workspace or manifest.artifacts.workspace)
    model_path = Path(args.model_path or manifest.artifacts.model_cache)
    verify_model(manifest, model_path)
    result = verify_control_evidence(manifest, workspace=workspace)
    print(json.dumps(result, indent=2, sort_keys=False))
    return 0 if result["verified"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-m5-control",
        description=(
            "Planned, preparatory Qwen3-8B MLX-LM self-conversion control for "
            "the M5 Pro lab (no conversion or benchmark has run yet), fully "
            "namespace-separated from the packaged Qwen3.8-27B lab. The "
            "default action is an offline/no-download plan."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=("plan", "convert", "bind", "run", "report", "verify"),
    )
    parser.add_argument("--conversion-manifest", default=None)
    parser.add_argument("--control-template", default=None)
    parser.add_argument("--source-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--conversion-workspace", default=None)
    parser.add_argument(
        "--receipt", default=None, help="Path to a completed conversion-receipt.json"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination path for `bind`'s materialized control manifest",
    )
    parser.add_argument(
        "--manifest", default=None, help="Path to an already-bound control manifest"
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
        "--run-mode",
        choices=("exploratory", "publication"),
        default="exploratory",
        help="`publication` requires --confirm-clean-boot",
    )
    parser.add_argument("--confirm-clean-boot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--row-timeout-seconds",
        type=float,
        default=300.0,
        help="Parent-enforced wall-clock bound for each isolated row subprocess",
    )
    parser.add_argument("--shareable-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "plan": _cmd_plan,
        "convert": _cmd_convert,
        "bind": _cmd_bind,
        "run": _cmd_run,
        "report": _cmd_report,
        "verify": _cmd_verify,
    }
    try:
        return commands[args.action](args)
    except (
        OSError,
        UnicodeError,
        LabManifestError,
        LabError,
        ConversionManifestError,
        ConversionError,
        ControlManifestError,
    ) as exc:
        print(f"M5 control failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
