"""Fail-closed Qwen3-8B acquisition inside the pinned runtime image."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import MODEL_ID, MODEL_REVISION

from .runner import BASE_IMAGE_REFERENCE

DOWNLOADER_PACKAGE = "huggingface-hub"
DOWNLOADER_VERSION = "1.13.0"
DOWNLOADER_INTERFACE = "huggingface_hub.snapshot_download"
DOWNLOADER_SOURCE = BASE_IMAGE_REFERENCE

_SAFE_FAILURE_MESSAGES = {
    "model_download_interface_missing": (
        "the pinned runtime image does not provide the authorized downloader interface"
    ),
    "model_download_version_mismatch": (
        "the pinned runtime image downloader version does not match authorization"
    ),
    "model_download_failed": "the in-container model download failed",
}


class ModelDownloadRefusal(RuntimeError):
    """A safe, allowlisted acquisition refusal."""

    def __init__(self, reason_code: str) -> None:
        if reason_code not in _SAFE_FAILURE_MESSAGES:
            raise ValueError("unknown model download refusal code")
        self.reason_code = reason_code
        super().__init__(_SAFE_FAILURE_MESSAGES[reason_code])


def downloader_attestation(
    *,
    version_fn: Callable[[str], str] | None = None,
    import_module_fn: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    """Verify the exact installed package and callable used for acquisition."""

    version_fn = version_fn or importlib.metadata.version
    import_module_fn = import_module_fn or importlib.import_module
    try:
        version = version_fn(DOWNLOADER_PACKAGE)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ModelDownloadRefusal("model_download_interface_missing") from exc
    if version != DOWNLOADER_VERSION:
        raise ModelDownloadRefusal("model_download_version_mismatch")
    try:
        module = import_module_fn("huggingface_hub")
        interface = module.snapshot_download
    except (ImportError, AttributeError) as exc:
        raise ModelDownloadRefusal("model_download_interface_missing") from exc
    if not callable(interface):
        raise ModelDownloadRefusal("model_download_interface_missing")
    return {
        "schema_version": "1",
        "package": DOWNLOADER_PACKAGE,
        "version": version,
        "interface": DOWNLOADER_INTERFACE,
        "source": DOWNLOADER_SOURCE,
    }


def _manifest_allow_patterns() -> list[str]:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "llmtracefx"
        / "optimizer"
        / "lab"
        / "qwen3_8b"
        / "data"
        / "qwen3-8b-conversion-manifest-v1.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload["source"]["files"]
    return [str(item["path"]) for item in files]


def download_model(
    destination: Path,
    scratch: Path,
    *,
    version_fn: Callable[[str], str] | None = None,
    import_module_fn: Callable[[str], Any] | None = None,
) -> dict[str, str]:
    """Download only the pinned model files through the attested interface."""

    import_module_fn = import_module_fn or importlib.import_module
    if not destination.is_absolute() or not scratch.is_absolute():
        raise ModelDownloadRefusal("model_download_failed")
    if (
        destination == scratch
        or destination in scratch.parents
        or scratch in destination.parents
    ):
        raise ModelDownloadRefusal("model_download_failed")
    if destination.is_symlink() or scratch.is_symlink():
        raise ModelDownloadRefusal("model_download_failed")

    original_environment = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(
            {
                "HOME": str(scratch),
                "HF_HOME": str(scratch),
                "HF_HUB_CACHE": str(scratch / "hub"),
                "HF_ASSETS_CACHE": str(scratch / "assets"),
                "XDG_CACHE_HOME": str(scratch / "xdg"),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            }
        )
        attestation = downloader_attestation(
            version_fn=version_fn,
            import_module_fn=import_module_fn,
        )
        module = import_module_fn("huggingface_hub")
        snapshot_download = module.snapshot_download
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(destination),
            allow_patterns=_manifest_allow_patterns(),
            token=False,
        )
    except ModelDownloadRefusal:
        raise
    except Exception as exc:
        raise ModelDownloadRefusal("model_download_failed") from exc
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
    return {
        **attestation,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("attest")
    download = subparsers.add_parser("download")
    download.add_argument("--destination", required=True, type=Path)
    download.add_argument("--scratch", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = (
            downloader_attestation()
            if args.command == "attest"
            else download_model(args.destination, args.scratch)
        )
    except ModelDownloadRefusal as exc:
        print(f"LLMTRACEFX_REASON={exc.reason_code}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
