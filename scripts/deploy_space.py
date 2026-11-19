"""Stage the evidence viewer Space for a release and upload it to Hugging Face.

The staged copy installs the released wheel straight from PyPI's file host
(``llmtracefx @ https://files.pythonhosted.org/...whl#sha256=...``) instead of
the git reference in the repository's requirements.txt. A direct URL means pip
never reads an index, so a stale index mirror in the Space builder cannot make
the build fail, and the hash pins the exact file.

Used by .github/workflows/deploy-space.yml. Reads the token from ``HF_TOKEN``.
``--dry-run`` resolves the wheel, stages the files and prints the pinned
requirements without importing huggingface_hub or contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

PACKAGE = "llmtracefx"
REPO_URL = "https://github.com/Siddhant-K-code/LLMTraceFX"
DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "spaces" / "evidence-viewer"
VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
SPACE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")
# Files the Space owns that this deploy may replace or remove. .gitattributes
# and other files Hugging Face creates are left alone. fnmatch `*` crosses `/`.
DELETE_PATTERNS = ["*.py", "*.txt", "*.md", "examples/*"]
PYPI_JSON = "https://pypi.org/pypi/{package}/{version}/json"
FILE_HOST = "https://files.pythonhosted.org/"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class WheelError(Exception):
    pass


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def release_wheel(version: str, fetch: Callable[[str], dict] = _fetch_json) -> str:
    """Return the release wheel's file host URL with its sha256 fragment."""
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"version must be X.Y.Z, got {version!r}")
    url = PYPI_JSON.format(package=PACKAGE, version=version)
    try:
        data = fetch(url)
    except Exception as exc:
        raise WheelError(f"could not read {url}: {exc}") from exc
    filename = f"{PACKAGE}-{version}-py3-none-any.whl"
    wheels = [
        entry
        for entry in data.get("urls", [])
        if entry.get("packagetype") == "bdist_wheel"
        and entry.get("filename") == filename
    ]
    if len(wheels) != 1:
        raise WheelError(f"expected one {filename} on PyPI, found {len(wheels)}")
    wheel_url = wheels[0].get("url", "")
    sha256 = wheels[0].get("digests", {}).get("sha256", "")
    if not wheel_url.startswith(FILE_HOST) or not wheel_url.endswith("/" + filename):
        raise WheelError(f"unexpected wheel URL {wheel_url!r}")
    if not SHA256_RE.fullmatch(sha256):
        raise WheelError(f"unexpected sha256 {sha256!r}")
    return f"{wheel_url}#sha256={sha256}"


def pin_requirements(text: str, version: str, wheel: str | None = None) -> str:
    """Replace the single llmtracefx requirement with an exact pin.

    With ``wheel`` the pin is a direct URL to that file; otherwise it is
    ``llmtracefx==X.Y.Z``.
    """
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"version must be X.Y.Z, got {version!r}")
    pin = f"{PACKAGE} @ {wheel}" if wheel else f"{PACKAGE}=={version}"
    lines = []
    replaced = 0
    for line in text.splitlines():
        match = NAME_RE.match(line.strip())
        if match and _normalize(match.group(1)) == PACKAGE:
            lines.append(pin)
            replaced += 1
        else:
            lines.append(line)
    if replaced != 1:
        raise ValueError(
            f"expected exactly one {PACKAGE} requirement, found {replaced}"
        )
    return "\n".join(lines) + "\n"


def stage(source: Path, destination: Path, version: str, wheel: str) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"),
    )
    requirements = destination / "requirements.txt"
    requirements.write_text(pin_requirements(requirements.read_text(), version, wheel))


def wake(api, space: str) -> str:
    """Restart the Space if it is paused, so it serves the deployed files.

    Hardware is never changed. A failed restart is reported, not raised: the
    files are already uploaded.
    """
    runtime = api.get_space_runtime(space)
    hardware = runtime.requested_hardware or runtime.hardware or "none"
    if str(runtime.stage) != "PAUSED":
        return f"Hardware: {hardware}, stage {str(runtime.stage)}"
    try:
        runtime = api.restart_space(space)
    except Exception as exc:
        return (
            f"::warning::{space} is paused and could not be restarted: {exc}. "
            "Restart it in the Space settings."
        )
    return f"Hardware: {hardware}, restarted, stage {str(runtime.stage)}"


def upload(staged: Path, space: str, token: str, version: str, revision: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        space,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )
    info = api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=staged,
        commit_message=f"Deploy {PACKAGE} {version}",
        commit_description=f"Source: {REPO_URL}/tree/{revision}",
        delete_patterns=DELETE_PATTERNS,
    )
    print(f"Commit: {info.commit_url}")
    return wake(api, space)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="released version X.Y.Z")
    parser.add_argument("--space", required=True, help="Space id, owner/name")
    parser.add_argument(
        "--revision", help="git ref the files came from (default: vVERSION)"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not VERSION_RE.fullmatch(args.version):
        parser.error(f"--version must be X.Y.Z, got {args.version!r}")
    if not SPACE_ID_RE.fullmatch(args.space):
        parser.error(f"--space must be owner/name, got {args.space!r}")
    revision = args.revision or f"v{args.version}"

    token = os.environ.get("HF_TOKEN", "")
    if not args.dry_run and not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 1

    try:
        wheel = release_wheel(args.version)
    except WheelError as exc:
        print(exc, file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "space"
        stage(args.source, staged, args.version, wheel)
        if args.dry_run:
            for path in sorted(p for p in staged.rglob("*") if p.is_file()):
                print(path.relative_to(staged))
            print("--- requirements.txt")
            print((staged / "requirements.txt").read_text(), end="")
            return 0
        status = upload(staged, args.space, token, args.version, revision)
    print(
        f"Deployed {PACKAGE} {args.version} to https://huggingface.co/spaces/{args.space}"
    )
    print(status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
