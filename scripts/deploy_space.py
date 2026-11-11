"""Stage the evidence viewer Space for a release and upload it to Hugging Face.

The staged copy installs the released package from PyPI (``llmtracefx==X.Y.Z``)
instead of the git reference in the repository's requirements.txt.

Used by .github/workflows/deploy-space.yml. Reads the token from ``HF_TOKEN``.
``--dry-run`` stages the files and prints the pinned requirements without
importing huggingface_hub or contacting Hugging Face.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
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


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pin_requirements(text: str, version: str) -> str:
    """Replace the single llmtracefx requirement with an exact PyPI pin."""
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"version must be X.Y.Z, got {version!r}")
    lines = []
    replaced = 0
    for line in text.splitlines():
        match = NAME_RE.match(line.strip())
        if match and _normalize(match.group(1)) == PACKAGE:
            lines.append(f"{PACKAGE}=={version}")
            replaced += 1
        else:
            lines.append(line)
    if replaced != 1:
        raise ValueError(
            f"expected exactly one {PACKAGE} requirement, found {replaced}"
        )
    return "\n".join(lines) + "\n"


def stage(source: Path, destination: Path, version: str) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"),
    )
    requirements = destination / "requirements.txt"
    requirements.write_text(pin_requirements(requirements.read_text(), version))


def upload(staged: Path, space: str, token: str, version: str, revision: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    info = api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=staged,
        commit_message=f"Deploy {PACKAGE} {version}",
        commit_description=f"Source: {REPO_URL}/tree/{revision}",
        delete_patterns=DELETE_PATTERNS,
    )
    return str(info.commit_url)


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

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "space"
        stage(args.source, staged, args.version)
        if args.dry_run:
            for path in sorted(p for p in staged.rglob("*") if p.is_file()):
                print(path.relative_to(staged))
            print("--- requirements.txt")
            print((staged / "requirements.txt").read_text(), end="")
            return 0
        url = upload(staged, args.space, token, args.version, revision)
    print(f"Deployed {PACKAGE} {args.version} to https://huggingface.co/spaces/{args.space}")
    print(f"Commit: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
