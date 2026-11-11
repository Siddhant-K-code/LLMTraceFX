from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deploy_space", ROOT / "scripts" / "deploy_space.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


deploy_space = _load_script()


def test_pins_the_git_requirement_to_the_release() -> None:
    text = (
        "gradio==6.31.0\n"
        "llmtracefx @ git+https://github.com/Siddhant-K-code/LLMTraceFX@v1.10.0\n"
    )
    assert deploy_space.pin_requirements(text, "1.11.0") == (
        "gradio==6.31.0\nllmtracefx==1.11.0\n"
    )


@pytest.mark.parametrize(
    "line", ["llmtracefx>=1.0", "LLMTraceFX==1.9.0", "llmtracefx[mlx]==1.9.0"]
)
def test_pins_any_spelling_of_the_requirement(line: str) -> None:
    assert deploy_space.pin_requirements(line, "2.0.1") == "llmtracefx==2.0.1\n"


def test_leaves_similarly_named_packages_alone() -> None:
    text = "llmtracefx-extra==1.0\nllmtracefx==1.0\n"
    assert deploy_space.pin_requirements(text, "1.2.3") == (
        "llmtracefx-extra==1.0\nllmtracefx==1.2.3\n"
    )


@pytest.mark.parametrize("text", ["gradio==6.31.0\n", "llmtracefx\nllmtracefx\n"])
def test_refuses_unless_exactly_one_requirement(text: str) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        deploy_space.pin_requirements(text, "1.0.0")


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0rc1", "1.0.0\n"])
def test_refuses_non_release_versions(version: str) -> None:
    with pytest.raises(ValueError, match="X.Y.Z"):
        deploy_space.pin_requirements("llmtracefx\n", version)


def test_repository_requirements_can_be_pinned() -> None:
    text = (ROOT / "spaces" / "evidence-viewer" / "requirements.txt").read_text()
    pinned = deploy_space.pin_requirements(text, "1.10.0")
    assert "llmtracefx==1.10.0" in pinned.splitlines()
    assert "git+" not in pinned


def test_dry_run_stages_the_space_without_uploading(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    status = deploy_space.main(
        ["--version", "1.10.0", "--space", "owner/space", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert status == 0
    for name in ("app.py", "viewer.py", "README.md", "requirements.txt"):
        assert name in out.splitlines()
    assert "examples/synthetic-record.json" in out.splitlines()
    assert "llmtracefx==1.10.0" in out.splitlines()


def test_upload_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert deploy_space.main(["--version", "1.10.0", "--space", "owner/space"]) == 1


@pytest.mark.parametrize("space", ["space", "owner/space/extra", "-x/y", "a b/c"])
def test_rejects_malformed_space_ids(space: str) -> None:
    with pytest.raises(SystemExit):
        deploy_space.main(["--version", "1.10.0", "--space", space, "--dry-run"])
