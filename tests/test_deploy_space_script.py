from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
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


def test_space_python_satisfies_requires_python() -> None:
    # Spaces default to Python 3.10 when python_version is unset, which cannot
    # install llmtracefx.
    readme = (ROOT / "spaces" / "evidence-viewer" / "README.md").read_text()
    match = re.search(r'^python_version:\s*"?(\d+)\.(\d+)"?\s*$', readme, re.M)
    assert match, "spaces/evidence-viewer/README.md must set python_version"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    floor = re.fullmatch(r">=\s*(\d+)\.(\d+)", project["requires-python"])
    assert floor
    space = (int(match[1]), int(match[2]))
    assert space >= (int(floor[1]), int(floor[2]))


WHEEL = (
    "https://files.pythonhosted.org/packages/d7/05/abc/"
    "llmtracefx-1.10.0-py3-none-any.whl"
)
SHA = "3d" * 32


def _pypi(version: str = "1.10.0", **wheel: object) -> dict:
    entry = {
        "packagetype": "bdist_wheel",
        "filename": f"llmtracefx-{version}-py3-none-any.whl",
        "url": WHEEL.replace("1.10.0", version),
        "digests": {"sha256": SHA},
    }
    entry.update(wheel)
    sdist = {
        "packagetype": "sdist",
        "filename": f"llmtracefx-{version}.tar.gz",
        "url": "https://files.pythonhosted.org/packages/ff/x.tar.gz",
        "digests": {"sha256": "ab" * 32},
    }
    return {"urls": [sdist, entry]}


def test_resolves_the_release_wheel_with_its_hash() -> None:
    urls: list[str] = []

    def fetch(url: str) -> dict:
        urls.append(url)
        return _pypi()

    assert deploy_space.release_wheel("1.10.0", fetch) == f"{WHEEL}#sha256={SHA}"
    assert urls == ["https://pypi.org/pypi/llmtracefx/1.10.0/json"]


@pytest.mark.parametrize(
    "data",
    [
        {"urls": []},
        _pypi(filename="llmtracefx-1.10.0-cp312-none-any.whl"),
        _pypi(url="https://example.com/llmtracefx-1.10.0-py3-none-any.whl"),
        _pypi(digests={"sha256": "nothex"}),
        _pypi(digests={}),
    ],
)
def test_refuses_a_missing_or_unexpected_wheel(data: dict) -> None:
    with pytest.raises(deploy_space.WheelError):
        deploy_space.release_wheel("1.10.0", lambda url: data)


def test_unreachable_pypi_is_a_wheel_error() -> None:
    calls: list[str] = []

    def fetch(url: str) -> dict:
        calls.append(url)
        raise OSError("404 Not Found")

    with pytest.raises(deploy_space.WheelError, match="404 Not Found"):
        deploy_space.release_wheel("1.10.0", fetch, sleep=lambda seconds: None)
    assert len(calls) == deploy_space.FETCH_ATTEMPTS


def test_retries_a_stale_404_until_the_release_is_listed() -> None:
    responses: list[object] = [OSError("404 Not Found"), OSError("404"), _pypi()]
    waits: list[float] = []

    def fetch(url: str) -> dict:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    assert (
        deploy_space.release_wheel("1.10.0", fetch, sleep=waits.append)
        == f"{WHEEL}#sha256={SHA}"
    )
    assert waits == [deploy_space.FETCH_RETRY_SECONDS] * 2


def test_pins_the_requirement_to_a_wheel_url() -> None:
    wheel = f"{WHEEL}#sha256={SHA}"
    assert deploy_space.pin_requirements(
        "gradio==6.31.0\nllmtracefx\n", "1.10.0", wheel
    ) == (f"gradio==6.31.0\nllmtracefx @ {wheel}\n")


def test_dry_run_stages_the_space_without_uploading(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        deploy_space, "release_wheel", lambda version: f"{WHEEL}#sha256={SHA}"
    )
    status = deploy_space.main(
        ["--version", "1.10.0", "--space", "owner/space", "--dry-run"]
    )
    out = capsys.readouterr().out
    assert status == 0
    for name in ("app.py", "viewer.py", "README.md", "requirements.txt"):
        assert name in out.splitlines()
    assert "examples/synthetic-record.json" in out.splitlines()
    assert f"llmtracefx @ {WHEEL}#sha256={SHA}" in out.splitlines()


def test_unresolved_wheel_stops_the_deploy(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(version: str) -> str:
        raise deploy_space.WheelError("expected one wheel, found 0")

    monkeypatch.setattr(deploy_space, "release_wheel", missing)
    status = deploy_space.main(
        ["--version", "1.10.0", "--space", "owner/space", "--dry-run"]
    )
    assert status == 1
    assert "found 0" in capsys.readouterr().err


def test_upload_requires_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert deploy_space.main(["--version", "1.10.0", "--space", "owner/space"]) == 1


@pytest.mark.parametrize("space", ["space", "owner/space/extra", "-x/y", "a b/c"])
def test_rejects_malformed_space_ids(space: str) -> None:
    with pytest.raises(SystemExit):
        deploy_space.main(["--version", "1.10.0", "--space", space, "--dry-run"])


class _Runtime:
    def __init__(self, stage: str, hardware: str | None, requested: str | None):
        self.stage = stage
        self.hardware = hardware
        self.requested_hardware = requested


class _FakeApi:
    def __init__(self, runtime: _Runtime, fail: bool = False) -> None:
        self.runtime = runtime
        self.fail = fail
        self.calls: list[str] = []

    def get_space_runtime(self, space: str) -> _Runtime:
        return self.runtime

    def restart_space(self, space: str) -> _Runtime:
        self.calls.append("restart")
        if self.fail:
            raise RuntimeError("403 Forbidden")
        self.runtime = _Runtime("BUILDING", None, self.runtime.requested_hardware)
        return self.runtime


def test_restarts_a_paused_space_without_changing_hardware() -> None:
    api = _FakeApi(_Runtime("PAUSED", None, "zero-a10g"))
    assert deploy_space.wake(api, "owner/space") == (
        "Hardware: zero-a10g, restarted, stage BUILDING"
    )
    assert api.calls == ["restart"]


@pytest.mark.parametrize("stage", ["BUILDING", "RUNNING"])
def test_leaves_a_live_space_alone(stage: str) -> None:
    api = _FakeApi(_Runtime(stage, "zero-a10g", "zero-a10g"))
    assert deploy_space.wake(api, "owner/space") == (
        f"Hardware: zero-a10g, stage {stage}"
    )
    assert api.calls == []


def test_failed_restart_is_a_warning_not_an_error() -> None:
    api = _FakeApi(_Runtime("PAUSED", None, None), fail=True)
    message = deploy_space.wake(api, "owner/space")
    assert message.startswith("::warning::owner/space is paused")
    assert "403 Forbidden" in message
