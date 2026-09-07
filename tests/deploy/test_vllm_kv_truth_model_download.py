from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_kv_truth import model_download


def test_downloader_attestation_binds_exact_package_interface_and_source() -> None:
    module = SimpleNamespace(snapshot_download=lambda **kwargs: kwargs)
    receipt = model_download.downloader_attestation(
        version_fn=lambda package: model_download.DOWNLOADER_VERSION,
        import_module_fn=lambda module_name: module,
    )
    assert receipt == {
        "schema_version": "1",
        "package": "huggingface-hub",
        "version": model_download.DOWNLOADER_VERSION,
        "interface": "huggingface_hub.snapshot_download",
        "source": model_download.DOWNLOADER_SOURCE,
    }


def test_downloader_attestation_refuses_absent_package() -> None:
    def missing(_package: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    with pytest.raises(
        model_download.ModelDownloadRefusal,
        match="authorized downloader interface",
    ) as excinfo:
        model_download.downloader_attestation(version_fn=missing)
    assert excinfo.value.reason_code == "model_download_interface_missing"


def test_downloader_attestation_refuses_version_mismatch() -> None:
    with pytest.raises(
        model_download.ModelDownloadRefusal,
        match="version does not match",
    ) as excinfo:
        model_download.downloader_attestation(version_fn=lambda _package: "0.0.0")
    assert excinfo.value.reason_code == "model_download_version_mismatch"


def test_downloader_attestation_refuses_noncallable_interface() -> None:
    module = SimpleNamespace(snapshot_download=None)
    with pytest.raises(model_download.ModelDownloadRefusal) as excinfo:
        model_download.downloader_attestation(
            version_fn=lambda _package: model_download.DOWNLOADER_VERSION,
            import_module_fn=lambda _module_name: module,
        )
    assert excinfo.value.reason_code == "model_download_interface_missing"


def test_download_uses_exact_pins_allowlist_and_sanitized_scratch_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "model"
    scratch = tmp_path / "hf-scratch"
    destination.mkdir()
    scratch.mkdir()
    observed: dict[str, Any] = {}

    def snapshot_download(**kwargs: Any) -> None:
        observed["kwargs"] = kwargs
        observed["environment"] = dict(os.environ)

    module = SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-downloader")
    receipt = model_download.download_model(
        destination,
        scratch,
        version_fn=lambda _package: model_download.DOWNLOADER_VERSION,
        import_module_fn=lambda _module_name: module,
    )

    kwargs = observed["kwargs"]
    environment = observed["environment"]
    assert kwargs["repo_id"] == model_download.MODEL_ID
    assert kwargs["revision"] == model_download.MODEL_REVISION
    assert kwargs["token"] is False
    assert len(kwargs["allow_patterns"]) == 15
    assert environment["HF_HOME"] == str(scratch)
    assert environment["HF_HUB_CACHE"] == str(scratch / "hub")
    assert environment["HF_ASSETS_CACHE"] == str(scratch / "assets")
    assert environment["XDG_CACHE_HOME"] == str(scratch / "xdg")
    assert "HF_TOKEN" not in environment
    assert os.environ["HF_TOKEN"] == "must-not-reach-downloader"
    assert receipt["version"] == model_download.DOWNLOADER_VERSION


@pytest.mark.parametrize("python_minor", ["3.10", "3.11", "3.12", "3.13"])
def test_attestation_logic_is_python_minor_and_platform_independent(
    python_minor: str,
) -> None:
    calls: list[str] = []

    def version(package: str) -> str:
        calls.append(f"{python_minor}:{package}")
        return model_download.DOWNLOADER_VERSION

    module = SimpleNamespace(snapshot_download=lambda **_kwargs: None)
    receipt = model_download.downloader_attestation(
        version_fn=version,
        import_module_fn=lambda _module_name: module,
    )
    assert receipt["version"] == model_download.DOWNLOADER_VERSION
    assert calls == [f"{python_minor}:{model_download.DOWNLOADER_PACKAGE}"]


def test_failure_cli_emits_only_allowlisted_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(**_kwargs: Any) -> dict[str, str]:
        raise model_download.ModelDownloadRefusal("model_download_interface_missing")

    monkeypatch.setattr(model_download, "downloader_attestation", refuse)
    assert model_download.main(["attest"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "LLMTRACEFX_REASON=model_download_interface_missing\n"
