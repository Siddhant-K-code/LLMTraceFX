from __future__ import annotations

import ast
import hashlib
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.cache_audit import real_mlx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = PROJECT_ROOT / "scripts" / "run-real-mlx-cache-audit-trusted.py"


def _bootstrap_namespace() -> dict[str, Any]:
    return runpy.run_path(str(BOOTSTRAP), run_name="trusted_bootstrap_test")


def test_bootstrap_imports_only_python_standard_library() -> None:
    tree = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported <= sys.stdlib_module_names | {"__future__"}
    assert "llmtracefx" not in imported
    namespace = _bootstrap_namespace()
    identity = (
        PROJECT_ROOT / "llmtracefx/cache_audit/data/"
        "apple-silicon-python313-mlx-lm-runtime-v1.json"
    ).read_bytes()
    assert namespace["EXPECTED_IDENTITY_SHA256"] == (
        "sha256:" + hashlib.sha256(identity).hexdigest()
    )


@pytest.mark.parametrize(
    "relative",
    [
        Path("unsafe.pth"),
        Path("sitecustomize.py"),
        Path("usercustomize.py"),
        Path("__pycache__"),
        Path("package/compiled.pyc"),
        Path("package/optimized.pyo"),
    ],
)
def test_bootstrap_rejects_all_startup_and_bytecode_artifacts(
    tmp_path: Path, relative: Path
) -> None:
    namespace = _bootstrap_namespace()
    site_root = tmp_path / "site-packages"
    target = site_root / relative
    if relative.name == "__pycache__":
        target.mkdir(parents=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"unsafe")
    with pytest.raises(namespace["BootstrapError"], match="startup hook|bytecode"):
        namespace["_scan_site_root"](site_root)


def test_portable_identity_ignores_record_and_out_of_root_scripts(
    tmp_path: Path,
) -> None:
    namespace = _bootstrap_namespace()

    class FakeDistribution:
        version = "1.0"
        metadata = {"Name": "fixture"}
        files = [
            Path("fixture/__init__.py"),
            Path("fixture-1.0.dist-info/METADATA"),
            Path("fixture-1.0.dist-info/RECORD"),
            Path("../../../bin/fixture"),
        ]

        def __init__(self, site_root: Path) -> None:
            self.site_root = site_root

        def locate_file(self, item: object) -> Path:
            return self.site_root / Path(str(item))

    identities = []
    for name in ("first-env", "second-env-with-a-different-path"):
        root = tmp_path / name / "lib" / "python3.13" / "site-packages"
        (root / "fixture").mkdir(parents=True)
        (root / "fixture-1.0.dist-info").mkdir()
        (root / "fixture/__init__.py").write_text("VALUE = 1\n", encoding="ascii")
        (root / "fixture-1.0.dist-info/METADATA").write_text(
            "Name: fixture\nVersion: 1.0\n", encoding="ascii"
        )
        (root / "fixture-1.0.dist-info/RECORD").write_text(
            f"absolute-environment={root}\n", encoding="ascii"
        )
        identities.append(
            namespace["_distribution_identity"](
                FakeDistribution(root), "fixture", "1.0", root
            )
        )
    assert identities[0] == identities[1]
    assert identities[0]["file_count"] == 2


def test_built_wheel_contains_trusted_bootstrap(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    wheel = next(tmp_path.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert any(
            name.endswith("/scripts/run-real-mlx-cache-audit-trusted.py")
            or name.endswith(".data/scripts/run-real-mlx-cache-audit-trusted.py")
            for name in archive.namelist()
        )


def test_bootstrap_rejection_is_safe_json_and_imports_no_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(BOOTSTRAP), "--bootstrap-self-test"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert set(payload) in (
        {"error", "ok"},
        {"bootstrap_verified", "distribution_count"},
    )


def test_canonical_commands_refuse_direct_console_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLMTRACEFX_TRUSTED_BOOTSTRAP", raising=False)
    args = real_mlx._parser().parse_args(
        [
            "run-all",
            "--workload",
            "workload.json",
            "--model-dir",
            "model",
            "--output-workspace",
            "run",
            "--expected-commit",
            "a" * 40,
        ]
    )
    with pytest.raises(real_mlx.RealMLXExperimentError, match="trusted Python -I -S"):
        real_mlx._run_cli(args)


def test_bootstrap_prefers_verified_site_packages_before_repository() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "sys.path.extend((str(site_root), str(repo_root)))" in source
    assert "sys.path.insert(0, str(repo_root))" not in source
