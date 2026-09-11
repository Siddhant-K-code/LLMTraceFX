from __future__ import annotations

import argparse
import ast
import hashlib
import json
import runpy
import subprocess
import sys
import zipfile
from dataclasses import dataclass
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
    assert "sys.path.extend((str(site_root), str(snapshot)))" in source
    assert "sys.path.insert(0, str(repo_root))" not in source


@dataclass
class _FakeDistribution:
    site_root: Path
    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}

    @property
    def files(self) -> list[Path]:
        package = self.name.replace("-", "_")
        return [
            Path(package) / "__init__.py",
            Path(f"{package}-1.dist-info") / "METADATA",
            Path(f"{package}-1.dist-info") / "RECORD",
        ]

    def locate_file(self, item: object) -> Path:
        return self.site_root / Path(str(item))


def _fake_distribution_tree(root: Path, name: str) -> _FakeDistribution:
    package = name.replace("-", "_")
    (root / package).mkdir(parents=True)
    (root / f"{package}-1.dist-info").mkdir()
    (root / package / "__init__.py").write_text("", encoding="ascii")
    (root / f"{package}-1.dist-info/METADATA").write_text(
        f"Name: {name}\nVersion: 1\n", encoding="ascii"
    )
    (root / f"{package}-1.dist-info/RECORD").write_text("", encoding="ascii")
    return _FakeDistribution(root, name, "1")


def test_bootstrap_rejects_unexpected_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _bootstrap_namespace()
    site = tmp_path / "site-packages"
    expected = _fake_distribution_tree(site, "expected")
    unexpected = _fake_distribution_tree(site, "unexpected")
    namespace["EXPECTED_DISTRIBUTIONS"].clear()
    namespace["EXPECTED_DISTRIBUTIONS"]["expected"] = "1"
    monkeypatch.setattr(
        namespace["importlib"].metadata,
        "distributions",
        lambda **_kwargs: [expected, unexpected],
    )
    with pytest.raises(namespace["BootstrapError"], match="closure is not exact"):
        namespace["_actual_identity"](site)


@pytest.mark.parametrize("relative", ["llmtracefx.py", "llmtracefx/__init__.py"])
def test_bootstrap_rejects_unowned_project_shadow(
    tmp_path: Path, relative: str
) -> None:
    namespace = _bootstrap_namespace()
    site = tmp_path / "site-packages"
    target = site / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("raise RuntimeError('shadow')\n", encoding="ascii")
    with pytest.raises(namespace["BootstrapError"], match="unowned regular file"):
        namespace["_scan_site_root"](site, owned_files=set(), generated_records=set())


def test_bootstrap_snapshot_matches_exact_commit_and_has_no_symlinks(
    tmp_path: Path,
) -> None:
    namespace = _bootstrap_namespace()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snapshot = namespace["_materialize_snapshot"](PROJECT_ROOT, commit, tmp_path)
    assert not any(path.is_symlink() for path in snapshot.rglob("*"))
    for package in ("llmtracefx", "vllm_kv_truth"):
        tracked = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", package],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert {
            path.relative_to(snapshot).as_posix()
            for path in (snapshot / package).rglob("*")
            if path.is_file()
        } == set(tracked)
        for relative in tracked:
            committed = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
            ).stdout
            assert (snapshot / relative).read_bytes() == committed
            assert (snapshot / relative).stat().st_mode & 0o222 == 0


def test_runtime_identity_record_hashing_matches_bootstrap_fixture() -> None:
    namespace = _bootstrap_namespace()
    expected = namespace["_identity_record_bytes"](
        "site-packages/example/data.bin", 17, "sha256:" + "a" * 64
    )
    assert (
        real_mlx._runtime_identity_record_bytes(
            "site-packages/example/data.bin", 17, "sha256:" + "a" * 64
        )
        == expected
    )


def test_modified_tracked_worktree_refuses_before_project_import(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    script = repository / "scripts/run-real-mlx-cache-audit-trusted.py"
    identity = (
        repository / "llmtracefx/cache_audit/data/"
        "apple-silicon-python313-mlx-lm-runtime-v1.json"
    )
    script.parent.mkdir(parents=True)
    identity.parent.mkdir(parents=True)
    script.write_bytes(BOOTSTRAP.read_bytes())
    identity.write_bytes(
        (
            PROJECT_ROOT / "llmtracefx/cache_audit/data/"
            "apple-silicon-python313-mlx-lm-runtime-v1.json"
        ).read_bytes()
    )
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    script.write_bytes(script.read_bytes() + b"\n")
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(script),
            "--bootstrap-self-test",
            "--expected-commit",
            commit,
        ],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout)["error"] == "tracked worktree must be clean"


@pytest.mark.parametrize(
    "command",
    [
        "aggregate",
        "calibrate",
        "compile",
        "preflight",
        "replicate",
        "run-all",
        "sandbox-probe",
        "sanitize",
        "verify",
    ],
)
def test_all_canonical_commands_require_bootstrap(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLMTRACEFX_TRUSTED_BOOTSTRAP", raising=False)
    with pytest.raises(real_mlx.RealMLXExperimentError, match="trusted Python -I -S"):
        real_mlx._run_cli(argparse.Namespace(command=command))
