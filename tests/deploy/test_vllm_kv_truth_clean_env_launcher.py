"""Process tests for the native-env, pre-import Python bootstrap."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_SOURCE = PROJECT_ROOT / "scripts" / "run-vllm-kv-truth-clean-env.py"
SAFE_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
CONTAMINATED_ENVIRONMENT = {
    "COPILOT_TRAMPOLINE_TOKEN": "copilot-secret",
    "GH_TOKEN": "github-secret",
    "SSH_AUTH_SOCK": "/host-agent.sock",
    "PYTHONPATH": "/hostile-imports",
    "PYTHONHOME": "/hostile-python",
    "DOCKER_HOST": "tcp://127.0.0.1:2375",
    "DOCKER_CONFIG": "/host-docker-config",
    "GIT_ASKPASS": "/host-askpass",
    "GIT_CONFIG_GLOBAL": "/host-git-config",
    "HTTP_PROXY": "http://127.0.0.1:8080",
    "HTTPS_PROXY": "http://127.0.0.1:8080",
    "BASH_ENV": "/host-bash-env",
    "ENV": "/host-shell-env",
    "SHELLOPTS": "braceexpand:hashall:interactive-comments:xtrace",
    "PS4": "$(touch should-never-run)",
    "ARBITRARY_SECRET": "not-for-the-child",
}
PLATFORM_ENVIRONMENT_NAMES = {"__CF_USER_TEXT_ENCODING"}

FAKE_CLI = r"""
import json
import os
import signal
import sys
import time
from pathlib import Path

EXPECTED = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}

def bootstrap_dispatch(argv):
    args = list(sys.argv[1:] if argv is None else argv)
    unexpected = set(os.environ) - set(EXPECTED) - {"__CF_USER_TEXT_ENCODING"}
    if unexpected or any(os.environ.get(k) != v for k, v in EXPECTED.items()):
        return 91
    if args == ["preflight"]:
        print("clean environment preflight: ok")
        return 0
    output = Path(args[args.index("--output-dir") + 1])
    auth = Path(args[args.index("--authorization") + 1])
    authorization = json.loads(auth.read_text(encoding="utf-8"))
    (output / "started").write_text("yes", encoding="utf-8")
    def finish(signum, _frame):
        (output / "cleanup").write_text(f"signal={signum}", encoding="utf-8")
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, finish)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, finish)
    if authorization.get("wait_for_signal"):
        while True:
            time.sleep(0.02)
    (output / "observation.json").write_text(
        json.dumps({"environment": dict(os.environ), "lifecycle": "complete"}),
        encoding="utf-8",
    )
    (output / "cleanup").write_text("complete", encoding="utf-8")
    return int(authorization.get("exit_code", 0))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_hash(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _write_wheel(path: Path, bootstrap: bytes, cli_source: str = FAKE_CLI) -> None:
    files = {
        "llmtracefx/__init__.py": b"",
        "vllm_kv_truth/__init__.py": b"",
        "vllm_kv_truth/cli.py": cli_source.encode(),
        ("llmtracefx-1.0.0.data/scripts/" "run-vllm-kv-truth-clean-env.py"): bootstrap,
        "llmtracefx-1.0.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: llmtracefx\nVersion: 1.0.0\n"
        ),
        "llmtracefx-1.0.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
    }
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name, payload in files.items():
        writer.writerow((name, _record_hash(payload), len(payload)))
    record_name = "llmtracefx-1.0.0.dist-info/RECORD"
    writer.writerow((record_name, "", ""))
    files[record_name] = output.getvalue().encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def _rewrite_wheel(path: Path, mutate: Callable[[dict[str, bytes]], None]) -> None:
    with zipfile.ZipFile(path) as archive:
        files = {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
            if not info.is_dir()
        }
    mutate(files)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def _install_fake_environment(
    tmp_path: Path, *, interpreter_target: Path | None = None
) -> tuple[Path, Path, Path]:
    root = tmp_path / "dedicated-venv"
    bin_dir = root / "bin"
    site_packages = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    bin_dir.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    interpreter = bin_dir / "python"
    interpreter.symlink_to(interpreter_target or Path(sys.executable).resolve())
    bootstrap = bin_dir / "run-vllm-kv-truth-clean-env.py"
    source = BOOTSTRAP_SOURCE.read_bytes()
    bootstrap_body = source.split(b"\n", 1)[1]
    bootstrap.write_bytes(f"#!{interpreter}\n".encode() + bootstrap_body)
    bootstrap.chmod(0o755)
    console_script = bin_dir / "llmtracefx-vllm-kv-truth"
    console_script.write_text("#!/bin/sh\nexit 99\n")
    console_script.chmod(0o755)
    (site_packages / "llmtracefx").mkdir()
    (site_packages / "llmtracefx" / "__init__.py").write_bytes(b"")
    (site_packages / "vllm_kv_truth").mkdir()
    (site_packages / "vllm_kv_truth" / "__init__.py").write_bytes(b"")
    (site_packages / "vllm_kv_truth" / "cli.py").write_text(FAKE_CLI)
    dist_info = site_packages / "llmtracefx-1.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_bytes(
        b"Metadata-Version: 2.1\nName: llmtracefx\nVersion: 1.0.0\n"
    )
    (dist_info / "WHEEL").write_bytes(
        b"Wheel-Version: 1.0\nGenerator: tests\nRoot-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    )
    wheel = tmp_path / "llmtracefx-1.0.0-py3-none-any.whl"
    _write_wheel(wheel, b"#!python\n" + bootstrap_body)
    return interpreter, bootstrap, wheel


def _native_command(interpreter: Path, bootstrap: Path, *args: str) -> list[str]:
    return [
        "/usr/bin/env",
        "-i",
        *[f"{name}={value}" for name, value in SAFE_ENVIRONMENT.items()],
        str(interpreter),
        "-I",
        "-S",
        "-B",
        str(bootstrap),
        *args,
    ]


def _record_trust(
    tmp_path: Path, interpreter: Path, bootstrap: Path, wheel: Path
) -> tuple[Path, str]:
    manifest = tmp_path / "trusted-launch-manifest.json"
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "record-trust",
            "--wheel",
            str(wheel),
            "--output",
            str(manifest),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600
    return manifest, _sha256(manifest)


def _trusted_args(wheel: Path, manifest: Path, manifest_sha256: str) -> list[str]:
    return [
        "--wheel",
        str(wheel),
        "--trusted-manifest",
        str(manifest),
        "--trusted-manifest-sha256",
        manifest_sha256,
    ]


def _protected_inputs(
    tmp_path: Path, authorization: dict[str, object] | None = None
) -> tuple[Path, Path, Path]:
    private = tmp_path / "private"
    private.mkdir()
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    config = private / "execution-config.json"
    config.write_text(json.dumps({"local_evidence_dir": str(output)}))
    config.chmod(0o600)
    auth = private / "authorization.json"
    auth.write_text(json.dumps(authorization or {}))
    auth.chmod(0o600)
    return config, auth, output


def test_real_setuptools_wheel_install_and_public_cli_split(tmp_path: Path) -> None:
    build_venv = tmp_path / "exact-build-venv"
    created_build_venv = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(build_venv)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created_build_venv.returncode == 0, created_build_venv.stderr
    build_python = build_venv / "bin" / "python"
    installed_build_tools = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(build_python),
            "setuptools==84.0.0",
            "wheel==0.48.0",
            "packaging==26.3",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed_build_tools.returncode == 0, installed_build_tools.stderr
    versions = subprocess.run(
        [
            str(build_python),
            "-c",
            (
                "import packaging,setuptools,wheel;"
                "print(setuptools.__version__,wheel.__version__,packaging.__version__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert versions.returncode == 0, versions.stderr
    assert versions.stdout.strip() == "84.0.0 0.48.0 26.3"

    dist = tmp_path / "dist"
    built = subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--no-build-isolation",
            "--python",
            str(build_python),
            "--out-dir",
            str(dist),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stderr
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    venv = tmp_path / "real-wheel-venv"
    created = subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    installed = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin" / "python"),
            "--no-deps",
            str(wheel),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stderr

    interpreter = venv / "bin" / "python"
    bootstrap = venv / "bin" / "run-vllm-kv-truth-clean-env.py"
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    preflight = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(wheel, manifest, digest),
        ),
        env={**os.environ, **CONTAMINATED_ENVIRONMENT},
        capture_output=True,
        text=True,
        check=False,
    )
    assert preflight.returncode == 0, preflight.stderr
    assert preflight.stdout == "clean environment preflight: ok\n"

    public_cli = venv / "bin" / "llmtracefx-vllm-kv-truth"
    private_read_marker = tmp_path / "private-input-was-read"
    for command in ("run", "preflight", "preflight-clean-environment"):
        refused = subprocess.run(
            [
                "/usr/bin/env",
                "-i",
                *[f"{name}={value}" for name, value in SAFE_ENVIRONMENT.items()],
                str(public_cli),
                command,
                "--execution-config",
                str(private_read_marker),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert refused.returncode == 2
    assert not private_read_marker.exists()


def test_runtime_inventory_detects_executable_runtime_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location(
        "clean_bootstrap_runtime_test", BOOTSTRAP_SOURCE
    )
    assert spec is not None and spec.loader is not None
    bootstrap_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap_module)

    base = tmp_path / "python-runtime"
    stdlib = base / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    dynload = stdlib / "lib-dynload"
    cache = stdlib / "__pycache__"
    site_packages = stdlib / "site-packages"
    for directory in (dynload, cache, site_packages):
        directory.mkdir(parents=True, exist_ok=True)
    (stdlib / "os.py").write_text("RUNTIME = 'trusted'\n")
    extension = dynload / "_hashlib.test-extension"
    extension.write_bytes(b"trusted-native-extension")
    bytecode = cache / "os.test.pyc"
    bytecode.write_bytes(b"trusted-bytecode")
    ignored_site_package = site_packages / "ambient.py"
    ignored_site_package.write_text("AMBIENT = 'one'\n")
    companion = (
        base
        / "lib"
        / (f"libpython{sys.version_info.major}.{sys.version_info.minor}.test")
    )
    companion.write_bytes(b"trusted-runtime-library")

    monkeypatch.setattr(
        bootstrap_module,
        "_runtime_paths",
        lambda: (base, (stdlib,), (companion,)),
    )
    first = bootstrap_module._runtime_inventory()

    ignored_site_package.write_text("AMBIENT = 'two'\n")
    assert bootstrap_module._runtime_inventory() == first

    extension.write_bytes(b"tampered-native-extension")
    native_tamper = bootstrap_module._runtime_inventory()
    assert native_tamper[2] != first[2]
    assert native_tamper[3] == first[3]

    extension.write_bytes(b"trusted-native-extension")
    bytecode.write_bytes(b"tampered-bytecode")
    bytecode_tamper = bootstrap_module._runtime_inventory()
    assert bytecode_tamper[2] != first[2]
    assert bytecode_tamper[3] == first[3]


def test_native_env_strips_shell_hooks_and_ambient_state(tmp_path: Path) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    config, auth, output = _protected_inputs(tmp_path)
    command = _native_command(
        interpreter,
        bootstrap,
        "run",
        *_trusted_args(wheel, manifest, digest),
        "--execution-config",
        str(config),
        "--authorization",
        str(auth),
        "--output-dir",
        str(output),
    )
    completed = subprocess.run(
        command,
        env={**os.environ, **CONTAMINATED_ENVIRONMENT},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observation = json.loads((output / "observation.json").read_text())
    child_environment = observation["environment"]
    assert {
        name: child_environment[name] for name in SAFE_ENVIRONMENT
    } == SAFE_ENVIRONMENT
    assert set(child_environment) <= set(SAFE_ENVIRONMENT) | PLATFORM_ENVIRONMENT_NAMES
    assert set(child_environment).isdisjoint(CONTAMINATED_ENVIRONMENT)
    assert observation["lifecycle"] == "complete"
    assert (output / "cleanup").read_text() == "complete"
    assert not (tmp_path / "should-never-run").exists()


def test_preflight_requires_native_clean_environment(tmp_path: Path) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    args = [
        str(interpreter),
        "-I",
        "-S",
        str(bootstrap),
        "preflight",
        *_trusted_args(wheel, manifest, digest),
    ]
    refused = subprocess.run(
        args,
        env={**os.environ, "GH_TOKEN": "secret"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "fixed clean environment" in refused.stderr

    clean = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(wheel, manifest, digest),
        ),
        env={**os.environ, **CONTAMINATED_ENVIRONMENT},
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout == "clean environment preflight: ok\n"


def test_trust_accepts_long_path_installer_trampoline(tmp_path: Path) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    body = bootstrap.read_bytes().split(b"\n", 1)[1]
    bootstrap.write_bytes(
        b"#!/bin/sh\n"
        + f"'''exec' '{interpreter}' \"$0\" \"$@\"\n".encode()
        + b"' '''\n"
        + body
    )
    bootstrap.chmod(0o755)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(wheel, manifest, digest),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_trust_rejects_installer_shebang_for_another_interpreter(
    tmp_path: Path,
) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    body = bootstrap.read_bytes().split(b"\n", 1)[1]
    bootstrap.write_bytes(b"#!/usr/bin/python3\n" + body)
    bootstrap.chmod(0o755)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "record-trust",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "manifest.json"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "invalid PEP 427 shebang rewrite" in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("wheel", "wheel does not match"),
        ("bootstrap", "bootstrap does not match"),
        ("package", "installed package contents do not match"),
        ("dependency", "installed environment does not match"),
        ("runtime-manifest", "Python base runtime does not match"),
        ("manifest", "externally recorded SHA-256"),
        ("stale-wheel", "launch paths do not match"),
    ],
)
def test_trust_root_rejects_tampering(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    selected_wheel = wheel
    if mutation == "wheel":
        with wheel.open("ab") as handle:
            handle.write(b"tampered")
    elif mutation == "bootstrap":
        with bootstrap.open("a") as handle:
            handle.write("\n# tampered after authorization\n")
    elif mutation == "package":
        console_script = interpreter.parent / "llmtracefx-vllm-kv-truth"
        original_console_digest = _sha256(console_script)
        package = (
            interpreter.parent.parent
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "vllm_kv_truth"
            / "cli.py"
        )
        package.write_text(FAKE_CLI + "\n# changed with same console entry point\n")
        assert _sha256(console_script) == original_console_digest
    elif mutation == "dependency":
        dependency = (
            interpreter.parent.parent
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
            / "dependency.py"
        )
        dependency.write_text("# dependency added after trust recording\n")
    elif mutation == "runtime-manifest":
        payload = json.loads(manifest.read_text())
        payload["python_runtime_sha256"] = "0" * 64
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        manifest.chmod(0o600)
        digest = _sha256(manifest)
    elif mutation == "manifest":
        with manifest.open("a") as handle:
            handle.write(" ")
    elif mutation == "stale-wheel":
        selected_wheel = tmp_path / "stale.whl"
        shutil.copyfile(wheel, selected_wheel)

    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(selected_wheel, manifest, digest),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("unrecorded", "contents do not exactly match RECORD"),
        ("unsigned", "payload entries must have hashes and sizes"),
    ],
)
def test_record_trust_rejects_incomplete_wheel_record(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)

    def mutate(files: dict[str, bytes]) -> None:
        if mutation == "unrecorded":
            files["unrecorded.py"] = b"raise RuntimeError('unrecorded')\n"
            return
        record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"))))
        rows[0][1] = ""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
        files[record_name] = output.getvalue().encode()

    _rewrite_wheel(wheel, mutate)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "record-trust",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "manifest.json"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr


def test_record_trust_rejects_validly_recorded_unexpected_module(
    tmp_path: Path,
) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)

    def mutate(files: dict[str, bytes]) -> None:
        record_name = next(name for name in files if name.endswith(".dist-info/RECORD"))
        payload = b"raise RuntimeError('unexpected')\n"
        rows = list(csv.reader(io.StringIO(files[record_name].decode("utf-8"))))
        rows.insert(-1, ["subprocess.py", _record_hash(payload), str(len(payload))])
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerows(rows)
        files[record_name] = output.getvalue().encode()
        files["subprocess.py"] = payload

    _rewrite_wheel(wheel, mutate)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "record-trust",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "manifest.json"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "unexpected top-level path" in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("relative-config", "execution config must be an absolute path"),
        ("config-mode", "execution config must have mode 0600"),
        ("auth-mode", "authorization must have mode 0600"),
        ("output-mode", "mode 0700"),
        ("nonempty-output", "must be empty"),
        ("symlink-config", "regular non-symlink file"),
        ("symlink-output", "unsafe directory component"),
    ],
)
def test_bootstrap_rejects_unsafe_run_inputs(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    config, auth, output = _protected_inputs(tmp_path)
    config_arg, output_arg = str(config), str(output)
    if mutation == "relative-config":
        config_arg = "relative.json"
    elif mutation == "config-mode":
        config.chmod(0o644)
    elif mutation == "auth-mode":
        auth.chmod(0o640)
    elif mutation == "output-mode":
        output.chmod(0o755)
    elif mutation == "nonempty-output":
        (output / "existing").write_text("x")
    elif mutation == "symlink-config":
        link = config.with_name("config-link")
        link.symlink_to(config)
        config_arg = str(link)
    elif mutation == "symlink-output":
        link = output.with_name("output-link")
        link.symlink_to(output, target_is_directory=True)
        output_arg = str(link)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "run",
            *_trusted_args(wheel, manifest, digest),
            "--execution-config",
            config_arg,
            "--authorization",
            str(auth),
            "--output-dir",
            output_arg,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert expected in completed.stderr
    assert not (output / "started").exists()


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_signal_exit_semantics_follow_cleanup(tmp_path: Path, signum: int) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    config, auth, output = _protected_inputs(
        tmp_path, authorization={"wait_for_signal": True}
    )
    process = subprocess.Popen(
        _native_command(
            interpreter,
            bootstrap,
            "run",
            *_trusted_args(wheel, manifest, digest),
            "--execution-config",
            str(config),
            "--authorization",
            str(auth),
            "--output-dir",
            str(output),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not (output / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (output / "started").exists()
    process.send_signal(signum)
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 128 + signum, stderr
    assert (output / "cleanup").read_text() == f"signal={signum}"


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_real_cli_process_preserves_signal_status_after_fake_teardown(
    tmp_path: Path, signum: int
) -> None:
    harness = tmp_path / "real-cli-signal-harness.py"
    harness.write_text(f"""
import importlib.util
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
sys.path.insert(0, str(PROJECT_ROOT))

from vllm_kv_truth import cli, lifecycle

spec = importlib.util.spec_from_file_location(
    "kv_truth_lifecycle_test_support",
    PROJECT_ROOT / "tests/deploy/test_vllm_kv_truth_lifecycle.py",
)
support = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = support
spec.loader.exec_module(support)

root = Path(sys.argv[1])
started = root / "real-cli-started"
cleanup = root / "real-cli-cleanup"
config = support._config(root)
authorization = support._authorization()

class BlockingRunner(support.FakeCommandRunner):
    def run(self, argv, *, description, timeout, input_text=None):
        if description == "stage_preflight":
            started.write_text("yes", encoding="utf-8")
            while True:
                time.sleep(0.05)
        result = super().run(
            argv,
            description=description,
            timeout=timeout,
            input_text=input_text,
        )
        if description == "stage_teardown_cleanup":
            cleanup.write_text("complete", encoding="utf-8")
        return result

runner = BlockingRunner()
cli.ProtectedExecutionConfig.load = classmethod(lambda cls, path: config)
cli.RunAuthorization.read = classmethod(lambda cls, path: authorization)
cli.SubprocessCommandRunner = lambda: runner

def orchestrator_factory(*, config, authorization, runner):
    return lifecycle.RemoteOrchestrator(
        config=config,
        authorization=authorization,
        runner=runner,
        now_fn=lambda: support.BILLING_STARTED_AT,
    )

cli.RemoteOrchestrator = orchestrator_factory
raise SystemExit(
    cli.bootstrap_dispatch(
        [
            "run",
            "--execution-config",
            str(root / "config.json"),
            "--authorization",
            str(root / "authorization.json"),
            "--output-dir",
            str(config.local_evidence_dir),
        ]
    )
)
""")
    process = subprocess.Popen(
        [sys.executable, "-I", str(harness), str(tmp_path)],
        env=SAFE_ENVIRONMENT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started = tmp_path / "real-cli-started"
    deadline = time.monotonic() + 10
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert started.exists()
    process.send_signal(signum)
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 128 + signum, (stdout, stderr)
    assert (tmp_path / "real-cli-cleanup").read_text() == "complete"
    assert "SAFE TO TERMINATE INSTANCE NOW" in stdout
    assert "run_interrupted" in stderr


def test_interpreter_directory_alias_symlink_chain_is_accepted(tmp_path: Path) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    root = interpreter.parent.parent
    target_directory = tmp_path / "uv-python-target"
    target_bin = target_directory / "bin"
    target_bin.mkdir(parents=True)
    (target_bin / "python").symlink_to(Path(sys.executable).resolve())
    alias = tmp_path / "cpython-version-alias"
    alias.symlink_to(target_directory, target_is_directory=True)
    interpreter.unlink()
    interpreter.symlink_to(alias / "bin" / "python")
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(wheel, manifest, digest),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert root.is_dir()
    assert completed.returncode == 0, completed.stderr


def test_writable_interpreter_link_parent_is_rejected(tmp_path: Path) -> None:
    interpreter, bootstrap, wheel = _install_fake_environment(tmp_path)
    unsafe = tmp_path / "unsafe-interpreter"
    unsafe.mkdir()
    unsafe.chmod(0o777)
    (unsafe / "python").symlink_to(Path(sys.executable).resolve())
    interpreter.unlink()
    interpreter.symlink_to(unsafe / "python")
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "record-trust",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "manifest.json"),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "group- or world-writable" in completed.stderr


def test_root_owned_system_interpreter_target_is_accepted(tmp_path: Path) -> None:
    candidates = [
        path
        for path in (Path("/usr/bin/python3"), Path("/bin/python3"))
        if path.exists() and path.resolve().stat().st_uid == 0
    ]
    if not candidates:
        pytest.skip("no root-owned system Python is available")
    version = subprocess.check_output(
        [
            str(candidates[0]),
            "-c",
            "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version != f"{sys.version_info.major}.{sys.version_info.minor}":
        pytest.skip("root-owned Python minor does not match the test wheel layout")
    interpreter, bootstrap, wheel = _install_fake_environment(
        tmp_path, interpreter_target=candidates[0]
    )
    manifest, digest = _record_trust(tmp_path, interpreter, bootstrap, wheel)
    completed = subprocess.run(
        _native_command(
            interpreter,
            bootstrap,
            "preflight",
            *_trusted_args(wheel, manifest, digest),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
