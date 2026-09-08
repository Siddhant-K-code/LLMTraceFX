"""Process-level tests for the pre-Python clean-environment launcher."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "scripts" / "run-vllm-kv-truth-clean-env.sh"
CONTAMINATED_ENVIRONMENT = {
    "COPILOT_TRAMPOLINE_TOKEN": "copilot-secret",
    "GH_TOKEN": "github-secret",
    "SSH_AUTH_SOCK": "/tmp/host-agent.sock",
    "PYTHONPATH": "/tmp/hostile-imports",
    "DOCKER_HOST": "tcp://127.0.0.1:2375",
    "DOCKER_CONFIG": "/tmp/host-docker-config",
    "GIT_ASKPASS": "/tmp/host-askpass",
    "GIT_CONFIG_GLOBAL": "/tmp/host-git-config",
    "HOME": "/tmp/host-home",
    "HTTP_PROXY": "http://127.0.0.1:8080",
    "HTTPS_PROXY": "http://127.0.0.1:8080",
    "PIP_CONFIG_FILE": "/tmp/host-pip-config",
    "VIRTUAL_ENV": "/tmp/host-venv",
    "ARBITRARY_SECRET": "not-for-the-child",
}
EXPECTED_CHILD_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}
PLATFORM_SYNTHESIZED_ENVIRONMENT_NAMES = {"__CF_USER_TEXT_ENCODING"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_cli(tmp_path: Path, *, shebang: str | None = None) -> Path:
    bin_dir = tmp_path / "isolated-install" / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python3"
    interpreter.symlink_to(Path(sys.executable).resolve())
    cli = bin_dir / "llmtracefx-vllm-kv-truth"
    first_line = shebang or f"#!{interpreter}"
    cli.write_text(
        first_line + """
import json
import os
import signal
import sys
import time
from pathlib import Path

EXPECTED_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}

if sys.argv[1:] == ["preflight-clean-environment"]:
    unexpected = set(os.environ) - set(EXPECTED_ENV) - {"__CF_USER_TEXT_ENCODING"}
    changed = {
        name for name, value in EXPECTED_ENV.items() if os.environ.get(name) != value
    }
    if unexpected or changed:
        raise SystemExit(91)
    print("clean environment preflight: ok")
    raise SystemExit(0)

args = sys.argv[1:]
expected_flags = [
    "run",
    "--execution-config",
    args[2],
    "--authorization",
    args[4],
    "--output-dir",
    args[6],
]
if args != expected_flags:
    raise SystemExit(92)

config_path = Path(args[2])
authorization_path = Path(args[4])
output_dir = Path(args[6])
config = json.loads(config_path.read_text(encoding="utf-8"))
authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
if config["local_evidence_dir"] != str(output_dir):
    raise SystemExit(93)

(output_dir / "started").write_text("yes", encoding="utf-8")

def finish(signum, _frame):
    (output_dir / "cleanup").write_text(f"signal={signum}", encoding="utf-8")
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, finish)
signal.signal(signal.SIGINT, finish)

if authorization.get("wait_for_signal"):
    while True:
        time.sleep(0.05)

(output_dir / "lifecycle.json").write_text(
    json.dumps(
        {
            "environment": dict(os.environ),
            "stages": [
                "preflight",
                "identity",
                "image",
                "model",
                "canary",
                "pairs",
                "eviction",
                "evidence",
                "teardown",
            ],
        },
        sort_keys=True,
    ),
    encoding="utf-8",
)
(output_dir / "cleanup").write_text("complete", encoding="utf-8")
raise SystemExit(int(authorization.get("exit_code", 0)))
""",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    return cli


def _protected_inputs(
    tmp_path: Path, *, authorization: dict[str, object] | None = None
) -> tuple[Path, Path, Path]:
    private_dir = tmp_path / "private inputs"
    private_dir.mkdir()
    output_dir = tmp_path / "evidence output"
    output_dir.mkdir(mode=0o700)
    config = private_dir / "execution;config.json"
    config.write_text(
        json.dumps({"local_evidence_dir": str(output_dir)}), encoding="utf-8"
    )
    config.chmod(0o600)
    auth = private_dir / "authorization file.json"
    auth.write_text(json.dumps(authorization or {}), encoding="utf-8")
    auth.chmod(0o600)
    return config, auth, output_dir


def _run_args(cli: Path, config: Path, auth: Path, output: Path) -> list[str]:
    return [
        str(LAUNCHER),
        "run",
        "--cli",
        str(cli),
        "--cli-sha256",
        _sha256(cli),
        "--execution-config",
        str(config),
        "--authorization",
        str(auth),
        "--output-dir",
        str(output),
    ]


def _contaminated_environment(tmp_path: Path) -> dict[str, str]:
    malicious_bin = tmp_path / "malicious-bin"
    malicious_bin.mkdir(exist_ok=True)
    marker = tmp_path / "path-command-ran"
    for command in ("env", "sha256sum", "shasum", "stat", "sed"):
        executable = malicious_bin / command
        executable.write_text(f"#!/bin/sh\n: > '{marker}'\nexit 99\n", encoding="utf-8")
        executable.chmod(0o755)
    return {
        **os.environ,
        **CONTAMINATED_ENVIRONMENT,
        "PATH": str(malicious_bin),
        "MALICIOUS_PATH_MARKER": str(marker),
    }


def test_clean_launcher_strips_ambient_state_and_completes_fake_lifecycle(
    tmp_path: Path,
) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(tmp_path)
    completed = subprocess.run(
        _run_args(cli, config, auth, output),
        env=_contaminated_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lifecycle = json.loads((output / "lifecycle.json").read_text(encoding="utf-8"))
    child_environment = lifecycle["environment"]
    assert {
        name: child_environment[name] for name in EXPECTED_CHILD_ENVIRONMENT
    } == EXPECTED_CHILD_ENVIRONMENT
    assert (
        set(child_environment)
        <= set(EXPECTED_CHILD_ENVIRONMENT) | PLATFORM_SYNTHESIZED_ENVIRONMENT_NAMES
    )
    assert set(child_environment).isdisjoint(CONTAMINATED_ENVIRONMENT)
    assert lifecycle["stages"][-1] == "teardown"
    assert (output / "cleanup").read_text(encoding="utf-8") == "complete"
    assert not (tmp_path / "path-command-ran").exists()


def test_actual_cli_refuses_contamination_but_launcher_preflight_succeeds(
    tmp_path: Path,
) -> None:
    cli = Path(sys.executable).parent / "llmtracefx-vllm-kv-truth"
    assert cli.is_file(), "the test environment must install the project console script"
    contaminated = {**os.environ, **CONTAMINATED_ENVIRONMENT}
    direct = subprocess.run(
        [str(cli), "preflight-clean-environment"],
        env=contaminated,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == 1
    for name in ("COPILOT_TRAMPOLINE_TOKEN", "GH_TOKEN", "SSH_AUTH_SOCK"):
        assert name in direct.stderr
    assert "run-vllm-kv-truth-clean-env.sh" in direct.stderr
    assert "instead of unsetting individual variables" in direct.stderr

    clean = subprocess.run(
        [
            str(LAUNCHER),
            "preflight",
            "--cli",
            str(cli),
            "--cli-sha256",
            _sha256(cli),
        ],
        env=contaminated,
        capture_output=True,
        text=True,
        check=False,
    )
    assert clean.returncode == 0, clean.stderr
    assert clean.stdout == "clean environment preflight: ok\n"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("relative-config", "execution config must be an absolute path"),
        ("config-mode", "execution config must have mode 0600"),
        ("auth-mode", "authorization must have mode 0600"),
        ("nonempty-output", "output directory must be empty"),
        ("output-mode", "output directory must have mode 0700"),
        ("missing-auth", "authorization must be a regular file"),
        ("wrong-hash", "does not match the expected SHA-256"),
        ("malformed-hash", "exactly 64 lowercase hexadecimal"),
        ("wrong-name", "wrong installed name"),
        ("nonexecutable-cli", "CLI executable is not executable"),
        ("symlink-cli", "must not contain symlink components"),
        ("symlink-config", "must not contain symlink components"),
        ("symlink-output", "must not contain symlink components"),
    ],
)
def test_launcher_refuses_unsafe_inputs(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(tmp_path)
    args = _run_args(cli, config, auth, output)
    if mutation == "relative-config":
        args[7] = "relative.json"
    elif mutation == "config-mode":
        config.chmod(0o644)
    elif mutation == "auth-mode":
        auth.chmod(0o640)
    elif mutation == "nonempty-output":
        (output / "existing").write_text("x", encoding="utf-8")
    elif mutation == "output-mode":
        output.chmod(0o755)
    elif mutation == "missing-auth":
        auth.unlink()
    elif mutation == "wrong-hash":
        args[5] = "0" * 64
    elif mutation == "malformed-hash":
        args[5] = "ABC"
    elif mutation == "wrong-name":
        renamed = cli.with_name("other-cli")
        cli.rename(renamed)
        args[3] = str(renamed)
        args[5] = _sha256(renamed)
    elif mutation == "nonexecutable-cli":
        cli.chmod(0o644)
    elif mutation == "symlink-cli":
        link = cli.with_name("linked-cli")
        link.symlink_to(cli)
        args[3] = str(link)
        args[5] = _sha256(cli)
    elif mutation == "symlink-config":
        link = config.with_name("linked-config.json")
        link.symlink_to(config)
        args[7] = str(link)
    elif mutation == "symlink-output":
        link = output.with_name("linked-output")
        link.symlink_to(output, target_is_directory=True)
        args[11] = str(link)
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert completed.returncode == 2
    assert expected_error in completed.stderr
    assert not (output / "started").exists()


def test_launcher_rejects_ambiguous_paths_and_argument_injection(
    tmp_path: Path,
) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(tmp_path)
    ambiguous = _run_args(cli, config, auth, output)
    ambiguous[7] = str(config.parent / ".." / config.parent.name / config.name)
    refused = subprocess.run(ambiguous, capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "path is ambiguous" in refused.stderr

    injected = _run_args(cli, config, auth, output) + [
        ";touch",
        str(tmp_path / "pwned"),
    ]
    refused = subprocess.run(injected, capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "usage:" in refused.stderr
    assert not (tmp_path / "pwned").exists()


def test_launcher_accepts_spaces_and_shell_metacharacters_as_path_data(
    tmp_path: Path,
) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(tmp_path)
    completed = subprocess.run(
        _run_args(cli, config, auth, output),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "cleanup").is_file()
    assert not (tmp_path / "config.json").exists()


def test_launcher_rejects_malicious_shebang_even_with_matching_hash(
    tmp_path: Path,
) -> None:
    cli = _write_fake_cli(tmp_path, shebang="#!/usr/bin/env python3")
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "preflight",
            "--cli",
            str(cli),
            "--cli-sha256",
            _sha256(cli),
        ],
        env=_contaminated_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "shebang must not contain arguments" in completed.stderr
    assert not (tmp_path / "path-command-ran").exists()


def test_launcher_rejects_symlinked_launcher_path(tmp_path: Path) -> None:
    cli = _write_fake_cli(tmp_path)
    launcher_link = tmp_path / "launcher-link"
    launcher_link.symlink_to(LAUNCHER)
    completed = subprocess.run(
        [
            str(launcher_link),
            "preflight",
            "--cli",
            str(cli),
            "--cli-sha256",
            _sha256(cli),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "launcher must not contain symlink components" in completed.stderr


def test_cli_exit_code_is_returned_after_fake_cleanup(tmp_path: Path) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(tmp_path, authorization={"exit_code": 37})
    completed = subprocess.run(
        _run_args(cli, config, auth, output),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 37
    assert (output / "cleanup").read_text(encoding="utf-8") == "complete"


def test_sigterm_reaches_cli_and_fake_cleanup_runs(tmp_path: Path) -> None:
    cli = _write_fake_cli(tmp_path)
    config, auth, output = _protected_inputs(
        tmp_path, authorization={"wait_for_signal": True}
    )
    process = subprocess.Popen(
        _run_args(cli, config, auth, output),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while not (output / "started").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert (output / "started").exists()
    process.send_signal(signal.SIGTERM)
    _stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 128 + signal.SIGTERM, stderr
    assert (output / "cleanup").read_text(encoding="utf-8") == (
        f"signal={signal.SIGTERM}"
    )


@pytest.mark.skipif(os.geteuid() != 0, reason="changing file owner requires root")
def test_launcher_rejects_cli_owned_by_another_user(tmp_path: Path) -> None:
    cli = _write_fake_cli(tmp_path)
    os.chown(cli, 1, -1)
    completed = subprocess.run(
        [
            str(LAUNCHER),
            "preflight",
            "--cli",
            str(cli),
            "--cli-sha256",
            _sha256(cli),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "owned by the current user" in completed.stderr
