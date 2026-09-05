from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmtracefx.cache_audit import cli, compile_audit, verify_audit_bundle


def _invoke(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 0
    return json.loads(capsys.readouterr().out)


def _invoke_error(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    return json.loads(capsys.readouterr().out)


def test_compile_run_verify_and_sanitize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workload = tmp_path / "workload.json"
    compiled = _invoke(
        ["compile", "--output", str(workload), "--block-size", "4"], capsys
    )
    assert compiled["requests"] == 9

    bundle = tmp_path / "bundle"
    completed = _invoke(
        [
            "run",
            "--backend",
            "reference",
            "--workload",
            str(workload),
            "--output-dir",
            str(bundle),
        ],
        capsys,
    )
    assert completed["completed"] is True
    verified = _invoke(["verify", str(bundle)], capsys)
    assert verified["verified"] is True

    redacted = tmp_path / "redacted"
    sanitized = _invoke(
        ["sanitize", str(bundle), "--output-dir", str(redacted)], capsys
    )
    assert sanitized["token_identity_reproducible"] is False
    evidence = (redacted / "request-evidence.jsonl").read_text(encoding="utf-8")
    assert '"input_token_ids":null' in evidence
    assert '"output_token_ids":null' in evidence


def test_reference_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    result = _invoke(["capabilities", "--backend", "reference"], capsys)
    assert result["supported"] is True
    assert result["backend"] == "synthetic_reference"


def test_direct_public_redacted_run_is_refused_with_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = _invoke_error(
        [
            "run",
            "--backend",
            "reference",
            "--publication-mode",
            "public_redacted",
            "--output-dir",
            str(tmp_path / "redacted"),
        ],
        capsys,
    )
    assert "run privately" in result["error"]
    assert "sanitize" in result["error"]


def test_real_backend_capabilities_fail_closed_without_import_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mlx = _invoke(["capabilities", "--backend", "mlx"], capsys)
    assert mlx["backend"] == "mlx_lm_local"
    assert isinstance(mlx["supported"], bool)
    vllm = _invoke(["capabilities", "--backend", "vllm"], capsys)
    assert vllm["backend"] == "vllm"
    assert isinstance(vllm["reasons"], list)


def test_python_api_compiles_and_verifies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = compile_audit(block_size=4)
    assert requests[0].request_id == "cold"
    bundle = tmp_path / "bundle"
    _invoke(
        [
            "run",
            "--backend",
            "reference",
            "--output-dir",
            str(bundle),
        ],
        capsys,
    )
    assert verify_audit_bundle(bundle)["request_count"] == len(requests)
