"""CLI-surface tests: what flags exist, what never gets printed, exit codes.

The public ``main`` exposes only offline verification. Bootstrap-only run and
preflight behavior is tested through the separate internal dispatcher.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_kv_truth import cli as cli_module
from vllm_kv_truth import evidence
from vllm_kv_truth.cli import _build_parser, bootstrap_dispatch, main
from vllm_kv_truth.lifecycle import HostOrchestrationError


class TestArgumentParser:
    def test_public_parser_exposes_only_offline_verification(self) -> None:
        parser = _build_parser()
        choices = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
        assert set(choices) == {"verify-public-bundle"}

    def test_verify_public_bundle_requires_bundle_dir(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["verify-public-bundle"])

    def test_unknown_command_is_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["not-a-command"])


def _fixture_public_bundle_dir(tmp_path: Path) -> Path:
    matrix = evidence.build_claim_matrix(
        {
            "lane": "B",
            "records": [
                {
                    "request_id": f"req-{i}",
                    "scenario": "cold" if i == 0 else "identical_prefix",
                    "num_cached_tokens": 0,
                    "num_cache_creation_tokens": 1,
                    "boundary_valid": True,
                    "event_batches": [],
                }
                for i in range(10)
            ],
        }
    )
    ledger = evidence.ListRateLedger(entries=())
    teardown = evidence.TeardownReceipt(
        residual_containers=0,
        residual_gpu_processes=0,
        evidence_transferred_and_verified=True,
        shutdown_issued=True,
        safe_to_terminate_message_emitted=True,
    )
    private = evidence.PrivateEvidenceBundle(
        run_mode=evidence.RUN_MODE_SYNTHETIC_FIXTURE,
        experiment_nonce="f" * 40,
        authorization={"protocol_id": "qwen3-8b-vllm-kv-truth-v1"},
        ssh_options_public_record={"batch_mode": True},
        lane_receipts={},
        claim_matrix=matrix,
        list_rate_ledger=ledger,
        teardown_receipt=teardown,
    )
    public = evidence.PublicRedactedBundle.from_private(private)
    directory = tmp_path / "public-bundle"
    evidence.write_public_bundle_directory(public, directory)
    return directory


class TestVerifyPublicBundleCommand:
    def test_verifies_a_well_formed_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = _fixture_public_bundle_dir(tmp_path)
        exit_code = main(["verify-public-bundle", "--bundle-dir", str(directory)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "verification: ok" in captured.out
        assert evidence.RUN_MODE_SYNTHETIC_FIXTURE in captured.out

    def test_rejects_a_tampered_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = _fixture_public_bundle_dir(tmp_path)
        (directory / "report.txt").write_text("tampered", encoding="utf-8")
        exit_code = main(["verify-public-bundle", "--bundle-dir", str(directory)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "verification: ok" not in captured.out

    def test_error_output_never_leaks_directory_argv(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        missing = tmp_path / "does-not-exist"
        exit_code = main(["verify-public-bundle", "--bundle-dir", str(missing)])
        assert exit_code == 1


class TestRunCommandRejectsBadInputs:
    @pytest.mark.parametrize(
        "command", ["run", "preflight", "preflight-clean-environment"]
    )
    def test_public_cli_rejects_bootstrap_only_commands_before_private_reads(
        self, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        def fail_if_loaded(_path: Path) -> object:
            raise AssertionError("private inputs must not be read")

        monkeypatch.setattr(cli_module.ProtectedExecutionConfig, "load", fail_if_loaded)
        with pytest.raises(SystemExit) as caught:
            main(
                [
                    command,
                    "--execution-config",
                    "/protected/config.json",
                    "--authorization",
                    "/protected/authorization.json",
                    "--output-dir",
                    "/protected/output",
                ]
            )
        assert caught.value.code == 2

    def test_missing_execution_config_fails_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            cli_module, "_require_clean_parent_environment", lambda: None
        )
        exit_code = bootstrap_dispatch(
            [
                "run",
                "--execution-config",
                str(tmp_path / "missing-config.json"),
                "--authorization",
                str(tmp_path / "missing-auth.json"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        assert exit_code == 1

    def test_world_readable_config_is_rejected_before_any_network_activity(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"host": "x"}), encoding="utf-8")
        config_path.chmod(  # codeql[py/overly-permissive-file]
            config_path.stat().st_mode | stat.S_IRGRP
        )
        auth_path = tmp_path / "auth.json"
        auth_path.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            cli_module, "_require_clean_parent_environment", lambda: None
        )
        exit_code = bootstrap_dispatch(
            [
                "run",
                "--execution-config",
                str(config_path),
                "--authorization",
                str(auth_path),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "x" not in captured.out

    def test_output_directory_must_match_protected_config(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            cli_module, "_require_clean_parent_environment", lambda: None
        )
        configured_output = tmp_path / "configured-output"
        monkeypatch.setattr(
            cli_module.ProtectedExecutionConfig,
            "load",
            lambda _path: SimpleNamespace(local_evidence_dir=configured_output),
        )
        monkeypatch.setattr(
            cli_module.RunAuthorization,
            "read",
            lambda _path: SimpleNamespace(signature_path=None),
        )
        exit_code = bootstrap_dispatch(
            [
                "run",
                "--execution-config",
                str(tmp_path / "config.json"),
                "--authorization",
                str(tmp_path / "authorization.json"),
                "--output-dir",
                str(tmp_path / "different-output"),
            ]
        )
        assert exit_code == 1

    def test_run_reports_safe_failed_substage_and_reason(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            cli_module, "_require_clean_parent_environment", lambda: None
        )
        authorization = SimpleNamespace(signature_path=None)
        output_dir = Path.cwd() / "unused"
        monkeypatch.setattr(
            cli_module.ProtectedExecutionConfig,
            "load",
            lambda _path: SimpleNamespace(local_evidence_dir=output_dir),
        )
        monkeypatch.setattr(
            cli_module.RunAuthorization,
            "read",
            lambda _path: authorization,
        )

        class RefusingOrchestrator:
            def __init__(self, **_kwargs: object) -> None:
                pass

            def run(self, **_kwargs: object) -> tuple[object, ...]:
                raise HostOrchestrationError(
                    "the remote host does not provide Docker",
                    stage="preflight",
                    substage="host_probe",
                    reason_code="preflight_missing_docker",
                )

        monkeypatch.setattr(cli_module, "SubprocessCommandRunner", lambda: object())
        monkeypatch.setattr(cli_module, "RemoteOrchestrator", RefusingOrchestrator)
        exit_code = bootstrap_dispatch(
            [
                "run",
                "--execution-config",
                "protected.json",
                "--authorization",
                "authorization.json",
                "--output-dir",
                str(output_dir),
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert (
            "preflight/host_probe: preflight_missing_docker: "
            "the remote host does not provide Docker"
        ) in captured.err
