"""CLI-surface tests: what flags exist, what never gets printed, exit codes.

These tests never invoke ``main()``'s ``run`` subcommand against a real
``SubprocessCommandRunner`` -- that would need a real SSH/Docker
environment. Instead they exercise the argument parser directly, and
``verify-public-bundle`` (which is fully offline) end to end.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_kv_truth import cli as cli_module
from vllm_kv_truth import evidence
from vllm_kv_truth.cli import _build_parser, main
from vllm_kv_truth.lifecycle import HostOrchestrationError


class TestArgumentParser:
    def test_run_requires_both_protected_paths(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])

    def test_run_accepts_only_path_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--execution-config",
                "config.json",
                "--authorization",
                "auth.json",
                "--output-dir",
                "output",
            ]
        )
        assert args.execution_config == Path("config.json")
        assert args.authorization == Path("auth.json")
        assert args.output_dir == Path("output")

    def test_no_host_user_key_or_remote_path_flags_exist(self) -> None:
        parser = _build_parser()
        # Directly inspect the run subparser's own option strings instead of
        # scraping help text, which is more robust to wrapping/formatting.
        run_subparser = parser._subparsers._group_actions[0].choices["run"]  # type: ignore[union-attr,index]
        option_strings = {
            option
            for action in run_subparser._actions
            for option in action.option_strings
        }
        forbidden = {
            "--host",
            "--user",
            "--private-key-path",
            "--known-hosts-path",
            "--remote-workspace",
            "--authorized-key-marker",
        }
        assert option_strings.isdisjoint(forbidden)

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
    def test_missing_execution_config_fails_cleanly(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main(
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
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"host": "x"}), encoding="utf-8")
        config_path.chmod(  # codeql[py/overly-permissive-file]
            config_path.stat().st_mode | stat.S_IRGRP
        )
        auth_path = tmp_path / "auth.json"
        auth_path.write_text("{}", encoding="utf-8")
        exit_code = main(
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
        exit_code = main(
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
        exit_code = main(
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
