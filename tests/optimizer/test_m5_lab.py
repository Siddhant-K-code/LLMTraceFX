"""Safety, reproducibility, resume, and privacy tests for the M5 lab."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.lab import cli
from llmtracefx.optimizer.lab.core import (
    HostSnapshot,
    LabError,
    SafetyDecision,
    assert_shareable,
    assess_safety,
    build_shareable_report,
    render_lab_report_html,
    run_lab,
    verify_catalog,
    verify_evidence,
    write_reports,
)
from llmtracefx.optimizer.lab.manifest import (
    LabManifest,
    LabManifestError,
    ModelFilePin,
)
from llmtracefx.optimizer.manifest import EnvironmentManifest
from llmtracefx.optimizer.schema import PlatformInfo
from llmtracefx.optimizer.workloads.verify import RowStatus, RowVerification

MANIFEST_PATH = Path("llmtracefx/optimizer/lab/data/lab-manifest-v1.json")


@pytest.fixture(autouse=True)
def stable_m5_platform(monkeypatch):
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.mlx.record_platform",
        lambda accelerator: PlatformInfo(
            os_name="Darwin",
            os_version="25.6.0",
            architecture="arm64",
            cpu_cores=18,
            total_memory_gb=24.0,
            accelerator=accelerator,
        ),
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.mlx.collect_environment_manifest",
        lambda **kwargs: EnvironmentManifest(
            schema_version="1",
            collected_at="2026-08-31T00:00:00.000000Z",
            os_name="Darwin",
            os_release="25.6.0",
            architecture="arm64",
            python_implementation="CPython",
            python_version="3.13.15",
            cpu_count=18,
            total_memory_gb=24.0,
            package_versions={
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "mlx-vlm": "0.6.8",
                "transformers": "5.16.1",
            },
        ),
    )


@dataclass
class FakeResponse:
    text: str
    from_draft: bool = False
    prompt_tokens: int = 3
    generation_tokens: int = 1
    finish_reason: str | None = None


class FakeTokenizer:
    bos_token = None


class FakeRuntime:
    mlx_version = "0.32.2"
    mlx_lm_version = "0.6.8"
    runtime_name = "mlx-vlm"
    runtime_version = "0.6.8"
    generate_calls = 0
    peak_bytes = 1024**3
    fail_generation = False

    def __init__(self, **kwargs):
        self.prompt = ""

    def load_model(self, path):
        return object(), FakeTokenizer()

    def encode(self, tokenizer, prompt):
        self.prompt = prompt
        return [1, 2, 3]

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return MLXMemorySnapshot(
            active_bytes=512 * 1024**2,
            cache_bytes=128 * 1024**2,
            peak_bytes=type(self).peak_bytes,
        )

    def accelerator_name(self):
        return "Apple M5 Pro"

    def stream_generate(
        self,
        model,
        tokenizer,
        prompt_tokens,
        *,
        max_tokens,
        draft_model,
        num_draft_tokens,
    ):
        type(self).generate_calls += 1
        if type(self).fail_generation:
            raise RuntimeError("simulated warmup failure")
        text = (
            '{"name":"Priya","age":34,"is_active":true}'
            if "Priya Nakamura" in self.prompt
            else "3 hours. The combined closing speed is 70 miles per hour."
        )
        yield FakeResponse(text=text)


def _safe_decision() -> SafetyDecision:
    return SafetyDecision(
        safe=True,
        blockers=(),
        snapshot=HostSnapshot(
            collected_at="2026-08-31T00:00:00.000000Z",
            os_name="Darwin",
            os_release="25.0",
            architecture="arm64",
            python_implementation="CPython",
            python_version="3.13.15",
            cpu_count=18,
            chip="Apple M5 Pro",
            total_memory_bytes=24 * 1024**3,
            memory_free_percent=50.0,
            swap_used_bytes=0,
            disk_free_bytes=500 * 1024**3,
            package_versions={
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "mlx-vlm": "0.6.8",
            },
        ),
    )


def test_pinned_manifest_and_catalog_validate() -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    verify_catalog(manifest)
    assert manifest.model.repository_id == "mlx-community/Qwen3.8-27B-4bit"
    assert manifest.model.revision == "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
    assert manifest.model.license == "Apache-2.0"


def test_manifest_rejects_unpinned_revision() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["model"]["revision"] = "main"
    with pytest.raises(LabManifestError, match="40-character git revision"):
        LabManifest.from_dict(payload)


def test_model_verification_rejects_unpinned_root_entries(
    tmp_path, monkeypatch
) -> None:
    from llmtracefx.optimizer.lab.core import verify_model

    manifest = LabManifest.read_json(MANIFEST_PATH)
    pinned_file = ModelFilePin(
        path="config.json",
        size_bytes=1,
        sha256="0" * 64,
    )
    manifest = replace(
        manifest,
        model=replace(
            manifest.model,
            expected_download_bytes=1,
            files=(pinned_file,),
        ),
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_bytes(b"x")
    (model_path / "stale-tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core._sha256_file",
        lambda path: "0" * 64,
    )
    with pytest.raises(LabError, match="unpinned model-root entry"):
        verify_model(manifest, model_path)


def test_manifest_rejects_credential_bearing_source_url() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["model"]["sources"][0] = "https://token@example.com/model"
    with pytest.raises(LabManifestError, match="credential-free"):
        LabManifest.from_dict(payload)


def test_manifest_rejects_source_url_query_credentials() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["model"]["sources"][0] = "https://example.com/model?token=secret"
    with pytest.raises(LabManifestError, match="credential-free"):
        LabManifest.from_dict(payload)


def test_manifest_rejects_encoded_or_unapproved_source_url() -> None:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["model"]["sources"][0] = "https://example.com/%68%66_abcdefghijklmnopqrstuv"
    with pytest.raises(LabManifestError, match="credential-free"):
        LabManifest.from_dict(payload)


def test_default_cli_action_is_no_download_plan(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "assess_safety", lambda *args, **kwargs: _safe_decision())
    monkeypatch.setattr(cli, "model_files_present", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        cli,
        "acquire_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan must not download")
        ),
    )
    assert cli.main(["--manifest", str(MANIFEST_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["no_spend"] is True
    assert payload["downloads_performed"] is False


def test_default_manifest_loads_outside_source_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args([])
    manifest, source, _, _ = cli._load(args)
    assert manifest.model.repository_id == "mlx-community/Qwen3.8-27B-4bit"
    assert source == ("package:llmtracefx.optimizer.lab/data/lab-manifest-v1.json")


def test_safety_fails_closed_when_swap_is_unavailable(tmp_path, monkeypatch) -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    snapshot = _safe_decision().snapshot
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.collect_host_snapshot",
        lambda path: HostSnapshot(**{**snapshot.to_dict(), "swap_used_bytes": None}),
    )
    decision = assess_safety(manifest, tmp_path, include_download=False)
    assert decision.safe is False
    assert any("swap usage could not be measured" in item for item in decision.blockers)


def test_run_resumes_hash_matching_rows(tmp_path, monkeypatch) -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    model_path = tmp_path / "model"
    model_path.mkdir()
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.verify_model",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.assess_safety",
        lambda *args, **kwargs: _safe_decision(),
    )
    monkeypatch.setattr("llmtracefx.optimizer.lab.core.MLXVLMRuntime", FakeRuntime)
    FakeRuntime.generate_calls = 0
    FakeRuntime.peak_bytes = 1024**3
    FakeRuntime.fail_generation = False

    run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier="2k",
        resume=True,
    )
    first_calls = FakeRuntime.generate_calls
    run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier="2k",
        resume=True,
    )

    assert first_calls == 5
    assert FakeRuntime.generate_calls == first_calls + 1
    statuses = [
        RowVerification.read_json(path).status
        for path in (workspace / "results" / "runs").glob("*/verification.json")
    ]
    assert statuses and set(statuses) == {RowStatus.SKIPPED}

    missing = next((workspace / "results" / "runs").glob("*/verification.json"))
    missing.unlink()
    verification = verify_evidence(manifest, workspace=workspace)
    assert verification["verified"] is False
    assert any(
        "expected current-run artifact is missing" in failure
        for failure in verification["failures"]
    )


def test_warmup_memory_gate_stops_before_measurements(tmp_path, monkeypatch) -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    model_path = tmp_path / "model"
    model_path.mkdir()
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.verify_model",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.assess_safety",
        lambda *args, **kwargs: _safe_decision(),
    )
    monkeypatch.setattr("llmtracefx.optimizer.lab.core.MLXVLMRuntime", FakeRuntime)
    FakeRuntime.generate_calls = 0
    FakeRuntime.peak_bytes = manifest.safety.maximum_peak_memory_bytes + 1
    FakeRuntime.fail_generation = False

    state = run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier="2k",
        resume=True,
    )

    assert FakeRuntime.generate_calls == 1
    assert state["tiers"][0]["status"] == "failed_warmup_safety"
    assert not (workspace / "results" / "runs").exists()
    write_reports(manifest, workspace=workspace)


def test_multiple_warmups_stop_and_verify_only_attempted_runs(
    tmp_path, monkeypatch
) -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    manifest = replace(
        manifest,
        repetitions=replace(manifest.repetitions, warmup_per_tier=2),
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.verify_model",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.assess_safety",
        lambda *args, **kwargs: _safe_decision(),
    )
    monkeypatch.setattr("llmtracefx.optimizer.lab.core.MLXVLMRuntime", FakeRuntime)
    FakeRuntime.generate_calls = 0
    FakeRuntime.peak_bytes = 1024**3
    FakeRuntime.fail_generation = True

    state = run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier="2k",
        resume=True,
    )
    verification = verify_evidence(manifest, workspace=workspace)

    assert FakeRuntime.generate_calls == 1
    assert state["tiers"][0]["run_ids"] == [
        "structured-json-profile-extraction-2k-warmup-01"
    ]
    assert verification["verified"] is True


def test_shareable_report_and_chart_are_deterministic_and_private(
    tmp_path, monkeypatch
) -> None:
    manifest = LabManifest.read_json(MANIFEST_PATH)
    model_path = tmp_path / "model"
    model_path.mkdir()
    workspace = tmp_path / "private-user-name" / "workspace"
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.verify_model",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.core.assess_safety",
        lambda *args, **kwargs: _safe_decision(),
    )
    monkeypatch.setattr("llmtracefx.optimizer.lab.core.MLXVLMRuntime", FakeRuntime)
    FakeRuntime.peak_bytes = 1024**3
    FakeRuntime.fail_generation = False
    run_lab(
        manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier="2k",
        resume=True,
    )
    report = build_shareable_report(manifest, workspace=workspace)
    first = render_lab_report_html(report)
    second = render_lab_report_html(report)
    shareable = tmp_path / "shareable"
    written = write_reports(manifest, workspace=workspace, shareable_dir=shareable)

    assert first == second
    assert str(tmp_path) not in json.dumps(report)
    assert str(tmp_path) not in first
    assert "http://" not in first
    assert "https://" not in first
    assert_shareable(report)
    assert written == report
    assert (workspace / "reports" / "tune-report.json").is_file()
    assert (workspace / "reports" / "compare-report.json").is_file()
    assert (shareable / "evidence-summary.json").is_file()
    assert (shareable / "report.html").read_text(encoding="utf-8") == first

    record_path = next((workspace / "results" / "runs").glob("*/final_record.json"))
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["command"]["config_hash"] = "sha256:" + "0" * 64
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="unverified current-run evidence"):
        write_reports(manifest, workspace=workspace)


@pytest.mark.parametrize(
    "private_value",
    [
        "/Users/alice/private/run.json",
        "/Volumes/PrivateSSD/alice/model/config.json",
        "hf_abcdefghijklmnopqrstuv",
    ],
)
def test_shareable_artifacts_reject_private_values(private_value) -> None:
    with pytest.raises(LabError, match="private"):
        assert_shareable({"value": private_value})
