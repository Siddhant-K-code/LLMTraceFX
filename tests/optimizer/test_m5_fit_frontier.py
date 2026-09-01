"""Contract tests for the process-isolated M5 context-fit frontier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.lab import frontier
from llmtracefx.optimizer.lab.core import HostSnapshot, LabError, SafetyDecision
from llmtracefx.optimizer.lab.frontier import (
    ChildProcessResult,
    build_frontier_report,
    fit_prompt,
    load_bound_base_manifest,
    load_frontier_manifest,
    machine_state,
    render_frontier_report_html,
    run_frontier,
    verify_frontier_evidence,
)

FRONTIER_MANIFEST_PATH = Path(
    "llmtracefx/optimizer/lab/data/fit-frontier-manifest-v1.json"
)
BASE_MANIFEST_PATH = Path("llmtracefx/optimizer/lab/data/lab-manifest-v1.json")


def _safe_decision() -> SafetyDecision:
    return SafetyDecision(
        safe=True,
        blockers=(),
        snapshot=HostSnapshot(
            collected_at="2026-09-01T00:00:00.000000Z",
            os_name="Darwin",
            os_release="25.6.0",
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
                "transformers": "5.16.1",
            },
        ),
    )


def _load():
    manifest, _ = load_frontier_manifest(FRONTIER_MANIFEST_PATH)
    return manifest, load_bound_base_manifest(manifest)


def _result(frontier_manifest, tier, *, status="completed", mode="exploratory"):
    completed = status == "completed"
    return {
        "schema_version": "1",
        "frontier_id": frontier_manifest.frontier_id,
        "frontier_manifest_hash": frontier.frontier_manifest_hash(frontier_manifest),
        "run_mode": mode,
        "clean_boot_confirmed": mode == "publication",
        "tier": tier.name,
        "requested_tokens": tier.requested_tokens,
        "actual_tokens": tier.requested_tokens - 1,
        "prompt_hash": "sha256:" + "a" * 64,
        "status": status,
        "reason": None if completed else "MLX/Metal reported insufficient memory",
        "timing": {
            "total_ms": 10.0 if completed else 8.0,
            "prefill_ms": 5.0 if completed else None,
            "decode_ms": 2.0 if completed else None,
        },
        "quality": (
            {
                "success": True,
                "quality_score": 1.0,
                "quality_metric": "structured_json_exact_field_match",
            }
            if completed
            else None
        ),
        "record_sha256": hashlib.sha256(b"{}").hexdigest(),
    }


def _launcher(frontier_manifest, statuses, calls):
    def launch(**kwargs):
        tier = kwargs["tier"]
        calls.append(tier.name)
        output_dir = kwargs["output_dir"]
        payload = _result(
            frontier_manifest,
            tier,
            status=statuses.get(tier.name, "completed"),
            mode=kwargs["run_mode"],
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "final_record.json").write_text("{}", encoding="utf-8")
        (output_dir / "child-result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return ChildProcessResult(
            exit_code=0 if payload["status"] == "completed" else 2,
            timed_out=False,
            descendants_cleaned=True,
        )

    return launch


def _prepare(monkeypatch):
    monkeypatch.setattr(frontier, "verify_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        frontier, "assess_safety", lambda *args, **kwargs: _safe_decision()
    )


def test_manifest_binds_exact_checkpoint_and_preserves_base_evidence() -> None:
    manifest, base = _load()
    assert manifest.model_repository_id == "mlx-community/Qwen3.8-27B-4bit"
    assert manifest.model_revision == "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
    assert manifest.model_expected_download_bytes == 16081490933
    assert manifest.base_lab_id == "m5-pro-qwen3.8-27b-v1"
    assert base.model.revision == manifest.model_revision
    assert [tier.requested_tokens for tier in manifest.tiers] == [
        256,
        512,
        1024,
        1536,
        2048,
    ]
    original = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert original["context_tiers"][0] == {
        "name": "2k",
        "target_tokens": 2048,
        "order": 1,
    }


def test_packaged_manifest_loads_from_external_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, source = load_frontier_manifest(None)
    base = load_bound_base_manifest(manifest)
    assert source.startswith("package:llmtracefx.optimizer.lab/")
    assert base.model.revision == manifest.model_revision


def test_plan_does_not_construct_runtime_or_load_weights(monkeypatch, capsys) -> None:
    _prepare(monkeypatch)
    monkeypatch.setattr(frontier, "model_files_present", lambda *args: True)
    monkeypatch.setattr(
        frontier,
        "MLXVLMRuntime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan loaded weights")
        ),
    )
    assert frontier.main(["plan", "--manifest", str(FRONTIER_MANIFEST_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["weights_loaded"] is False
    assert payload["downloads_performed"] is False


def test_fit_prompt_separates_requested_and_actual_token_counts() -> None:
    prompt, tokens = fit_prompt(
        lambda text: list(range((len(text) // 4) + 10)),
        base_prompt="task",
        requested_tokens=256,
        maximum_shortfall=8,
    )
    assert prompt.endswith("task")
    assert len(tokens) <= 256
    assert 256 - len(tokens) <= 8
    assert len(tokens) != 2048


def test_subprocess_launcher_uses_new_session_and_hidden_child(
    tmp_path, monkeypatch
) -> None:
    manifest, _ = _load()
    captured = {}

    class FakeProcess:
        pid = 12345
        returncode = 0

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def wait(self, timeout):
            return 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(frontier.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(frontier, "_clean_process_group", lambda *args: True)
    result = frontier.launch_tier_subprocess(
        manifest_path=FRONTIER_MANIFEST_PATH,
        tier=manifest.tier("t256"),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        timeout_seconds=10,
        cleanup_grace_seconds=1,
    )
    assert captured["argv"][1:4] == [
        "-m",
        "llmtracefx.optimizer.lab.frontier",
        "_child",
    ]
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False
    assert result.descendants_cleaned is True


def test_real_launcher_timeout_path_reaps_before_cleanup(tmp_path, monkeypatch) -> None:
    manifest, _ = _load()

    class FakeProcess:
        pid = 12345
        returncode = -15

        def __init__(self, argv, **kwargs):
            self.wait_calls = 0

        def wait(self, timeout):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise frontier.subprocess.TimeoutExpired("child", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

    cleaned = []
    monkeypatch.setattr(frontier.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        frontier,
        "_clean_process_group",
        lambda process, group, grace: cleaned.append(process.poll()) or True,
    )
    result = frontier.launch_tier_subprocess(
        manifest_path=FRONTIER_MANIFEST_PATH,
        tier=manifest.tier("t256"),
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        timeout_seconds=0.01,
        cleanup_grace_seconds=0.01,
    )
    assert result.timed_out is True
    assert result.descendants_cleaned is True
    assert cleaned == [-15]


def test_cleanup_polls_to_reap_leader_and_guards_signal_races(monkeypatch) -> None:
    polls = []

    class FakeProcess:
        def poll(self):
            polls.append(True)
            return -15

    group_checks = iter((True, False))
    monkeypatch.setattr(frontier, "_group_exists", lambda group: next(group_checks))
    monkeypatch.setattr(
        frontier, "_signal_process_group", lambda group, requested: True
    )
    assert frontier._clean_process_group(FakeProcess(), 12345, 0.1) is True
    assert len(polls) >= 2


def test_stop_on_first_failure_keeps_failure_shape_and_skips_later_tiers(
    tmp_path, monkeypatch
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    calls = []
    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="t2048",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {"t512": "oom"}, calls),
    )
    assert calls == ["t256", "t512"]
    assert [row["status"] for row in state["results"]] == [
        "completed",
        "oom",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert state["results"][1]["actual_tokens"] == 511
    assert state["results"][1]["timing"]["prefill_ms"] is None


def test_resume_reuses_only_matching_completed_artifact(tmp_path, monkeypatch) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    first_calls = []
    run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t256",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {}, first_calls),
    )
    second_calls = []
    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t256",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {}, second_calls),
    )
    assert first_calls == ["t256"]
    assert second_calls == []
    assert state["results"][0]["status"] == "skipped"
    assert state["results"][0]["actual_tokens"] == 255


def test_resume_rejects_stale_artifact(tmp_path, monkeypatch) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    calls = []
    run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t256",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {}, calls),
    )
    result_path = workspace / "exploratory/tiers/t256/result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["frontier_manifest_hash"] = "sha256:" + "0" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path = workspace / "exploratory/state.json"
    previous_state = state_path.read_text(encoding="utf-8")
    with pytest.raises(LabError, match="stale tier artifact"):
        run_frontier(
            manifest,
            base,
            manifest_path=FRONTIER_MANIFEST_PATH,
            workspace=workspace,
            model_path=tmp_path / "model",
            max_tier="t256",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, {}, []),
        )
    assert state_path.read_text(encoding="utf-8") == previous_state


def test_resume_refuses_to_narrow_an_existing_wider_sweep(
    tmp_path, monkeypatch
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t512",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {}, []),
    )
    state_path = workspace / "exploratory/state.json"
    prior = state_path.read_text(encoding="utf-8")
    with pytest.raises(LabError, match="cannot lower --max-tier"):
        run_frontier(
            manifest,
            base,
            manifest_path=FRONTIER_MANIFEST_PATH,
            workspace=workspace,
            model_path=tmp_path / "model",
            max_tier="t256",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, {}, []),
        )
    assert state_path.read_text(encoding="utf-8") == prior


def test_publication_requires_explicit_clean_boot_confirmation(
    tmp_path, monkeypatch
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    with pytest.raises(LabError, match="operator assertion"):
        run_frontier(
            manifest,
            base,
            manifest_path=FRONTIER_MANIFEST_PATH,
            workspace=tmp_path,
            model_path=tmp_path / "model",
            max_tier="t256",
            run_mode="publication",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, {}, []),
        )
    calls = []
    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=tmp_path,
        model_path=tmp_path / "model",
        max_tier="t256",
        run_mode="publication",
        clean_boot_confirmed=True,
        resume=True,
        launcher=_launcher(manifest, {}, calls),
    )
    assert state["run_mode"] == "publication"
    assert state["clean_boot_confirmed"] is True
    assert calls == ["t256"]


def test_machine_state_is_private_and_does_not_claim_free_gpu_memory() -> None:
    state = machine_state(_safe_decision())
    serialized = json.dumps(state)
    assert state["physical_memory_bytes"] == 24 * 1024**3
    assert state["available_memory_estimate_bytes"] == 12 * 1024**3
    assert "not free GPU memory" in state["available_memory_estimate_provenance"]
    assert "hostname" not in serialized
    assert "username" not in serialized
    assert "/Users/" not in serialized


def test_report_is_deterministic_sanitized_and_keeps_missing_null(
    tmp_path, monkeypatch
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "private-user" / "workspace"
    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t512",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {"t512": "oom"}, []),
    )
    assert state["status"] == "stopped"
    report = build_frontier_report(
        manifest, workspace=workspace, run_mode="exploratory"
    )
    first = render_frontier_report_html(report)
    second = render_frontier_report_html(report)
    assert first == second
    assert report["rows"][0]["requested_tokens"] == 256
    assert report["rows"][0]["actual_tokens"] == 255
    assert report["rows"][1]["outcome"] == "oom"
    assert report["rows"][1]["prefill_ms"] is None
    assert report["rows"][2]["outcome"] == "skipped"
    assert report["maximum_completed"] == {
        "tier": "t256",
        "requested_tokens": 256,
        "actual_tokens": 255,
    }
    assert str(tmp_path) not in json.dumps(report)
    assert "GPU utilization" in report["limitations"][-2]
    verification = verify_frontier_evidence(
        manifest, workspace=workspace, run_mode="exploratory"
    )
    assert verification["verified"] is True


def test_invalid_completed_child_without_evaluator_stops_failure_shaped(
    tmp_path, monkeypatch
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)

    def launch(**kwargs):
        payload = _result(manifest, kwargs["tier"])
        payload["quality"] = None
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True)
        (output_dir / "child-result.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        return ChildProcessResult(0, False, True)

    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="t256",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["status"] == "stopped"
    assert state["results"][0]["status"] == "failed"
    assert state["results"][0]["reason"] == "child artifact failed validation"


def test_unattempted_rows_verify_with_null_measurements(tmp_path, monkeypatch) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="t2048",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, {"t512": "oom"}, []),
    )
    result = verify_frontier_evidence(
        manifest, workspace=workspace, run_mode="exploratory"
    )
    assert result == {"verified": True, "failures": []}


@pytest.mark.parametrize(
    ("process_result", "expected_status", "expected_reason"),
    [
        (
            ChildProcessResult(None, True, True),
            "timeout",
            "tier exceeded parent-enforced timeout",
        ),
        (
            ChildProcessResult(1, False, False),
            "failed",
            "child process group cleanup could not be verified",
        ),
    ],
)
def test_timeout_and_cleanup_failures_stop_without_advancing(
    tmp_path,
    monkeypatch,
    process_result,
    expected_status,
    expected_reason,
) -> None:
    manifest, base = _load()
    _prepare(monkeypatch)
    calls = []

    def launch(**kwargs):
        calls.append(kwargs["tier"].name)
        return process_result

    state = run_frontier(
        manifest,
        base,
        manifest_path=FRONTIER_MANIFEST_PATH,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="t512",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert calls == ["t256"]
    assert state["results"][0]["status"] == expected_status
    assert state["results"][0]["reason"] == expected_reason
    assert state["results"][1]["status"] == "skipped"
