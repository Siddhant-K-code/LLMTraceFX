"""Unit tests for the shared lab tier lifecycle."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtracefx.optimizer.lab.manifest import LabManifest
from llmtracefx.optimizer.lab.run_owner import current_run_owner
from llmtracefx.optimizer.lab.tier_lifecycle import (
    WARMUP_FAILURE_TIER_STATUSES,
    InvalidRunTransition,
    RunStatus,
    StopPhase,
    TierStatus,
    run_tiers,
    start_run,
    transition_run_status,
)
from llmtracefx.optimizer.workloads.verify import RowStatus

MANIFEST_PATH = Path("llmtracefx/optimizer/lab/data/lab-manifest-v1.json")


class Decision:
    def __init__(self, blockers: tuple[str, ...] = ()):
        self.safe = not blockers
        self.blockers = blockers

    def to_dict(self):
        return {"safe": self.safe, "blockers": list(self.blockers)}


def _row(run_id, *, status=RowStatus.COMPLETED, reason=None, peak=1024.0, ok=True):
    return SimpleNamespace(
        entry=SimpleNamespace(run_id=run_id),
        verification=SimpleNamespace(status=status, reason=reason, outcome_success=ok),
        final_record=SimpleNamespace(
            memory=SimpleNamespace(peak=SimpleNamespace(value=peak))
        ),
    )


class Rows:
    """Records calls and returns rows, optionally failing chosen run ids."""

    def __init__(self, failures=None):
        self.calls: list[dict] = []
        self.failures = failures or {}

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        label = "warmup" if kwargs["warmup"] else "rep"
        run_id = (
            f"{kwargs['workload_id']}-{kwargs['tier'].value}-{label}-"
            f"{kwargs['repetition_index'] + 1:02d}"
        )
        return _row(run_id, **self.failures.get(run_id, {}))


class Safety:
    """Returns an unsafe decision on the chosen 1-based call numbers."""

    def __init__(self, unsafe_calls=()):
        self.count = 0
        self.unsafe_calls = set(unsafe_calls)

    def __call__(self):
        self.count += 1
        return Decision(("host unsafe",) if self.count in self.unsafe_calls else ())


@pytest.fixture
def manifest() -> LabManifest:
    return LabManifest.read_json(MANIFEST_PATH)


def _start(workspace: Path, **kwargs) -> dict:
    return start_run(
        workspace=workspace,
        schema_version="1",
        lab_id="lab",
        requested_max_tier="16k",
        preflight=Decision(),
        hardware_fingerprint="hw",
        **kwargs,
    )


def _run(manifest, workspace, *, max_tier="16k", rows=None, safety=None, **kwargs):
    state = _start(workspace)
    return run_tiers(
        manifest,
        state=state,
        workspace=workspace,
        max_tier=max_tier,
        resume=True,
        run_row=rows or Rows(),
        assess=safety or Safety(),
        **kwargs,
    )


def test_run_status_allows_only_running_to_terminal() -> None:
    for terminal in (RunStatus.COMPLETED, RunStatus.STOPPED):
        state = {"status": "running"}
        transition_run_status(state, terminal)
        assert state["status"] == terminal.value


@pytest.mark.parametrize(
    ("current", "new"),
    [
        ("completed", RunStatus.STOPPED),
        ("completed", RunStatus.COMPLETED),
        ("stopped", RunStatus.RUNNING),
        ("running", RunStatus.RUNNING),
    ],
)
def test_run_status_rejects_invalid_transition(current, new) -> None:
    state = {"status": current}
    with pytest.raises(InvalidRunTransition):
        transition_run_status(state, new)
    assert state["status"] == current


def test_warmup_failure_statuses_match_raw_strings() -> None:
    assert "failed_warmup" in WARMUP_FAILURE_TIER_STATUSES
    assert "failed_warmup_safety" in WARMUP_FAILURE_TIER_STATUSES
    assert "failed" not in WARMUP_FAILURE_TIER_STATUSES


def test_start_run_persists_running_state_with_header_order(tmp_path) -> None:
    state = _start(
        tmp_path / "ws",
        header={"run_mode": "exploratory", "clean_boot_confirmed": False},
        identity={"run_mode": "exploratory"},
    )
    on_disk = json.loads((tmp_path / "ws" / "state.json").read_text())
    assert on_disk == state
    assert list(state) == [
        "schema_version",
        "lab_id",
        "run_mode",
        "clean_boot_confirmed",
        "started_at",
        "status",
        "owner",
        "requested_max_tier",
        "preflight",
        "tiers",
        "stopped_before_tier",
        "stopped_during",
        "unattempted_tiers",
        "stop_reasons",
        "execution_id",
    ]
    assert state["status"] == RunStatus.RUNNING.value
    assert state["owner"] == current_run_owner()


def test_start_run_identity_changes_execution_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "llmtracefx.optimizer.lab.tier_lifecycle.utc_now_iso",
        lambda: "2000-01-01T00:00:00Z",
    )
    plain = _start(tmp_path / "a")
    moded = _start(tmp_path / "b", identity={"run_mode": "publication"})
    assert plain["execution_id"] != moded["execution_id"]


def test_all_tiers_pass_and_rows_use_expected_layout(manifest, tmp_path) -> None:
    rows = Rows()
    workspace = tmp_path / "ws"
    state = _run(manifest, workspace, max_tier="8k", rows=rows)

    assert state["status"] == "completed"
    assert [tier["tier"] for tier in state["tiers"]] == ["2k", "8k"]
    assert {tier["status"] for tier in state["tiers"]} == {"passed"}
    assert state["stop_reasons"] == []
    assert json.loads((workspace / "state.json").read_text()) == state
    warmups = [call for call in rows.calls if call["warmup"]]
    measured = [call for call in rows.calls if not call["warmup"]]
    assert {call["resume"] for call in warmups} == {False}
    assert warmups[0]["output_dir"] == workspace / "warmups" / "2k"
    assert {call["resume"] for call in measured} == {True}
    assert {call["output_dir"] for call in measured} == {workspace / "results"}
    per_tier = (
        manifest.repetitions.warmup_per_tier
        + len(manifest.workloads) * manifest.repetitions.measured_per_workload
    )
    assert len(rows.calls) == 2 * per_tier


def test_unsafe_tier_preflight_stops_before_that_tier(manifest, tmp_path) -> None:
    # Calls: 2k preflight, 2k warmup gate, 2k postflight, 8k preflight.
    state = _run(manifest, tmp_path, safety=Safety({4}))
    assert [tier["tier"] for tier in state["tiers"]] == ["2k"]
    assert state["stopped_before_tier"] == "8k"
    assert state["stopped_during"] is None
    assert state["unattempted_tiers"] == ["8k", "16k"]
    assert state["stop_reasons"] == ["host unsafe"]
    assert state["status"] == "stopped"


def test_failed_warmup_prefers_recorded_reason(manifest, tmp_path) -> None:
    warmup_id = f"{manifest.workloads[0].workload_id}-2k-warmup-01"
    rows = Rows({warmup_id: {"status": RowStatus.FAILED, "reason": "oom"}})
    state = _run(manifest, tmp_path, rows=rows)
    assert state["tiers"][0]["status"] == TierStatus.FAILED_WARMUP.value
    assert state["stopped_during"] == {"tier": "2k", "phase": "warmup"}
    assert state["stop_reasons"] == ["oom"]
    assert state["unattempted_tiers"] == ["8k", "16k"]
    assert not any(not call["warmup"] for call in rows.calls)


def test_failed_warmup_without_reason_uses_describe_failure(manifest, tmp_path) -> None:
    warmup_id = f"{manifest.workloads[0].workload_id}-2k-warmup-01"
    rows = Rows({warmup_id: {"ok": False}})
    state = _run(
        manifest,
        tmp_path,
        rows=rows,
        row_ok=lambda result: result.verification.outcome_success is True,
        describe_failure=lambda result: f"{result.entry.run_id} was wrong",
    )
    assert state["tiers"][0]["status"] == "failed_warmup"
    assert state["stop_reasons"] == [f"{warmup_id} was wrong"]


def test_extra_blockers_fail_warmup_safety_gate(manifest, tmp_path) -> None:
    state = _run(manifest, tmp_path, extra_blockers=lambda results: ("extra",))
    assert state["tiers"][0]["status"] == "failed_warmup_safety"
    assert state["stopped_during"] == {
        "tier": "2k",
        "phase": StopPhase.WARMUP_SAFETY.value,
    }
    assert state["stop_reasons"] == ["extra"]


def test_warmup_peak_over_limit_fails_warmup_safety(manifest, tmp_path) -> None:
    warmup_id = f"{manifest.workloads[0].workload_id}-2k-warmup-01"
    limit = manifest.safety.maximum_peak_memory_bytes
    state = _run(manifest, tmp_path, rows=Rows({warmup_id: {"peak": limit + 1.0}}))
    assert state["tiers"][0]["status"] == "failed_warmup_safety"
    assert state["tiers"][0]["maximum_observed_peak_memory_bytes"] == limit + 1.0


def test_measured_failure_stops_rows_and_later_tiers(manifest, tmp_path) -> None:
    first = f"{manifest.workloads[0].workload_id}-8k-rep-01"
    rows = Rows({first: {"status": RowStatus.FAILED}})
    state = _run(manifest, tmp_path, rows=rows)
    assert [tier["status"] for tier in state["tiers"]] == ["passed", "failed"]
    assert state["tiers"][1]["run_ids"] == [first]
    assert state["stopped_before_tier"] == "16k"
    assert state["stopped_during"] == {
        "tier": "8k",
        "phase": "measured_or_postflight",
    }
    assert state["unattempted_tiers"] == ["16k"]
    assert state["stop_reasons"] == [f"{first} ended as failed"]


def test_measured_failure_on_last_tier_has_no_next_tier(manifest, tmp_path) -> None:
    first = f"{manifest.workloads[0].workload_id}-2k-rep-01"
    rows = Rows({first: {"status": RowStatus.FAILED}})
    state = _run(manifest, tmp_path, max_tier="2k", rows=rows)
    assert state["stopped_before_tier"] is None
    assert state["unattempted_tiers"] == []
    assert state["status"] == "stopped"


def test_without_stop_on_failure_all_measured_rows_run(manifest, tmp_path) -> None:
    manifest = replace(
        manifest, safety=replace(manifest.safety, stop_on_any_failed_row=False)
    )
    first = f"{manifest.workloads[0].workload_id}-2k-rep-01"
    rows = Rows({first: {"status": RowStatus.FAILED}})
    state = _run(
        manifest,
        tmp_path,
        max_tier="2k",
        rows=rows,
        extra_blockers=lambda results: tuple(
            f"{result.entry.run_id} bad"
            for result in results
            if result.verification.status is RowStatus.FAILED
        ),
    )
    expected = len(manifest.workloads) * manifest.repetitions.measured_per_workload
    assert len(state["tiers"][0]["run_ids"]) == expected
    assert state["tiers"][0]["status"] == "failed"
    assert state["stop_reasons"] == [f"{first} bad"]


def test_mid_run_state_is_written_after_each_measured_tier(manifest, tmp_path) -> None:
    workspace = tmp_path / "ws"
    snapshots: list[list[str]] = []
    rows = Rows()

    def recording_rows(**kwargs):
        on_disk = json.loads((workspace / "state.json").read_text())
        snapshots.append([tier["tier"] for tier in on_disk["tiers"]])
        return rows(**kwargs)

    _run(manifest, workspace, max_tier="8k", rows=recording_rows)
    assert snapshots[0] == []
    assert snapshots[-1] == ["2k"]
