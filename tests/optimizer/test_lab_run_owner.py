"""Owner metadata lets a reader tell a live lab run from an abandoned one."""

from __future__ import annotations

import os
import subprocess
import sys

from llmtracefx.optimizer.lab import run_owner
from llmtracefx.optimizer.lab.core import assert_shareable
from llmtracefx.optimizer.lab.run_owner import (
    current_run_owner,
    describe_running_state,
    running_state_liveness,
)


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _running(owner: object) -> dict[str, object]:
    state: dict[str, object] = {
        "status": "running",
        "started_at": "2026-11-09T00:00:00.000000Z",
    }
    if owner is not None:
        state["owner"] = owner
    return state


def test_owner_records_pid_and_hashed_host_only() -> None:
    owner = current_run_owner()
    assert owner["pid"] == os.getpid()
    assert owner["host_id"].startswith("sha256:")
    assert set(owner) == {"pid", "host_id"}
    assert_shareable(owner)


def test_owner_from_this_live_process_is_active() -> None:
    state = _running(current_run_owner())
    assert running_state_liveness(state) == "active"
    message = describe_running_state(state)
    assert f"still active as pid {os.getpid()}" in message
    assert "wait for it to finish" in message


def test_owner_whose_process_exited_is_abandoned() -> None:
    pid = _dead_pid()
    state = _running({"pid": pid, "host_id": current_run_owner()["host_id"]})
    assert running_state_liveness(state) == "abandoned"
    message = describe_running_state(state)
    assert "2026-11-09T00:00:00.000000Z" in message
    assert f"abandoned: pid {pid} is no longer running" in message
    assert "rerun to replace state.json" in message


def test_state_without_owner_is_unknown_not_abandoned() -> None:
    state = _running(None)
    assert running_state_liveness(state) == "unknown"
    assert "records no owner" in describe_running_state(state)


def test_malformed_owner_is_unknown() -> None:
    for owner in ("pid", {"pid": True}, {"pid": "12"}, {}):
        assert running_state_liveness(_running(owner)) == "unknown"


def test_owner_on_another_host_is_not_probed(monkeypatch) -> None:
    def fail(pid: int) -> bool:
        raise AssertionError("a pid from another host must not be probed")

    monkeypatch.setattr(run_owner, "_pid_is_alive", fail)
    state = _running({"pid": _dead_pid(), "host_id": "sha256:" + "0" * 64})
    assert running_state_liveness(state) == "unknown"
    assert "on another host" in describe_running_state(state)


def test_unprobeable_pid_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr(run_owner, "_pid_is_alive", lambda pid: None)
    state = _running(current_run_owner())
    assert running_state_liveness(state) == "unknown"
    assert "cannot be checked here" in describe_running_state(state)
