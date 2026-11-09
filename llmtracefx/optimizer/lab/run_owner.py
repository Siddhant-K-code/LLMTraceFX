"""Owner metadata for lab ``state.json`` files.

A writer records the owning process when it marks a run ``running``. A reader
uses that record to tell a live run from one whose process exited without
writing a final status, for example after a crash or ``kill -9``. Readers keep
refusing in every case; only the explanation and recovery hint change.

The host is stored as a hash so ``state.json`` never holds a raw hostname.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Literal

from ..collectors._shared import config_hash

RunLiveness = Literal["active", "abandoned", "unknown"]


def _host_id() -> str:
    return config_hash({"hostname": socket.gethostname()})


def current_run_owner() -> dict[str, Any]:
    return {"pid": os.getpid(), "host_id": _host_id()}


def _pid_is_alive(pid: int) -> bool | None:
    # On Windows os.kill terminates the target, so liveness is not probed there.
    if os.name == "nt" or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _owner_pid(state: dict[str, Any]) -> int | None:
    owner = state.get("owner")
    if not isinstance(owner, dict):
        return None
    pid = owner.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    return pid


def running_state_liveness(state: dict[str, Any]) -> RunLiveness:
    """Classify a ``running`` state as active, abandoned, or unknown.

    A reused pid reads as active, which keeps the reader refusing.
    """
    pid = _owner_pid(state)
    if pid is None:
        return "unknown"
    owner = state["owner"]
    if owner.get("host_id") != _host_id():
        return "unknown"
    alive = _pid_is_alive(pid)
    if alive is None:
        return "unknown"
    return "active" if alive else "abandoned"


def describe_running_state(state: dict[str, Any]) -> str:
    started = state.get("started_at") or "an unknown time"
    pid = _owner_pid(state)
    liveness = running_state_liveness(state)
    if liveness == "active":
        return (
            f"run started at {started} is still active as pid {pid}; "
            "wait for it to finish"
        )
    if liveness == "abandoned":
        return (
            f"run started at {started} was abandoned: pid {pid} is no longer "
            "running and never wrote a final status; rerun to replace state.json"
        )
    if pid is None:
        return (
            f"run started at {started} records no owner, so it may be live or "
            "abandoned; if no run is active, rerun to replace state.json"
        )
    owner = state["owner"]
    if owner.get("host_id") != _host_id():
        return (
            f"run started at {started} is owned by pid {pid} on another host; "
            "check that host before rerunning"
        )
    return (
        f"run started at {started} is owned by pid {pid}, whose liveness "
        "cannot be checked here; if no run is active, rerun to replace state.json"
    )
