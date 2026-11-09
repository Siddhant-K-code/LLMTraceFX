from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab import call_ledger as cl
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile


class ToyLedgerError(ValueError):
    pass


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


DIALECT = cl.LedgerDialect(
    error=ToyLedgerError,
    id_field="call_id",
    safe_id=re.compile(r"^[a-z0-9-]+$"),
    sha256_json=_sha256_json,
    messages=cl.ReplayMessages(
        event_not_object="not object",
        event_index="bad index",
        event_chain="broken chain",
        event_hash="bad hash",
        event_identity="bad identity",
        event_type="bad type",
        reserve_lifecycle="unknown lifecycle",
        reserve_replayed="reserve replayed",
        reserve_duplicate="duplicate identity",
        complete_unknown="complete unknown",
        complete_replayed="complete replayed",
        abort_unknown="abort unknown",
        abort_replayed="abort replayed",
    ),
)


def _planned() -> dict[str, dict[str, Any]]:
    return {
        lifecycle_id: {
            "lifecycle_id": lifecycle_id,
            "status": cl.PLANNED,
            "call_id": None,
            "note": None,
        }
        for lifecycle_id in ("a", "b")
    }


def _check_reserve(raw: Any, entry: Any) -> tuple[Decimal, dict[str, Any]]:
    return Decimal(raw["usd"]), {"note": "reserved"}


def _check_settle(raw: Any, entry: Any) -> dict[str, Any]:
    return {"note": raw["event_type"]}


def _chain(*events: dict[str, Any]) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {"events": [], "revision": 0}
    for fields in events:
        index, previous = cl.next_link(payload)
        event = {"index": index, **fields, "previous_event_sha256": previous}
        cl.commit_event(payload, event, sha256_json=_sha256_json)
    assert payload["revision"] == len(events)
    return payload["events"]


def _replay(events: list[Any]) -> tuple[dict[str, dict[str, Any]], Decimal]:
    return cl.replay_events(
        events,
        entries=_planned(),
        dialect=DIALECT,
        check_reserve=_check_reserve,
        check_complete=_check_settle,
        check_abort=_check_settle,
    )


RESERVE_A = {"event_type": "reserve", "call_id": "c1", "lifecycle_id": "a", "usd": "1"}
RESERVE_B = {"event_type": "reserve", "call_id": "c2", "lifecycle_id": "b", "usd": "2"}
COMPLETE_A = {"event_type": "complete", "call_id": "c1"}
ABORT_A = {"event_type": "abort", "call_id": "c1"}
ABORT_B = {"event_type": "abort", "call_id": "c2"}


class TestTransitionTable:
    def test_table_is_the_documented_state_machine(self) -> None:
        assert dict(cl.TRANSITIONS) == {
            "reserve": ("planned", "reserved"),
            "complete": ("reserved", "completed"),
            "abort": ("reserved", "aborted"),
        }
        with pytest.raises(TypeError):
            cl.TRANSITIONS["reset"] = ("completed", "planned")  # type: ignore[index]

    @pytest.mark.parametrize(
        ("status", "event_type", "expected"),
        [
            ("planned", "reserve", "reserved"),
            ("reserved", "complete", "completed"),
            ("reserved", "abort", "aborted"),
        ],
    )
    def test_legal_transitions(
        self, status: str, event_type: str, expected: str
    ) -> None:
        assert cl.next_status(status, event_type) == expected

    @pytest.mark.parametrize(
        ("status", "event_type"),
        [
            ("planned", "complete"),
            ("planned", "abort"),
            ("reserved", "reserve"),
            ("completed", "reserve"),
            ("completed", "complete"),
            ("completed", "abort"),
            ("aborted", "reserve"),
            ("aborted", "complete"),
            ("aborted", "abort"),
            ("planned", "reset"),
            ("planned", None),
            ("planned", ["reserve"]),
        ],
    )
    def test_illegal_transitions_have_no_target(
        self, status: str, event_type: Any
    ) -> None:
        assert cl.next_status(status, event_type) is None

    def test_terminal_states_have_no_outgoing_transition(self) -> None:
        sources = {source for source, _ in cl.TRANSITIONS.values()}
        assert cl.COMPLETED not in sources
        assert cl.ABORTED not in sources


class TestReplay:
    def test_happy_path_applies_states_and_totals(self) -> None:
        entries, total = _replay(_chain(RESERVE_A, RESERVE_B, COMPLETE_A, ABORT_B))
        assert total == Decimal("3")
        assert entries["a"] == {
            "lifecycle_id": "a",
            "status": "completed",
            "call_id": "c1",
            "note": "complete",
        }
        assert entries["b"]["status"] == "aborted"
        assert entries["b"]["note"] == "abort"

    def test_empty_log_keeps_every_entry_planned(self) -> None:
        entries, total = _replay([])
        assert total == Decimal()
        assert {entry["status"] for entry in entries.values()} == {"planned"}

    @pytest.mark.parametrize(
        ("events", "message"),
        [
            ((COMPLETE_A,), "complete unknown"),
            ((ABORT_A,), "abort unknown"),
            ((RESERVE_A, COMPLETE_A, ABORT_A), "abort replayed"),
            ((RESERVE_A, ABORT_A, COMPLETE_A), "complete replayed"),
            ((RESERVE_A, COMPLETE_A, COMPLETE_A), "complete replayed"),
            ((RESERVE_A, RESERVE_A), "reserve replayed"),
            ((RESERVE_A, {**RESERVE_B, "call_id": "c1"}), "duplicate identity"),
            (({**RESERVE_A, "lifecycle_id": "z"},), "unknown lifecycle"),
            (({**RESERVE_A, "event_type": "reset"},), "bad type"),
            (({**RESERVE_A, "call_id": "Bad Id"},), "bad identity"),
            (({**RESERVE_A, "call_id": None},), "bad identity"),
        ],
    )
    def test_illegal_transitions_and_identities_are_refused(
        self, events: tuple[dict[str, Any], ...], message: str
    ) -> None:
        with pytest.raises(ToyLedgerError, match=f"^{message}$"):
            _replay(_chain(*events))

    def test_ordering_chain_and_hash_are_checked_before_content(self) -> None:
        events = _chain(RESERVE_A, COMPLETE_A)
        with pytest.raises(ToyLedgerError, match="^bad index$"):
            _replay([events[1], events[0]])
        with pytest.raises(ToyLedgerError, match="^not object$"):
            _replay(["reserve"])
        broken = [dict(event) for event in events]
        broken[1]["previous_event_sha256"] = None
        with pytest.raises(ToyLedgerError, match="^broken chain$"):
            _replay(broken)
        tampered = [dict(event) for event in events]
        tampered[0]["usd"] = "0.5"
        with pytest.raises(ToyLedgerError, match="^bad hash$"):
            _replay(tampered)

    def test_check_errors_propagate_unchanged(self) -> None:
        def refuse(raw: Any, entry: Any) -> dict[str, Any]:
            raise ToyLedgerError("domain check failed")

        with pytest.raises(ToyLedgerError, match="^domain check failed$"):
            cl.replay_events(
                _chain(RESERVE_A, COMPLETE_A),
                entries=_planned(),
                dialect=DIALECT,
                check_reserve=_check_reserve,
                check_complete=refuse,
                check_abort=_check_settle,
            )


class TestLiveGuards:
    def _payload(self) -> dict[str, Any]:
        events = _chain(RESERVE_A, COMPLETE_A)
        entries, _ = _replay(events)
        return {"entries": list(entries.values()), "events": events}

    def test_reservable_entry(self) -> None:
        payload = self._payload()
        entry = cl.reservable_entry(
            payload, lifecycle_id="b", identity="c2", dialect=DIALECT
        )
        assert entry["lifecycle_id"] == "b"
        with pytest.raises(ToyLedgerError, match="already reserved or completed"):
            cl.reservable_entry(
                payload, lifecycle_id="a", identity="c9", dialect=DIALECT
            )
        with pytest.raises(ToyLedgerError, match="^call 'c1' is already reserved$"):
            cl.reservable_entry(
                payload, lifecycle_id="b", identity="c1", dialect=DIALECT
            )

    def test_settleable_entry(self) -> None:
        payload = self._payload()
        with pytest.raises(ToyLedgerError, match="^completion call_id is unknown$"):
            cl.settleable_entry(
                payload, identity="c9", event_type=cl.COMPLETE, dialect=DIALECT
            )
        with pytest.raises(ToyLedgerError, match="^abort call_id is not reservable$"):
            cl.settleable_entry(
                payload, identity="c1", event_type=cl.ABORT, dialect=DIALECT
            )

    def test_require_safe_identity(self) -> None:
        assert cl.require_safe_identity("c1", dialect=DIALECT) == "c1"
        for bad in ("", "Bad Id", None, 3):
            with pytest.raises(
                ToyLedgerError, match="^call_id is not a safe stable identity$"
            ):
                cl.require_safe_identity(bad, dialect=DIALECT)

    def test_apply_transition_sets_identity_only_on_reserve(self) -> None:
        entry = _planned()["a"]
        cl.apply_transition(
            entry, cl.RESERVE, dialect=DIALECT, identity="c1", updates={"note": "x"}
        )
        assert entry == {
            "lifecycle_id": "a",
            "status": "reserved",
            "call_id": "c1",
            "note": "x",
        }
        cl.apply_transition(
            entry, cl.ABORT, dialect=DIALECT, identity="ignored", updates={}
        )
        assert entry["status"] == "aborted"
        assert entry["call_id"] == "c1"


class TestHashing:
    def test_event_hash_and_seal_exclude_their_own_field(self) -> None:
        event = {"a": 1, "event_sha256": "old"}
        assert cl.event_hash(event, sha256_json=_sha256_json) == _sha256_json({"a": 1})
        sealed = cl.seal({"b": 2, "ledger_sha256": "old"}, sha256_json=_sha256_json)
        assert sealed == {"b": 2, "ledger_sha256": _sha256_json({"b": 2})}

    def test_locked_creates_the_parent_directory(self, tmp_path: Path) -> None:
        lock = tmp_path / "nested" / ".ledger.lock"
        with cl.locked(lock):
            assert lock.exists()


class TestCallersShareTheStateMachine:
    def test_both_ledgers_use_the_shared_module(self) -> None:
        assert modal._locked is cl.locked
        assert vllm_compile._locked is cl.locked
        assert modal._LEDGER_DIALECT.id_field == "call_id"
        assert vllm_compile._LEDGER_DIALECT.id_field == "command_id"

    def test_modal_completed_document_hash_is_pinned(self) -> None:
        document = modal.build_completed_ledger_document(
            plan=modal.build_default_plan(),
            source_head="a" * 40,
            experiment_nonce="b" * 32,
            ledger_path_sha256="sha256:" + "c" * 64,
            reserved_at="2026-09-04T00:00:00+00:00",
            completed_at="2026-09-04T00:10:00+00:00",
        )
        assert document["ledger_sha256"] == (
            "sha256:d1fc78774a7dfef0e193aa622474836b9985feae6b422daffe04c60be050b2e3"
        )

    def test_vllm_event_chain_hashes_are_pinned(self, tmp_path: Path) -> None:
        plan = vllm_compile.build_default_plan()
        first, second = plan.budget_lifecycles[:2]
        ledger = vllm_compile.LifecycleBudgetLedger.initialize(
            tmp_path / "ledger.json",
            plan=plan,
            git_head="a" * 40,
            workspace_path=tmp_path,
        )
        ledger.reserve(
            "cmd-a",
            line_id=first.line_id,
            lifecycle_id=first.lifecycle_id,
            ceiling_usd=first.ceiling_usd,
            argv=("timeout", "10"),
            reserved_at="2026-09-04T00:00:00Z",
        )
        ledger.complete("cmd-a", completed_at="2026-09-04T00:01:00Z", actual_seconds=13)
        ledger.reserve(
            "cmd-b",
            line_id=second.line_id,
            lifecycle_id=second.lifecycle_id,
            ceiling_usd=second.ceiling_usd,
            argv=("timeout", "10"),
            reserved_at="2026-09-04T00:02:00Z",
        )
        ledger.abort("cmd-b", aborted_at="2026-09-04T00:03:00Z", reason="stop")
        assert [event["event_sha256"] for event in ledger.snapshot()["events"]] == [
            "sha256:b5349e48f45dda348e24b4663cf746942eb13257f3406e986aa01bf8d6c0b952",
            "sha256:86dcc26bbc8a8fb26c7925307e588820b24ecac9f50cbc4ef5395cbdff83606f",
            "sha256:0a18791d67976d0949dd91f3f8f5dffa0bfac5782c242c00bf2bd6a161a87b1a",
            "sha256:a8ee26eaff3afa6482c60d1ef48239fdfbfaf533c7e6cd6594ed427c0f09ec3b",
        ]
