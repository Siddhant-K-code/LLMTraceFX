"""Shared state machine for append-only, hash-chained per-call ledgers.

A ledger holds one entry per planned lifecycle. Each entry moves
``planned -> reserved -> completed`` or ``planned -> reserved -> aborted`` and
never moves again. Every transition appends one JSON event that carries a
1-based ``index``, the previous event's ``event_sha256`` and its own
``event_sha256``. The whole document is sealed with ``ledger_sha256``.

This module owns the rules every such ledger shares: the transition table,
the event hash chain, the document seal, the file lock, the replay checks for
event order, chain integrity, identity uniqueness and allowed source states,
and the matching guards used before a live transition is written.

Each ledger keeps its own vocabulary and field checks. It passes a
:class:`LedgerDialect` (error type, identity field name, replay messages) and
per-event check callbacks. The shared code never adds, renames or reorders
persisted fields, so documents and hashes are the same as before it existed.
"""

from __future__ import annotations

import fcntl
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

PLANNED = "planned"
RESERVED = "reserved"
COMPLETED = "completed"
ABORTED = "aborted"

RESERVE = "reserve"
COMPLETE = "complete"
ABORT = "abort"

# event_type -> (required source status, resulting status)
TRANSITIONS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        RESERVE: (PLANNED, RESERVED),
        COMPLETE: (RESERVED, COMPLETED),
        ABORT: (RESERVED, ABORTED),
    }
)

_SETTLE_NOUN = MappingProxyType({COMPLETE: "completion", ABORT: "abort"})


def next_status(status: Any, event_type: Any) -> str | None:
    """Return the status ``event_type`` moves ``status`` to, or None if illegal."""

    rule = TRANSITIONS.get(event_type) if isinstance(event_type, str) else None
    if rule is None or status != rule[0]:
        return None
    return rule[1]


@dataclass(frozen=True)
class ReplayMessages:
    """Exact error messages a ledger raises while replaying its event log."""

    event_not_object: str
    event_index: str
    event_chain: str
    event_hash: str
    event_identity: str
    event_type: str
    reserve_lifecycle: str
    reserve_replayed: str
    reserve_duplicate: str
    complete_unknown: str
    complete_replayed: str
    abort_unknown: str
    abort_replayed: str


@dataclass(frozen=True)
class LedgerDialect:
    """How one ledger spells its identity field, errors and replay messages."""

    error: type[Exception]
    id_field: str
    safe_id: re.Pattern[str]
    sha256_json: Callable[[Any], str]
    messages: ReplayMessages

    def fail(self, message: str) -> NoReturn:
        raise self.error(message)

    @property
    def identity_noun(self) -> str:
        return self.id_field.removesuffix("_id")


ReserveCheck = Callable[
    [Mapping[str, Any], Mapping[str, Any]], tuple[Decimal, Mapping[str, Any]]
]
SettleCheck = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def event_hash(event: Mapping[str, Any], *, sha256_json: Callable[[Any], str]) -> str:
    material = {key: value for key, value in event.items() if key != "event_sha256"}
    return sha256_json(material)


def seal(
    payload: Mapping[str, Any], *, sha256_json: Callable[[Any], str]
) -> dict[str, Any]:
    material = {key: value for key, value in payload.items() if key != "ledger_sha256"}
    material["ledger_sha256"] = sha256_json(material)
    return material


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def next_link(payload: Mapping[str, Any]) -> tuple[int, str | None]:
    """Return the ``index`` and ``previous_event_sha256`` for the next event."""

    events = payload["events"]
    return len(events) + 1, events[-1]["event_sha256"] if events else None


def commit_event(
    payload: dict[str, Any],
    event: dict[str, Any],
    *,
    sha256_json: Callable[[Any], str],
) -> dict[str, Any]:
    """Hash a linked event, append it and bump the document revision."""

    event["event_sha256"] = event_hash(event, sha256_json=sha256_json)
    payload["events"].append(event)
    payload["revision"] += 1
    return event


def apply_transition(
    entry: dict[str, Any],
    event_type: str,
    *,
    dialect: LedgerDialect,
    identity: str,
    updates: Mapping[str, Any],
) -> None:
    """Move ``entry`` along a checked transition and record its fields."""

    entry["status"] = TRANSITIONS[event_type][1]
    if event_type == RESERVE:
        entry[dialect.id_field] = identity
    entry.update(updates)


def require_safe_identity(identity: Any, *, dialect: LedgerDialect) -> str:
    if not isinstance(identity, str) or dialect.safe_id.fullmatch(identity) is None:
        dialect.fail(f"{dialect.id_field} is not a safe stable identity")
    return identity


def reservable_entry(
    payload: Mapping[str, Any],
    *,
    lifecycle_id: str,
    identity: str,
    dialect: LedgerDialect,
) -> dict[str, Any]:
    """Return the planned entry a live ``reserve`` may move, or raise."""

    entry: dict[str, Any] = next(
        item for item in payload["entries"] if item["lifecycle_id"] == lifecycle_id
    )
    if next_status(entry["status"], RESERVE) is None:
        dialect.fail("planned lifecycle is already reserved or completed")
    if any(event[dialect.id_field] == identity for event in payload["events"]):
        dialect.fail(f"{dialect.identity_noun} {identity!r} is already reserved")
    return entry


def settleable_entry(
    payload: Mapping[str, Any],
    *,
    identity: str,
    event_type: str,
    dialect: LedgerDialect,
) -> dict[str, Any]:
    """Return the reserved entry a live ``complete`` or ``abort`` may move."""

    noun = _SETTLE_NOUN[event_type]
    matches: list[dict[str, Any]] = [
        item for item in payload["entries"] if item[dialect.id_field] == identity
    ]
    if len(matches) != 1:
        dialect.fail(f"{noun} {dialect.id_field} is unknown")
    entry = matches[0]
    if next_status(entry["status"], event_type) is None:
        dialect.fail(f"{noun} {dialect.id_field} is not reservable")
    return entry


def replay_events(
    events: Sequence[Any],
    *,
    entries: dict[str, dict[str, Any]],
    dialect: LedgerDialect,
    check_reserve: ReserveCheck,
    check_complete: SettleCheck,
    check_abort: SettleCheck,
) -> tuple[dict[str, dict[str, Any]], Decimal]:
    """Replay an event log onto planned ``entries`` and return them with the
    reserved total.

    Checks, in order, for every event: it is an object, its index, its link
    to the previous event, its own hash, its identity, and that its
    transition is legal from the entry's current status. The check callbacks
    then validate the event's own fields and return the entry fields to
    record; a reserve check also returns the reserved amount.
    """

    messages = dialect.messages
    total = Decimal()
    previous: str | None = None
    owners: dict[str, str] = {}
    for index, raw in enumerate(events, start=1):
        if not isinstance(raw, dict):
            dialect.fail(messages.event_not_object)
        if raw.get("index") != index:
            dialect.fail(messages.event_index)
        if raw.get("previous_event_sha256") != previous:
            dialect.fail(messages.event_chain)
        if raw.get("event_sha256") != event_hash(raw, sha256_json=dialect.sha256_json):
            dialect.fail(messages.event_hash)
        identity = raw.get(dialect.id_field)
        if not isinstance(identity, str) or dialect.safe_id.fullmatch(identity) is None:
            dialect.fail(messages.event_identity)
        event_type = raw.get("event_type")
        if event_type == RESERVE:
            lifecycle_id = raw.get("lifecycle_id")
            if lifecycle_id not in entries:
                dialect.fail(messages.reserve_lifecycle)
            entry = entries[lifecycle_id]
            if next_status(entry["status"], RESERVE) is None:
                dialect.fail(messages.reserve_replayed)
            if identity in owners:
                dialect.fail(messages.reserve_duplicate)
            amount, updates = check_reserve(raw, entry)
            owners[identity] = lifecycle_id
            total += amount
        elif event_type in (COMPLETE, ABORT):
            if event_type == COMPLETE:
                unknown, replayed = (
                    messages.complete_unknown,
                    messages.complete_replayed,
                )
                check = check_complete
            else:
                unknown, replayed = messages.abort_unknown, messages.abort_replayed
                check = check_abort
            if identity not in owners:
                dialect.fail(unknown)
            entry = entries[owners[identity]]
            if next_status(entry["status"], event_type) is None:
                dialect.fail(replayed)
            updates = check(raw, entry)
        else:
            dialect.fail(messages.event_type)
        apply_transition(
            entry, event_type, dialect=dialect, identity=identity, updates=updates
        )
        previous = raw["event_sha256"]
    return entries, total


__all__ = [
    "ABORT",
    "ABORTED",
    "COMPLETE",
    "COMPLETED",
    "PLANNED",
    "RESERVE",
    "RESERVED",
    "TRANSITIONS",
    "LedgerDialect",
    "ReplayMessages",
    "apply_transition",
    "commit_event",
    "event_hash",
    "locked",
    "next_link",
    "next_status",
    "replay_events",
    "require_safe_identity",
    "reservable_entry",
    "seal",
    "settleable_entry",
]
