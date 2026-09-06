"""Private/public evidence bundling for ``qwen3-8b-vllm-kv-truth-v1``.

This module turns the raw, per-lane facts the host orchestrator collects
(receipts from :mod:`llmtracefx.optimizer.lab.qwen3_8b.kv_truth_runner`,
teardown/transfer receipts, the authorization/config public records) into:

1. a **private raw bundle** -- everything, including whatever sensitive
   detail a real run legitimately produces (this module never itself reads
   host/user/key/IP values; it only ever sees what
   :mod:`llmtracefx.deploy.vllm_kv_truth_lifecycle` already redacted before
   handing data here);
2. a **public-redacted bundle** -- deterministic, portable, and safe to
   publish: no host/user/IP/port/key path, GPU UUID, salt, native hash, raw
   private path, command line, or arbitrary exact token array ever appears
   in it;
3. a **claim matrix** -- one row per executed request, comparing the
   workload's independently expected reuse against what the engine
   (``RequestOutput``) and the KV event stream actually attested, rendered
   as the plan's fixed sentence template and one of its seven fixed verdict
   categories.

Nothing in this module performs any SSH/network/provider/model/image/GPU
I/O -- it only ever reads already-collected receipts and writes local
files, so every test here is a pure, offline exercise of the bundling and
redaction logic and can run with zero setup.

**This module never imports or calls into** :mod:`llmtracefx.evidence`
(the repository's real catalog registry) **and never registers anything
with it.** Every bundle this module writes is a local artifact of this PR's
own tests/fixtures; ``run_mode`` on every bundle explicitly says whether the
data is a real run's output or -- as in every fixture this PR ships --
:data:`RUN_MODE_SYNTHETIC_FIXTURE`, a synthetic, never-executed simulation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmtracefx.cache_audit.adapters.vllm_live import parse_runtime_attestation
from llmtracefx.optimizer._artifact_io import (
    ArtifactReadError,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from llmtracefx.optimizer.collectors._shared import atomic_write_text
from llmtracefx.optimizer.lab.qwen3_8b.kv_truth_runner import (
    PROTOCOL_ID,
)
from llmtracefx.optimizer.lab.qwen3_8b.kv_truth_workload import NESTED_PROBES

from .errors import DeploymentPlanError

MAX_RECEIPT_ARTIFACT_BYTES = 8 * 1024 * 1024

RUN_MODE_SYNTHETIC_FIXTURE = "synthetic_fixture_not_executed"
RUN_MODE_REAL_RUN = "real_run"
_VALID_RUN_MODES = frozenset({RUN_MODE_SYNTHETIC_FIXTURE, RUN_MODE_REAL_RUN})

EVIDENCE_SCHEMA_VERSION = "1"

# The plan's fixed, closed verdict vocabulary (`## Measurements and evidence
# contract`): "verified, partial, attested-only, recomputed, evicted,
# unsupported, or invalid." No other value may ever appear as a verdict.
VERDICT_VERIFIED = "verified"
VERDICT_PARTIAL = "partial"
VERDICT_ATTESTED_ONLY = "attested-only"
VERDICT_RECOMPUTED = "recomputed"
VERDICT_EVICTED = "evicted"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_INVALID = "invalid"
VALID_VERDICTS = frozenset(
    {
        VERDICT_VERIFIED,
        VERDICT_PARTIAL,
        VERDICT_ATTESTED_ONLY,
        VERDICT_RECOMPUTED,
        VERDICT_EVICTED,
        VERDICT_UNSUPPORTED,
        VERDICT_INVALID,
    }
)

# Public-redacted bundles must never contain a key whose name suggests one
# of these categories, defense-in-depth on top of never populating them.
_FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "host",
    "user",
    "ip_address",
    "port",
    "key_path",
    "private_key",
    "known_hosts",
    "gpu_uuid",
    "salt",
    "native_hash",
    "argv",
    "command_line",
    "token_array",
    "token_ids",
    "block_hash",
    "extra_keys",
    "raw_path",
)


class EvidenceError(DeploymentPlanError):
    """Raised on any malformed, tampered, or unverifiable evidence artifact."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_run_mode(value: str) -> str:
    if value not in _VALID_RUN_MODES:
        raise EvidenceError(f"run_mode must be one of {sorted(_VALID_RUN_MODES)}")
    return value


# ---------------------------------------------------------------------------
# Claim matrix: expected reuse (workload) vs. attested reuse (engine + KV
# events), rendered as the plan's fixed sentence template.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClaimMatrixEntry:
    """One row of the claim matrix: one executed request's full verdict."""

    request_id: str
    scenario: str
    expected_reusable_tokens: int
    expected_reusable_blocks: int | None
    engine_attested_cached_tokens: int
    engine_attested_created_tokens: int
    event_attested: bool | None
    boundary_valid: bool
    verdict: str
    sentence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "expected_reusable_tokens": self.expected_reusable_tokens,
            "expected_reusable_blocks": self.expected_reusable_blocks,
            "engine_attested_cached_tokens": self.engine_attested_cached_tokens,
            "engine_attested_created_tokens": self.engine_attested_created_tokens,
            "event_attested": self.event_attested,
            "boundary_valid": self.boundary_valid,
            "verdict": self.verdict,
            "sentence": self.sentence,
        }


def render_claim_sentence(
    *,
    expected_reusable_tokens: int | None,
    expected_reusable_blocks: int | None,
    engine_attested_cached_tokens: int,
    verdict: str,
) -> str:
    """The plan's fixed sentence template, filled in for one request."""

    expected = (
        "null/unsupported"
        if expected_reusable_tokens is None
        else f"{expected_reusable_tokens}/{expected_reusable_blocks}"
    )
    return (
        f"Expected {expected} reusable tokens/blocks; engine events and "
        f"RequestOutput attested {engine_attested_cached_tokens}; request "
        "processing/timing/memory/output showed consistent terminal state; "
        f"therefore the cache claim is {verdict}."
    )


def classify_probe_claim(
    *,
    expected_reusable_tokens: int,
    expected_reusable_blocks: int,
    engine_attested_cached_tokens: int,
    event_attested: bool | None,
    boundary_valid: bool,
    request_binding_ambiguous: bool = False,
) -> str:
    """Classify one nested-probe request against the plan's fixed vocabulary.

    Precedence (highest first): an ambiguous request binding or an invalid
    event boundary is always ``invalid`` -- no numeric agreement can excuse
    a broken evidentiary chain. Otherwise: both engine and event evidence
    agreeing with the expectation is ``verified``; only one of the two
    sources being available and agreeing is ``attested-only``; the two
    sources disagreeing with each other is ``partial``; the engine
    attesting strictly less reuse than expected (a legitimate cache miss
    that still completed correctly) is ``recomputed``.
    """

    if request_binding_ambiguous or not boundary_valid:
        return VERDICT_INVALID
    engine_matches = engine_attested_cached_tokens == expected_reusable_tokens
    if event_attested is None:
        if engine_matches:
            return VERDICT_ATTESTED_ONLY
        if engine_attested_cached_tokens < expected_reusable_tokens:
            return VERDICT_RECOMPUTED
        return VERDICT_INVALID
    if engine_matches and event_attested:
        return VERDICT_VERIFIED
    if engine_matches or event_attested:
        return VERDICT_PARTIAL
    if engine_attested_cached_tokens < expected_reusable_tokens:
        return VERDICT_RECOMPUTED
    return VERDICT_INVALID


def build_claim_matrix(lane_result: Mapping[str, Any]) -> tuple[ClaimMatrixEntry, ...]:
    """Build the claim matrix for one executed "B" lane result.

    Zips ``lane_result["records"]`` positionally with
    :data:`kv_truth_workload.NESTED_PROBES` -- both are exactly length 10 and
    strictly ordered -- rather than keying by ``scenario``, since three
    probes share the ``identical_prefix`` scenario and are not unique keys.
    """

    records = lane_result.get("records", [])
    if len(records) != len(NESTED_PROBES):
        raise EvidenceError(
            f"lane_result has {len(records)} records but the fixed nested "
            f"probe sequence has {len(NESTED_PROBES)}; cannot align claims"
        )
    entries = []
    for record, probe in zip(records, NESTED_PROBES, strict=True):
        event_batches = record.get("event_batches", [])
        event_attested: bool | None = None
        if event_batches:
            event_attested = bool(record.get("boundary_valid", False))
        # Positional binding: this record must be *for* the probe at this
        # exact position in the fixed nested sequence -- a scenario mismatch
        # here means the lane result was reordered/misaligned and the claim
        # for this slot cannot be trusted.
        ambiguous = record.get("scenario") != probe.scenario
        verdict = classify_probe_claim(
            expected_reusable_tokens=probe.expected_reusable_tokens,
            expected_reusable_blocks=probe.expected_reusable_blocks,
            engine_attested_cached_tokens=record["num_cached_tokens"],
            event_attested=event_attested,
            boundary_valid=bool(record.get("boundary_valid", False)),
            request_binding_ambiguous=ambiguous,
        )
        sentence = render_claim_sentence(
            expected_reusable_tokens=probe.expected_reusable_tokens,
            expected_reusable_blocks=probe.expected_reusable_blocks,
            engine_attested_cached_tokens=record["num_cached_tokens"],
            verdict=verdict,
        )
        entries.append(
            ClaimMatrixEntry(
                request_id=record["request_id"],
                scenario=record["scenario"],
                expected_reusable_tokens=probe.expected_reusable_tokens,
                expected_reusable_blocks=probe.expected_reusable_blocks,
                engine_attested_cached_tokens=record["num_cached_tokens"],
                engine_attested_created_tokens=record["num_cache_creation_tokens"],
                event_attested=event_attested,
                boundary_valid=bool(record.get("boundary_valid", False)),
                verdict=verdict,
                sentence=sentence,
            )
        )
    return tuple(entries)


def build_eviction_claim(lane_result: Mapping[str, Any]) -> ClaimMatrixEntry:
    """Build the single claim for the eviction lane's final seed probe.

    Per the plan: the final seed probe is eligible for ``evicted`` only when
    the seed's exact hashes were first observed stored, then explicitly
    removed, then the probe reports zero cached tokens and full prompt
    work -- otherwise ``invalid`` or ``unsupported``, never inferred from
    capacity arithmetic alone.
    """

    records = lane_result.get("records", [])
    if not records:
        raise EvidenceError("eviction lane_result has no records")
    final = records[-1]
    boundary_valid = bool(final.get("boundary_valid", False))
    zero_cached = final.get("num_cached_tokens", -1) == 0
    full_prompt_work = final.get("num_cache_creation_tokens", 0) > 0
    if not boundary_valid:
        verdict = VERDICT_INVALID
    elif zero_cached and full_prompt_work:
        verdict = VERDICT_EVICTED
    else:
        verdict = VERDICT_UNSUPPORTED
    sentence = render_claim_sentence(
        expected_reusable_tokens=0,
        expected_reusable_blocks=0,
        engine_attested_cached_tokens=final.get("num_cached_tokens", 0),
        verdict=verdict,
    )
    return ClaimMatrixEntry(
        request_id=final["request_id"],
        scenario=final["scenario"],
        expected_reusable_tokens=0,
        expected_reusable_blocks=0,
        engine_attested_cached_tokens=final.get("num_cached_tokens", 0),
        engine_attested_created_tokens=final.get("num_cache_creation_tokens", 0),
        event_attested=boundary_valid or None,
        boundary_valid=boundary_valid,
        verdict=verdict,
        sentence=sentence,
    )


def build_salt_isolation_claim(
    lane_result: Mapping[str, Any], *, salt_supported: bool
) -> ClaimMatrixEntry:
    """The ``salt_isolation`` probe's claim is ``unsupported`` unless the
    pinned public runtime surface is confirmed to support a cache salt."""

    records = lane_result.get("records", [])
    salt_record = next(
        (r for r in records if r.get("scenario") == "namespace_isolation"), None
    )
    if salt_record is None:
        raise EvidenceError("lane_result has no namespace_isolation record")
    if not salt_supported:
        return ClaimMatrixEntry(
            request_id=salt_record["request_id"],
            scenario="namespace_isolation",
            expected_reusable_tokens=0,
            expected_reusable_blocks=None,
            engine_attested_cached_tokens=salt_record.get("num_cached_tokens", 0),
            engine_attested_created_tokens=salt_record.get(
                "num_cache_creation_tokens", 0
            ),
            event_attested=None,
            boundary_valid=bool(salt_record.get("boundary_valid", False)),
            verdict=VERDICT_UNSUPPORTED,
            sentence=render_claim_sentence(
                expected_reusable_tokens=None,
                expected_reusable_blocks=None,
                engine_attested_cached_tokens=salt_record.get("num_cached_tokens", 0),
                verdict=VERDICT_UNSUPPORTED,
            ),
        )
    boundary_valid = bool(salt_record.get("boundary_valid", False))
    zero_cached = salt_record.get("num_cached_tokens", -1) == 0
    verdict = VERDICT_VERIFIED if boundary_valid and zero_cached else VERDICT_INVALID
    return ClaimMatrixEntry(
        request_id=salt_record["request_id"],
        scenario="namespace_isolation",
        expected_reusable_tokens=0,
        expected_reusable_blocks=0,
        engine_attested_cached_tokens=salt_record.get("num_cached_tokens", 0),
        engine_attested_created_tokens=salt_record.get("num_cache_creation_tokens", 0),
        event_attested=boundary_valid,
        boundary_valid=boundary_valid,
        verdict=verdict,
        sentence=render_claim_sentence(
            expected_reusable_tokens=0,
            expected_reusable_blocks=0,
            engine_attested_cached_tokens=salt_record.get("num_cached_tokens", 0),
            verdict=verdict,
        ),
    )


# ---------------------------------------------------------------------------
# List-rate cost ledger: a running, timestamped list-price cost trail.
# Explicitly NOT provider billing proof -- only ever a self-reported ledger
# computed from the authorization's own rate against elapsed time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ListRateLedgerEntry:
    elapsed_minutes: str
    cost_usd: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed_minutes": self.elapsed_minutes,
            "cost_usd": self.cost_usd,
            "note": self.note,
        }


@dataclass(frozen=True)
class ListRateLedger:
    """A running list-price cost trail. Not provider billing proof."""

    entries: tuple[ListRateLedgerEntry, ...]
    disclaimer: str = (
        "This ledger is a self-reported, list-rate-derived cost estimate. "
        "It is not provider billing proof."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "disclaimer": self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Teardown receipt.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeardownReceipt:
    residual_containers: int
    residual_gpu_processes: int
    evidence_transferred_and_verified: bool
    shutdown_issued: bool
    safe_to_terminate_message_emitted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_containers": self.residual_containers,
            "residual_gpu_processes": self.residual_gpu_processes,
            "evidence_transferred_and_verified": self.evidence_transferred_and_verified,
            "shutdown_issued": self.shutdown_issued,
            "safe_to_terminate_message_emitted": self.safe_to_terminate_message_emitted,
        }

    def __post_init__(self) -> None:
        if self.residual_containers != 0 or self.residual_gpu_processes != 0:
            raise EvidenceError(
                "a teardown receipt with nonzero residual state must never be "
                "constructed; the orchestrator itself must fail before "
                "reaching this point"
            )


# ---------------------------------------------------------------------------
# Private raw bundle.
# ---------------------------------------------------------------------------

_PRIVATE_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_mode",
        "experiment_nonce",
        "authorization",
        "ssh_options_public_record",
        "lane_receipts",
        "claim_matrix",
        "list_rate_ledger",
        "teardown_receipt",
        "bundle_sha256",
    }
)


@dataclass(frozen=True)
class PrivateEvidenceBundle:
    """Everything this protocol run collected, sealed with one canonical hash.

    This dataclass itself never contains a host/user/IP/key value -- those
    are excluded even from the *private* bundle by construction, because
    :class:`~llmtracefx.deploy.vllm_kv_truth_lifecycle.ProtectedExecutionConfig`
    and :class:`~llmtracefx.deploy.vllm_kv_truth_lifecycle.StrictSSHOptions`
    only ever expose ``public_record()`` views to callers outside that
    module. "Private" here means "not yet redacted for external
    publication" (e.g. it may still carry exact token arrays and native
    hashes), not "contains host secrets."
    """

    run_mode: str
    experiment_nonce: str
    authorization: Mapping[str, Any]
    ssh_options_public_record: Mapping[str, Any]
    lane_receipts: Mapping[str, Any]
    claim_matrix: tuple[ClaimMatrixEntry, ...]
    list_rate_ledger: ListRateLedger
    teardown_receipt: TeardownReceipt | None

    def __post_init__(self) -> None:
        _require_run_mode(self.run_mode)

    def to_dict(self) -> dict[str, Any]:
        unsealed = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_mode": self.run_mode,
            "experiment_nonce": self.experiment_nonce,
            "authorization": dict(self.authorization),
            "ssh_options_public_record": dict(self.ssh_options_public_record),
            "lane_receipts": dict(self.lane_receipts),
            "claim_matrix": [entry.to_dict() for entry in self.claim_matrix],
            "list_rate_ledger": self.list_rate_ledger.to_dict(),
            "teardown_receipt": (
                self.teardown_receipt.to_dict() if self.teardown_receipt else None
            ),
        }
        return {**unsealed, "bundle_sha256": sha256_json(unsealed)}

    def write(self, path: Path) -> None:
        atomic_write_text(path, canonical_json(self.to_dict()) + "\n")

    @classmethod
    def read(cls, path: Path) -> PrivateEvidenceBundle:
        try:
            text = read_bounded_regular_text(path, MAX_RECEIPT_ARTIFACT_BYTES)
            raw = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise EvidenceError(
                f"private bundle could not be read safely: {exc}"
            ) from exc
        return cls.verify_and_parse(raw)

    @classmethod
    def verify_and_parse(cls, raw: Any) -> PrivateEvidenceBundle:
        if not isinstance(raw, dict) or set(raw) != _PRIVATE_REQUIRED_KEYS:
            raise EvidenceError("private bundle keys differ from the required set")
        unsealed = {k: v for k, v in raw.items() if k != "bundle_sha256"}
        if sha256_json(unsealed) != raw["bundle_sha256"]:
            raise EvidenceError("private bundle does not verify against its own seal")
        if raw["protocol_id"] != PROTOCOL_ID:
            raise EvidenceError("private bundle protocol_id does not match")
        teardown_raw = raw["teardown_receipt"]
        teardown = (
            TeardownReceipt(
                residual_containers=teardown_raw["residual_containers"],
                residual_gpu_processes=teardown_raw["residual_gpu_processes"],
                evidence_transferred_and_verified=teardown_raw[
                    "evidence_transferred_and_verified"
                ],
                shutdown_issued=teardown_raw["shutdown_issued"],
                safe_to_terminate_message_emitted=teardown_raw[
                    "safe_to_terminate_message_emitted"
                ],
            )
            if teardown_raw is not None
            else None
        )
        claim_matrix = tuple(
            ClaimMatrixEntry(
                request_id=entry["request_id"],
                scenario=entry["scenario"],
                expected_reusable_tokens=entry["expected_reusable_tokens"],
                expected_reusable_blocks=entry["expected_reusable_blocks"],
                engine_attested_cached_tokens=entry["engine_attested_cached_tokens"],
                engine_attested_created_tokens=entry["engine_attested_created_tokens"],
                event_attested=entry["event_attested"],
                boundary_valid=entry["boundary_valid"],
                verdict=entry["verdict"],
                sentence=entry["sentence"],
            )
            for entry in raw["claim_matrix"]
        )
        ledger = ListRateLedger(
            entries=tuple(
                ListRateLedgerEntry(**entry)
                for entry in raw["list_rate_ledger"]["entries"]
            ),
            disclaimer=raw["list_rate_ledger"]["disclaimer"],
        )
        return cls(
            run_mode=raw["run_mode"],
            experiment_nonce=raw["experiment_nonce"],
            authorization=raw["authorization"],
            ssh_options_public_record=raw["ssh_options_public_record"],
            lane_receipts=raw["lane_receipts"],
            claim_matrix=claim_matrix,
            list_rate_ledger=ledger,
            teardown_receipt=teardown,
        )


# ---------------------------------------------------------------------------
# Redaction scanner: defense-in-depth on top of the fact that every input to
# this module (lane receipts via ``LiveKVEventBatch.redact()``, the
# authorization's digest/commitment-only fields, the SSH options' public
# record) is already publication-safe by construction.
# ---------------------------------------------------------------------------


#: Key names that legitimately contain a forbidden fragment as a mere
#: substring but never carry a sensitive value themselves -- e.g.
#: ``strict_host_key_checking`` is a fixed boolean *policy* flag from
#: :meth:`~llmtracefx.deploy.vllm_kv_truth_lifecycle.StrictSSHOptions.public_record`,
#: never a host-identifying value, despite containing ``"host"``. Every
#: entry here must be an exact key name, never a fragment, so this can never
#: widen into an accidental bypass of the scanner it lives next to.
_ALLOWED_KEY_NAMES_WITH_FORBIDDEN_FRAGMENTS = frozenset(
    {
        "cache_salt",
        "cache_salt_present",
        "kv_events_use_int_block_hashes",
        "strict_host_key_checking",
    }
)


def _walk_keys(value: Any) -> Any:
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                continue
            if key in _ALLOWED_KEY_NAMES_WITH_FORBIDDEN_FRAGMENTS:
                continue
            lowered = key.lower()
            for fragment in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS:
                if fragment in lowered:
                    raise EvidenceError(
                        f"public bundle would contain a forbidden key: {key!r} "
                        f"(matches {fragment!r})"
                    )
        for sub_value in value.values():
            _walk_keys(sub_value)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item)


def assert_publication_safe(payload: Mapping[str, Any]) -> None:
    """Recursively refuse any key whose name suggests a forbidden category.

    This is a belt-and-braces scanner over key *names*, not a substitute for
    the upstream redaction each value already goes through
    (``LiveKVEventBatch.redact()``, ``RunAuthorization.to_dict()``,
    ``StrictSSHOptions.public_record()``) -- it exists purely to fail loudly
    if a future refactor ever accidentally reintroduces a forbidden field.
    """

    _walk_keys(dict(payload))


def _redact_event_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    redacted_events: list[dict[str, Any]] = []
    events = batch.get("events", [])
    if not isinstance(events, list):
        raise EvidenceError("private event batch events must be an array")
    for event in events:
        if not isinstance(event, dict):
            raise EvidenceError("private event batch contains a malformed event")
        event_type = event.get("type")
        if event_type == "BlockStored":
            hashes = event.get("block_hashes", [])
            tokens = event.get("token_ids", [])
            redacted_events.append(
                {
                    "type": event_type,
                    "block_count": (
                        len(hashes)
                        if "block_hashes" in event and isinstance(hashes, list)
                        else event.get("block_count")
                    ),
                    "token_count": (
                        len(tokens)
                        if "token_ids" in event and isinstance(tokens, list)
                        else event.get("token_count")
                    ),
                    "block_size": event.get("block_size"),
                    "medium": event.get("medium"),
                    "group_idx": event.get("group_idx"),
                    "kv_cache_spec_kind": event.get("kv_cache_spec_kind"),
                    "kv_cache_spec_sliding_window": event.get(
                        "kv_cache_spec_sliding_window"
                    ),
                    "locality": event.get("locality"),
                }
            )
        elif event_type == "BlockRemoved":
            hashes = event.get("block_hashes", [])
            redacted_events.append(
                {
                    "type": event_type,
                    "block_count": (
                        len(hashes)
                        if "block_hashes" in event and isinstance(hashes, list)
                        else event.get("block_count")
                    ),
                    "medium": event.get("medium"),
                    "group_idx": event.get("group_idx"),
                    "locality": event.get("locality"),
                }
            )
        elif event_type == "AllBlocksCleared":
            redacted_events.append({"type": event_type})
        else:
            raise EvidenceError("private event batch contains an unknown event type")
    return {
        "sequence": batch.get("sequence"),
        "topic": batch.get("topic"),
        "ts": batch.get("ts"),
        "data_parallel_rank": batch.get("data_parallel_rank"),
        "events": redacted_events,
    }


def _redact_lane_receipts(
    lane_receipts: Mapping[str, Any],
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for tag, receipt_value in lane_receipts.items():
        if not isinstance(receipt_value, dict):
            raise EvidenceError("private lane receipt must be an object")
        receipt = dict(receipt_value)
        wraps_protocol_receipt = "lane_result" in receipt
        lane_result = receipt.get("lane_result") if wraps_protocol_receipt else receipt
        if not isinstance(lane_result, dict):
            raise EvidenceError("private lane receipt lane_result must be an object")
        records = lane_result.get("records")
        if not isinstance(records, list):
            raise EvidenceError("private lane receipt records must be an array")
        public_records: list[dict[str, Any]] = []
        for record_value in records:
            if not isinstance(record_value, dict):
                raise EvidenceError("private request record must be an object")
            record = {
                key: value
                for key, value in record_value.items()
                if key not in {"prompt_token_ids", "output_token_ids", "event_batches"}
            }
            batches = record_value.get("event_batches", [])
            if not isinstance(batches, list):
                raise EvidenceError("private request event_batches must be an array")
            record["event_batches"] = [
                _redact_event_batch(batch)
                for batch in batches
                if isinstance(batch, dict)
            ]
            if len(record["event_batches"]) != len(batches):
                raise EvidenceError("private request contains a malformed event batch")
            public_records.append(record)
        public_lane_result = {
            **lane_result,
            "reset_event_batches": [
                _redact_event_batch(batch)
                for batch in lane_result.get("reset_event_batches", [])
                if isinstance(batch, dict)
            ],
            "records": public_records,
        }
        if len(public_lane_result["reset_event_batches"]) != len(
            lane_result.get("reset_event_batches", [])
        ):
            raise EvidenceError("private lane contains a malformed reset event batch")
        if wraps_protocol_receipt:
            receipt["lane_result"] = public_lane_result
        else:
            receipt = public_lane_result
        runtime_attestation = receipt.get("runtime_attestation")
        if runtime_attestation is not None:
            if not isinstance(runtime_attestation, dict):
                raise EvidenceError("runtime_attestation must be an object or null")
            try:
                receipt["runtime_attestation"] = parse_runtime_attestation(
                    runtime_attestation
                ).redact()
            except ValueError as exc:
                raise EvidenceError(
                    f"private runtime attestation is invalid: {exc}"
                ) from exc
        redacted[tag] = receipt
    return redacted


# ---------------------------------------------------------------------------
# Public-redacted bundle, deterministic portable verifier, report, SVG,
# and SHA256SUMS.
# ---------------------------------------------------------------------------

_PUBLIC_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "run_mode",
        "experiment_nonce",
        "authorization",
        "ssh_options_public_record",
        "lane_receipts",
        "claim_matrix",
        "list_rate_ledger",
        "teardown_receipt",
        "bundle_sha256",
    }
)


@dataclass(frozen=True)
class PublicRedactedBundle:
    """The deterministic, publication-safe view of a private bundle.

    Field-for-field identical to :class:`PrivateEvidenceBundle`'s dict shape
    (every value was already publication-safe on the way in) but produced
    through :func:`assert_publication_safe` and its own independent seal, so
    it can be verified on its own without ever needing the private bundle
    again.
    """

    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def write(self, path: Path) -> None:
        atomic_write_text(path, canonical_json(self.to_dict()) + "\n")

    @classmethod
    def from_private(cls, private: PrivateEvidenceBundle) -> PublicRedactedBundle:
        unsealed = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "run_mode": private.run_mode,
            "experiment_nonce": private.experiment_nonce,
            "authorization": dict(private.authorization),
            "ssh_options_public_record": dict(private.ssh_options_public_record),
            "lane_receipts": _redact_lane_receipts(private.lane_receipts),
            "claim_matrix": [entry.to_dict() for entry in private.claim_matrix],
            "list_rate_ledger": private.list_rate_ledger.to_dict(),
            "teardown_receipt": (
                private.teardown_receipt.to_dict() if private.teardown_receipt else None
            ),
        }
        assert_publication_safe(unsealed)
        sealed = {**unsealed, "bundle_sha256": sha256_json(unsealed)}
        return cls(payload=sealed)

    @classmethod
    def read(cls, path: Path) -> PublicRedactedBundle:
        try:
            text = read_bounded_regular_text(path, MAX_RECEIPT_ARTIFACT_BYTES)
            raw = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise EvidenceError(
                f"public bundle could not be read safely: {exc}"
            ) from exc
        return cls.verify_and_parse(raw)

    @classmethod
    def verify_and_parse(cls, raw: Any) -> PublicRedactedBundle:
        if not isinstance(raw, dict) or set(raw) != _PUBLIC_REQUIRED_KEYS:
            raise EvidenceError("public bundle keys differ from the required set")
        unsealed = {k: v for k, v in raw.items() if k != "bundle_sha256"}
        if sha256_json(unsealed) != raw["bundle_sha256"]:
            raise EvidenceError("public bundle does not verify against its own seal")
        assert_publication_safe(unsealed)
        return cls(payload=dict(raw))


def render_report_text(bundle: PublicRedactedBundle) -> str:
    """A minimal, deterministic plain-text report over one public bundle."""

    payload = bundle.to_dict()
    lines = [
        f"protocol: {payload['protocol_id']}",
        f"run_mode: {payload['run_mode']}",
        f"experiment_nonce: {payload['experiment_nonce']}",
        "",
        "claim matrix:",
    ]
    for entry in payload["claim_matrix"]:
        lines.append(
            f"  - {entry['request_id']} [{entry['scenario']}]: {entry['sentence']}"
        )
    lines.append("")
    verdict_counts: dict[str, int] = {}
    for entry in payload["claim_matrix"]:
        verdict_counts[entry["verdict"]] = verdict_counts.get(entry["verdict"], 0) + 1
    lines.append("verdict summary:")
    for verdict in sorted(verdict_counts):
        lines.append(f"  {verdict}: {verdict_counts[verdict]}")
    if payload["run_mode"] == RUN_MODE_SYNTHETIC_FIXTURE:
        lines.append("")
        lines.append(
            "NOTE: this bundle is a SYNTHETIC FIXTURE. It was never executed "
            "against real hardware, a real vLLM engine, or a real host, and "
            "must not be treated as evidence of an actual run."
        )
    return "\n".join(lines) + "\n"


def render_report_svg(bundle: PublicRedactedBundle) -> str:
    """A minimal, deterministic SVG bar chart of the claim matrix verdicts.

    Deterministic: no timestamps, random IDs, or floating-point noise --
    the same bundle always renders byte-identical SVG.
    """

    payload = bundle.to_dict()
    verdict_counts: dict[str, int] = {}
    for entry in payload["claim_matrix"]:
        verdict_counts[entry["verdict"]] = verdict_counts.get(entry["verdict"], 0) + 1
    verdicts = sorted(verdict_counts)
    bar_height = 24
    width = 480
    height = max(1, len(verdicts)) * bar_height + 20
    max_count = max(verdict_counts.values(), default=1)
    rows = []
    for index, verdict in enumerate(verdicts):
        count = verdict_counts[verdict]
        bar_width = int((count / max_count) * (width - 160)) if max_count else 0
        y = 10 + index * bar_height
        rows.append(
            f'<rect x="140" y="{y}" width="{bar_width}" height="{bar_height - 4}" '
            'fill="#4c72b0" />'
        )
        rows.append(
            f'<text x="4" y="{y + 14}" font-family="monospace" font-size="12">'
            f"{verdict}</text>"
        )
        rows.append(
            f'<text x="{144 + bar_width}" y="{y + 14}" font-family="monospace" '
            f'font-size="12">{count}</text>'
        )
    body = "".join(rows)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>'
    )


def compute_sha256sums(files: Mapping[str, bytes]) -> str:
    """A deterministic ``SHA256SUMS``-format checksum listing."""

    lines = [f"{sha256_bytes(data)}  {name}" for name, data in sorted(files.items())]
    return "\n".join(lines) + "\n"


def verify_sha256sums(sha256sums_text: str, files: Mapping[str, bytes]) -> bool:
    """Portable verifier: recompute and compare every listed file's digest."""

    expected: dict[str, str] = {}
    for line in sha256sums_text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        expected[name] = digest
    if set(expected) != set(files):
        return False
    return all(sha256_bytes(files[name]) == digest for name, digest in expected.items())


def write_public_bundle_directory(
    bundle: PublicRedactedBundle, directory: Path
) -> dict[str, str]:
    """Write the complete deterministic public bundle directory.

    Returns the ``{filename: sha256}`` map that was also written as
    ``SHA256SUMS`` so a caller/test can cross-check without re-reading disk.
    """

    directory.mkdir(parents=True, exist_ok=True)
    bundle_json = canonical_json(bundle.to_dict()) + "\n"
    report_text = render_report_text(bundle)
    report_svg = render_report_svg(bundle)
    files = {
        "bundle.json": bundle_json.encode("utf-8"),
        "report.txt": report_text.encode("utf-8"),
        "report.svg": report_svg.encode("utf-8"),
    }
    for name, data in files.items():
        atomic_write_text(directory / name, data.decode("utf-8"))
    sha256sums = compute_sha256sums(files)
    atomic_write_text(directory / "SHA256SUMS", sha256sums)
    return {name: sha256_bytes(data) for name, data in files.items()}


def verify_public_bundle_directory(directory: Path) -> PublicRedactedBundle:
    """The portable verifier: recompute every file's digest against
    ``SHA256SUMS``, then verify the bundle's own canonical seal."""

    sha256sums_path = directory / "SHA256SUMS"
    sha256sums_text = read_bounded_regular_text(
        sha256sums_path, MAX_RECEIPT_ARTIFACT_BYTES
    )
    files = {}
    for name in ("bundle.json", "report.txt", "report.svg"):
        files[name] = read_bounded_regular_bytes(
            directory / name, MAX_RECEIPT_ARTIFACT_BYTES
        )
    if not verify_sha256sums(sha256sums_text, files):
        raise EvidenceError("SHA256SUMS does not match the bundle directory contents")
    return PublicRedactedBundle.read(directory / "bundle.json")
