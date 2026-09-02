"""Fail-closed application-state budget accounting for hosted API requests.

The budget plan is an immutable input. The ledger records one irreversible
claim before each network attempt and settles that claim only when provider
usage proves a smaller charge. A failed or unaccounted request keeps its full
planned ceiling, and the same request ID can never be claimed twice.

The separate anchor detects accidental deletion, movement, partial writes and
one-sided rollback. Like every user-writable local file, it is not a defense
against a malicious operator who deletes or restores both ledger and anchor;
that threat requires a provider-side spending limit or external monotonic
service.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text, sha256_text
from ..collectors.openai_api import ProviderUsage
from ..schema import utc_now_iso

BUDGET_PLAN_SCHEMA_VERSION = "3"
BUDGET_LEDGER_SCHEMA_VERSION = "3"
BUDGET_ANCHOR_SCHEMA_VERSION = "1"
DEFAULT_HARD_LIMIT_USD = Decimal("5.00")
_MONEY_PLACES = Decimal("0.000000000001")
_LEDGER_SEAL_FIELD = "ledger_sha256"
_PLAN_SEAL_FIELD = "plan_sha256"
_ANCHOR_SEAL_FIELD = "anchor_sha256"
_REQUEST_STATUSES = frozenset({"planned", "attempted", "completed", "failed"})
_LEDGER_IDENTITY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$")


class BudgetError(ValueError):
    """Raised when a request cannot be proven to remain inside its budget."""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _money(value: Decimal) -> str:
    return format(value.quantize(_MONEY_PLACES), "f")


def _decimal(value: Any, *, context: str, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not value:
        raise BudgetError(f"{context} must be a non-empty decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise BudgetError(f"{context} must be a decimal string") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        relation = "> 0" if positive else ">= 0"
        raise BudgetError(f"{context} must be finite and {relation}")
    return result


def _string(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise BudgetError(f"{context}.{key} must be a non-empty string")
    return value


def _positive_int(data: dict[str, Any], key: str, *, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BudgetError(f"{context}.{key} must be a positive integer")
    return value


@dataclass(frozen=True)
class BudgetRequest:
    """One pre-authorized request and its conservative token-price ceiling."""

    request_id: str
    model_id: str
    workload_id: str
    workload_version: str
    prompt_sha256: str
    request_config_sha256: str
    endpoint_origin: str
    endpoint_path: str
    route_providers: tuple[str, ...]
    allow_fallbacks: bool
    require_parameters: bool
    max_provider_prompt_price_per_million: Decimal
    max_provider_completion_price_per_million: Decimal
    reasoning_effort: str
    input_token_ceiling: int
    max_output_tokens: int
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    cached_prompt_usd_per_token: Decimal
    cache_write_billing: str
    reasoning_billing: str

    @property
    def ceiling_usd(self) -> Decimal:
        # Cache reads are cheaper for the supported manifests, but the ceiling
        # prices every input token at the full prompt rate. Cache writes are
        # explicitly required to be covered by that same rate.
        input_rate = max(self.prompt_usd_per_token, self.cached_prompt_usd_per_token)
        return (
            Decimal(self.input_token_ceiling) * input_rate
            + Decimal(self.max_output_tokens) * self.completion_usd_per_token
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "prompt_sha256": self.prompt_sha256,
            "request_config_sha256": self.request_config_sha256,
            "endpoint_origin": self.endpoint_origin,
            "endpoint_path": self.endpoint_path,
            "route_providers": list(self.route_providers),
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "max_provider_prompt_price_per_million": str(
                self.max_provider_prompt_price_per_million
            ),
            "max_provider_completion_price_per_million": str(
                self.max_provider_completion_price_per_million
            ),
            "reasoning_effort": self.reasoning_effort,
            "input_token_ceiling": self.input_token_ceiling,
            "max_output_tokens": self.max_output_tokens,
            "prompt_usd_per_token": str(self.prompt_usd_per_token),
            "completion_usd_per_token": str(self.completion_usd_per_token),
            "cached_prompt_usd_per_token": str(self.cached_prompt_usd_per_token),
            "cache_write_billing": self.cache_write_billing,
            "reasoning_billing": self.reasoning_billing,
            "ceiling_usd": _money(self.ceiling_usd),
        }

    @classmethod
    def from_dict(cls, data: Any, *, index: int) -> BudgetRequest:
        context = f"budget plan request[{index}]"
        if not isinstance(data, dict):
            raise BudgetError(f"{context} must be an object")
        cache_write_billing = _string(data, "cache_write_billing", context=context)
        if cache_write_billing != "included_in_uncached_prompt_rate":
            raise BudgetError(
                f"{context}.cache_write_billing must explicitly state "
                "'included_in_uncached_prompt_rate'; unknown cache-write "
                "billing cannot be assumed free"
            )
        reasoning_billing = _string(data, "reasoning_billing", context=context)
        if reasoning_billing != "included_in_completion_tokens":
            raise BudgetError(
                f"{context}.reasoning_billing must explicitly state "
                "'included_in_completion_tokens'; unknown reasoning billing "
                "cannot be assumed free"
            )
        routes = data.get("route_providers")
        if (
            not isinstance(routes, list)
            or not routes
            or not all(isinstance(route, str) and route for route in routes)
            or len(set(routes)) != len(routes)
        ):
            raise BudgetError(
                f"{context}.route_providers must be a non-empty list of "
                "unique strings"
            )
        allow_fallbacks = data.get("allow_fallbacks")
        require_parameters = data.get("require_parameters")
        if not isinstance(allow_fallbacks, bool):
            raise BudgetError(f"{context}.allow_fallbacks must be a boolean")
        if not isinstance(require_parameters, bool):
            raise BudgetError(f"{context}.require_parameters must be a boolean")
        max_prompt_price = _decimal(
            data.get("max_provider_prompt_price_per_million"),
            context=f"{context}.max_provider_prompt_price_per_million",
        )
        max_completion_price = _decimal(
            data.get("max_provider_completion_price_per_million"),
            context=f"{context}.max_provider_completion_price_per_million",
        )
        prompt_rate = _decimal(
            data.get("prompt_usd_per_token"),
            context=f"{context}.prompt_usd_per_token",
        )
        completion_rate = _decimal(
            data.get("completion_usd_per_token"),
            context=f"{context}.completion_usd_per_token",
        )
        per_million = Decimal(1_000_000)
        if max_prompt_price / per_million > prompt_rate:
            raise BudgetError(
                f"{context} provider prompt price cap exceeds its planned rate"
            )
        if max_completion_price / per_million > completion_rate:
            raise BudgetError(
                f"{context} provider completion price cap exceeds its planned rate"
            )
        return cls(
            request_id=_string(data, "request_id", context=context),
            model_id=_string(data, "model_id", context=context),
            workload_id=_string(data, "workload_id", context=context),
            workload_version=_string(data, "workload_version", context=context),
            prompt_sha256=_string(data, "prompt_sha256", context=context),
            request_config_sha256=_string(
                data, "request_config_sha256", context=context
            ),
            endpoint_origin=_string(data, "endpoint_origin", context=context),
            endpoint_path=_string(data, "endpoint_path", context=context),
            route_providers=tuple(routes),
            allow_fallbacks=allow_fallbacks,
            require_parameters=require_parameters,
            max_provider_prompt_price_per_million=max_prompt_price,
            max_provider_completion_price_per_million=max_completion_price,
            reasoning_effort=_string(data, "reasoning_effort", context=context),
            input_token_ceiling=_positive_int(
                data, "input_token_ceiling", context=context
            ),
            max_output_tokens=_positive_int(data, "max_output_tokens", context=context),
            prompt_usd_per_token=prompt_rate,
            completion_usd_per_token=completion_rate,
            cached_prompt_usd_per_token=_decimal(
                data.get("cached_prompt_usd_per_token"),
                context=f"{context}.cached_prompt_usd_per_token",
            ),
            cache_write_billing=cache_write_billing,
            reasoning_billing=reasoning_billing,
        )


@dataclass(frozen=True)
class BudgetPlan:
    """Validated immutable authorization for an entire hosted experiment."""

    experiment_id: str
    ledger_identity: str
    ledger_file_name: str
    ledger_path_sha256: str
    authorized_total_usd: Decimal
    requests: tuple[BudgetRequest, ...]
    content_sha256: str

    @property
    def planned_ceiling_usd(self) -> Decimal:
        return sum((request.ceiling_usd for request in self.requests), Decimal())

    @classmethod
    def read(
        cls, path: Path, *, hard_limit_usd: Decimal = DEFAULT_HARD_LIMIT_USD
    ) -> BudgetPlan:
        try:
            text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
            raw = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise BudgetError(f"failed to read budget plan: {exc}") from exc
        if not isinstance(raw, dict):
            raise BudgetError("budget plan must be an object")
        if raw.get("schema_version") != BUDGET_PLAN_SCHEMA_VERSION:
            raise BudgetError(
                "budget plan has an unsupported or missing schema_version"
            )
        expected_plan_seal = raw.get(_PLAN_SEAL_FIELD)
        unsigned_plan = dict(raw)
        unsigned_plan.pop(_PLAN_SEAL_FIELD, None)
        actual_plan_seal = sha256_text(_canonical_json(unsigned_plan))
        if expected_plan_seal != actual_plan_seal:
            raise BudgetError("budget plan integrity seal does not verify")
        requests_raw = raw.get("requests")
        if not isinstance(requests_raw, list) or not requests_raw:
            raise BudgetError("budget plan.requests must be a non-empty list")
        requests = tuple(
            BudgetRequest.from_dict(item, index=index)
            for index, item in enumerate(requests_raw)
        )
        request_ids = [request.request_id for request in requests]
        if len(set(request_ids)) != len(request_ids):
            raise BudgetError("budget plan request_id values must be unique")
        authorized = _decimal(
            raw.get("authorized_total_usd"),
            context="budget plan.authorized_total_usd",
            positive=True,
        )
        if authorized > hard_limit_usd:
            raise BudgetError(
                f"budget plan authorizes {_money(authorized)} USD, above the "
                f"hard lifetime cap of {_money(hard_limit_usd)} USD"
            )
        ledger_identity = _string(raw, "ledger_identity", context="budget plan")
        if not _LEDGER_IDENTITY_PATTERN.fullmatch(ledger_identity):
            raise BudgetError(
                "budget plan.ledger_identity must be a stable 8-128 character "
                "identifier"
            )
        ledger_path_sha256 = _string(raw, "ledger_path_sha256", context="budget plan")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", ledger_path_sha256):
            raise BudgetError("budget plan.ledger_path_sha256 must be a sha256 digest")
        plan = cls(
            experiment_id=_string(raw, "experiment_id", context="budget plan"),
            ledger_identity=ledger_identity,
            ledger_file_name=_string(raw, "ledger_file_name", context="budget plan"),
            ledger_path_sha256=ledger_path_sha256,
            authorized_total_usd=authorized,
            requests=requests,
            content_sha256=actual_plan_seal,
        )
        if plan.planned_ceiling_usd > authorized:
            raise BudgetError(
                f"planned worst-case cost {_money(plan.planned_ceiling_usd)} "
                f"USD exceeds the authorized {_money(authorized)} USD"
            )
        return plan


def conservative_input_token_upper_bound(prompt: str, system_prompt: str | None) -> int:
    """Return the experiment's tokenizer-independent input ceiling check.

    UTF-8 bytes bound non-empty byte-token encodings. The fixed 4,096-token
    reserve covers provider chat framing and special tokens, which are not
    exposed by OpenRouter's catalog for tokenizer ``Other``.
    """

    return (
        len(prompt.encode("utf-8"))
        + (0 if system_prompt is None else len(system_prompt.encode("utf-8")))
        + 4096
    )


def _sealed(
    payload: dict[str, Any], *, seal_field: str = _LEDGER_SEAL_FIELD
) -> dict[str, Any]:
    without_seal = dict(payload)
    without_seal.pop(seal_field, None)
    return {
        **without_seal,
        seal_field: sha256_text(_canonical_json(without_seal)),
    }


def _validate_seal(
    payload: dict[str, Any], *, seal_field: str = _LEDGER_SEAL_FIELD
) -> None:
    expected = payload.get(seal_field)
    if not isinstance(expected, str):
        raise BudgetError("budget state is missing its integrity seal")
    actual = _sealed(payload, seal_field=seal_field)[seal_field]
    if expected != actual:
        raise BudgetError("budget state integrity seal does not verify")


def _default_authorization_state_dir() -> Path:
    return Path.home() / ".llmtracefx" / "budget-authorizations"


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class BudgetGate:
    """Atomic, sealed request claims against one immutable budget plan."""

    def __init__(
        self,
        plan_path: Path,
        ledger_path: Path,
        *,
        hard_limit_usd: Decimal = DEFAULT_HARD_LIMIT_USD,
    ) -> None:
        self._configure(
            plan_path,
            ledger_path,
            hard_limit_usd=hard_limit_usd,
        )
        with _exclusive_lock(self.lock_path):
            if not self.ledger_path.exists() or not self.anchor_path.exists():
                raise BudgetError(
                    "budget ledger or its monotonic authorization anchor does "
                    "not exist; execution never initializes or resets lifetime "
                    "authorization"
                )
            self._read_ledger()

    def _configure(
        self,
        plan_path: Path,
        ledger_path: Path,
        *,
        hard_limit_usd: Decimal,
    ) -> None:
        self.plan = BudgetPlan.read(plan_path, hard_limit_usd=hard_limit_usd)
        resolved_ledger = ledger_path.expanduser().resolve()
        if resolved_ledger.name != self.plan.ledger_file_name:
            raise BudgetError(
                "budget ledger file name does not match the immutable plan"
            )
        if sha256_text(str(resolved_ledger)) != self.plan.ledger_path_sha256:
            raise BudgetError(
                "budget ledger path does not match the sealed immutable plan"
            )
        self.ledger_path = resolved_ledger
        state_dir = _default_authorization_state_dir()
        self.anchor_path = state_dir / f"{self.plan.ledger_identity}.json"
        self.lock_path = state_dir / f".{self.plan.ledger_identity}.lock"
        self._requests = {request.request_id: request for request in self.plan.requests}

    @classmethod
    def initialize(
        cls,
        plan_path: Path,
        ledger_path: Path,
        *,
        hard_limit_usd: Decimal = DEFAULT_HARD_LIMIT_USD,
    ) -> BudgetGate:
        """Create the sole initial ledger before any execution is allowed."""

        gate = cls.__new__(cls)
        gate._configure(
            plan_path,
            ledger_path,
            hard_limit_usd=hard_limit_usd,
        )
        with _exclusive_lock(gate.lock_path):
            if gate.ledger_path.exists() or gate.anchor_path.exists():
                raise BudgetError(
                    "budget ledger or authorization anchor already exists and "
                    "cannot be reset"
                )
            gate._write_initial_state(gate._initial_ledger())
        return gate

    def _initial_ledger(self) -> dict[str, Any]:
        by_model: dict[str, Decimal] = defaultdict(Decimal)
        entries: list[dict[str, Any]] = []
        for request in self.plan.requests:
            by_model[request.model_id] += request.ceiling_usd
            entries.append(
                {
                    **request.to_dict(),
                    "status": "planned",
                    "attempt_started_at": None,
                    "settled_at": None,
                    "provider_usage": None,
                    "computed_observed_cost_usd": None,
                    "provider_reported_cost_usd_credits": None,
                    "accounted_cost_usd": "0.000000000000",
                    "failure": None,
                }
            )
        return {
            "schema_version": BUDGET_LEDGER_SCHEMA_VERSION,
            "revision": 0,
            "experiment_id": self.plan.experiment_id,
            "ledger_identity": self.plan.ledger_identity,
            "plan_sha256": self.plan.content_sha256,
            "authorized_total_usd": _money(self.plan.authorized_total_usd),
            "planned_request_count": len(self.plan.requests),
            "planned_ceiling_usd": _money(self.plan.planned_ceiling_usd),
            "model_ceilings_usd": {
                model: _money(amount) for model, amount in sorted(by_model.items())
            },
            "cumulative_accounted_usd": "0.000000000000",
            "remaining_authorized_usd": _money(self.plan.authorized_total_usd),
            "terminal_failure": None,
            "entries": entries,
            "events": [],
        }

    def _read_ledger(self) -> dict[str, Any]:
        try:
            text = read_bounded_regular_text(
                self.ledger_path, MAX_EVIDENCE_ARTIFACT_BYTES
            )
            payload = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise BudgetError(f"failed to read budget ledger: {exc}") from exc
        if not isinstance(payload, dict):
            raise BudgetError("budget ledger must be an object")
        _validate_seal(payload)
        if payload.get("schema_version") != BUDGET_LEDGER_SCHEMA_VERSION:
            raise BudgetError("budget ledger schema_version does not match")
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise BudgetError("budget ledger revision must be a non-negative integer")
        if payload.get("experiment_id") != self.plan.experiment_id:
            raise BudgetError("budget ledger experiment_id does not match its plan")
        if payload.get("ledger_identity") != self.plan.ledger_identity:
            raise BudgetError("budget ledger identity does not match its plan")
        if payload.get("plan_sha256") != self.plan.content_sha256:
            raise BudgetError("budget ledger is bound to a different budget plan")
        entries = payload.get("entries")
        if not isinstance(entries, list) or len(entries) != len(self.plan.requests):
            raise BudgetError("budget ledger entries do not match its plan")
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise BudgetError("budget ledger entry must be an object")
            request_id = entry.get("request_id")
            status = entry.get("status")
            if request_id not in self._requests or request_id in seen:
                raise BudgetError("budget ledger request identities do not match")
            if status not in _REQUEST_STATUSES:
                raise BudgetError("budget ledger contains an invalid request status")
            if entry.get("ceiling_usd") != _money(
                self._requests[request_id].ceiling_usd
            ):
                raise BudgetError("budget ledger request ceiling does not match")
            seen.add(request_id)
        self._recompute_totals(payload)
        self._validate_anchor(payload)
        return payload

    def _anchor_payload(self, sealed_ledger: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": BUDGET_ANCHOR_SCHEMA_VERSION,
            "ledger_identity": self.plan.ledger_identity,
            "plan_sha256": self.plan.content_sha256,
            "ledger_path_sha256": self.plan.ledger_path_sha256,
            "revision": sealed_ledger["revision"],
            "bound_ledger_sha256": sealed_ledger[_LEDGER_SEAL_FIELD],
        }

    def _write_initial_state(self, payload: dict[str, Any]) -> None:
        sealed_ledger = _sealed(payload)
        atomic_write_text(
            self.ledger_path,
            json.dumps(sealed_ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        sealed_anchor = _sealed(
            self._anchor_payload(sealed_ledger), seal_field=_ANCHOR_SEAL_FIELD
        )
        atomic_write_text(
            self.anchor_path,
            json.dumps(sealed_anchor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def _read_anchor(self) -> dict[str, Any]:
        try:
            text = read_bounded_regular_text(
                self.anchor_path, MAX_EVIDENCE_ARTIFACT_BYTES
            )
            anchor = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise BudgetError(
                f"failed to read budget authorization anchor: {exc}"
            ) from exc
        if not isinstance(anchor, dict):
            raise BudgetError("budget authorization anchor must be an object")
        _validate_seal(anchor, seal_field=_ANCHOR_SEAL_FIELD)
        if anchor.get("schema_version") != BUDGET_ANCHOR_SCHEMA_VERSION:
            raise BudgetError("budget authorization anchor schema does not match")
        if anchor.get("ledger_identity") != self.plan.ledger_identity:
            raise BudgetError("budget authorization anchor identity does not match")
        if anchor.get("plan_sha256") != self.plan.content_sha256:
            raise BudgetError("budget authorization anchor plan does not match")
        if anchor.get("ledger_path_sha256") != self.plan.ledger_path_sha256:
            raise BudgetError("budget authorization anchor path does not match")
        return anchor

    def _validate_anchor(self, ledger: dict[str, Any]) -> None:
        anchor = self._read_anchor()
        if anchor.get("revision") != ledger.get("revision"):
            raise BudgetError(
                "budget ledger revision does not match its monotonic anchor"
            )
        if anchor.get("bound_ledger_sha256") != ledger.get(_LEDGER_SEAL_FIELD):
            raise BudgetError(
                "budget ledger content does not match its monotonic anchor"
            )

    def _write_ledger(self, payload: dict[str, Any]) -> None:
        revision = payload.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int):
            raise BudgetError("budget ledger revision is invalid")
        payload["revision"] = revision + 1
        sealed = _sealed(payload)
        atomic_write_text(
            self.ledger_path,
            json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        sealed_anchor = _sealed(
            self._anchor_payload(sealed), seal_field=_ANCHOR_SEAL_FIELD
        )
        atomic_write_text(
            self.anchor_path,
            json.dumps(sealed_anchor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    @staticmethod
    def _entry(payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        for entry in payload["entries"]:
            if entry["request_id"] == request_id:
                return cast(dict[str, Any], entry)
        raise BudgetError("request_id is not present in the budget ledger")

    def _recompute_totals(self, payload: dict[str, Any]) -> None:
        accounted = sum(
            (
                _decimal(
                    entry.get("accounted_cost_usd"),
                    context="budget ledger entry.accounted_cost_usd",
                )
                for entry in payload["entries"]
            ),
            Decimal(),
        )
        remaining = self.plan.authorized_total_usd - accounted
        payload["cumulative_accounted_usd"] = _money(accounted)
        payload["remaining_authorized_usd"] = _money(remaining)

    def claim(
        self,
        request_id: str,
        *,
        model_id: str,
        workload_id: str,
        workload_version: str,
        prompt_sha256: str,
        request_config_sha256: str,
        endpoint_origin: str,
        endpoint_path: str,
        route_providers: tuple[str, ...],
        allow_fallbacks: bool,
        require_parameters: bool,
        max_provider_prompt_price_per_million: Decimal,
        max_provider_completion_price_per_million: Decimal,
        reasoning_effort: str,
        input_token_upper_bound: int,
        max_output_tokens: int,
    ) -> None:
        """Irreversibly reserve a request's full ceiling before network I/O."""

        planned = self._requests.get(request_id)
        if planned is None:
            raise BudgetError("request_id is not authorized by the budget plan")
        actual_binding = (
            model_id,
            workload_id,
            workload_version,
            prompt_sha256,
            request_config_sha256,
            endpoint_origin,
            endpoint_path,
            route_providers,
            allow_fallbacks,
            require_parameters,
            max_provider_prompt_price_per_million,
            max_provider_completion_price_per_million,
            reasoning_effort,
            max_output_tokens,
        )
        planned_binding = (
            planned.model_id,
            planned.workload_id,
            planned.workload_version,
            planned.prompt_sha256,
            planned.request_config_sha256,
            planned.endpoint_origin,
            planned.endpoint_path,
            planned.route_providers,
            planned.allow_fallbacks,
            planned.require_parameters,
            planned.max_provider_prompt_price_per_million,
            planned.max_provider_completion_price_per_million,
            planned.reasoning_effort,
            planned.max_output_tokens,
        )
        if actual_binding != planned_binding:
            raise BudgetError("request does not match its authorized plan binding")
        if input_token_upper_bound > planned.input_token_ceiling:
            raise BudgetError(
                "rendered input upper bound exceeds the authorized token ceiling"
            )

        with _exclusive_lock(self.lock_path):
            payload = self._read_ledger()
            if payload.get("terminal_failure") is not None:
                raise BudgetError(
                    "budget ledger is terminally failed; no later request is allowed"
                )
            entry = self._entry(payload, request_id)
            if entry["status"] != "planned":
                raise BudgetError(
                    "request was already attempted; automatic retries are forbidden"
                )
            if any(
                other["model_id"] == model_id and other["status"] == "failed"
                for other in payload["entries"]
            ):
                raise BudgetError(
                    "a prior failed request stopped this model; retries and later "
                    "requests for that model are forbidden"
                )
            remaining = _decimal(
                payload["remaining_authorized_usd"],
                context="budget ledger.remaining_authorized_usd",
            )
            if remaining < planned.ceiling_usd:
                raise BudgetError(
                    "remaining authorized budget cannot cover the next request ceiling"
                )
            entry["status"] = "attempted"
            entry["attempt_started_at"] = utc_now_iso()
            entry["accounted_cost_usd"] = _money(planned.ceiling_usd)
            self._recompute_totals(payload)
            payload["events"].append(
                {
                    "stage": "pre_request",
                    "at": entry["attempt_started_at"],
                    "request_id": request_id,
                    "model_id": model_id,
                    "request_ceiling_usd": _money(planned.ceiling_usd),
                    "cumulative_accounted_usd": payload["cumulative_accounted_usd"],
                    "remaining_authorized_usd": payload["remaining_authorized_usd"],
                }
            )
            self._write_ledger(payload)

    def settle(
        self,
        request_id: str,
        *,
        provider_success: bool,
        usage: ProviderUsage | None,
        failure: str | None,
    ) -> None:
        """Record provider accounting after an attempt, retaining uncertainty."""

        planned = self._requests.get(request_id)
        if planned is None:
            raise BudgetError("request_id is not authorized by the budget plan")
        with _exclusive_lock(self.lock_path):
            payload = self._read_ledger()
            entry = self._entry(payload, request_id)
            if entry["status"] != "attempted":
                raise BudgetError("only an attempted request can be settled")

            provider_cost: Decimal | None = None
            computed_cost: Decimal | None = None
            usage_payload: dict[str, Any] | None = None
            if usage is not None:
                usage_payload = usage.to_dict()
                if usage.cost_usd is not None:
                    provider_cost = Decimal(str(usage.cost_usd))
                if (
                    usage.prompt_tokens is not None
                    and usage.completion_tokens is not None
                    and usage.cached_prompt_tokens is not None
                    and usage.cached_prompt_tokens <= usage.prompt_tokens
                ):
                    uncached = usage.prompt_tokens - usage.cached_prompt_tokens
                    computed_cost = (
                        Decimal(uncached) * planned.prompt_usd_per_token
                        + Decimal(usage.cached_prompt_tokens)
                        * planned.cached_prompt_usd_per_token
                        + Decimal(usage.completion_tokens)
                        * planned.completion_usd_per_token
                    )

            breaches: list[str] = []
            if provider_cost is not None and provider_cost > planned.ceiling_usd:
                breaches.append(
                    "provider-reported cost exceeded the planned request ceiling"
                )
            if computed_cost is not None and computed_cost > planned.ceiling_usd:
                breaches.append(
                    "computed observed cost exceeded the planned request ceiling"
                )

            entry["status"] = (
                "failed" if breaches or not provider_success else "completed"
            )
            entry["settled_at"] = utc_now_iso()
            entry["provider_usage"] = usage_payload
            entry["computed_observed_cost_usd"] = (
                None if computed_cost is None else _money(computed_cost)
            )
            entry["provider_reported_cost_usd_credits"] = (
                None if provider_cost is None else _money(provider_cost)
            )
            entry["failure"] = "; ".join(breaches) if breaches else failure

            # A failed request or missing provider charge remains charged at its
            # entire ceiling. Only a successful response with an explicit cost
            # proves that reserving less is safe.
            if breaches:
                accounted = max(
                    (
                        planned.ceiling_usd,
                        provider_cost or Decimal(),
                        computed_cost or Decimal(),
                    )
                )
                payload["terminal_failure"] = {
                    "at": entry["settled_at"],
                    "request_id": request_id,
                    "reason": entry["failure"],
                }
            elif provider_success and provider_cost is not None:
                accounted = provider_cost
            else:
                accounted = planned.ceiling_usd
            entry["accounted_cost_usd"] = _money(accounted)
            self._recompute_totals(payload)
            payload["events"].append(
                {
                    "stage": "post_request",
                    "at": entry["settled_at"],
                    "request_id": request_id,
                    "model_id": planned.model_id,
                    "provider_success": provider_success,
                    "computed_observed_cost_usd": entry["computed_observed_cost_usd"],
                    "provider_reported_cost_usd_credits": entry[
                        "provider_reported_cost_usd_credits"
                    ],
                    "accounted_cost_usd": entry["accounted_cost_usd"],
                    "cumulative_accounted_usd": payload["cumulative_accounted_usd"],
                    "remaining_authorized_usd": payload["remaining_authorized_usd"],
                }
            )
            self._write_ledger(payload)

            if breaches:
                raise BudgetError("; ".join(breaches))

    def snapshot(self) -> dict[str, Any]:
        with _exclusive_lock(self.lock_path):
            return self._read_ledger()
