"""Authenticated local controller for the approved Modal compilation run.

All provider and harness effects are behind injected interfaces.  Local
approval, repository, path, page-policy, pricing, inventory, and budget gates
complete before the first paid operation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from ...collectors._shared import atomic_write_text
from .vllm_compile import (
    APPROVED_PLAN_SHA256,
    CELLS,
    CURRENT_RATES,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    HARD_CAP_USD,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_VLLM_IMAGE_DIGEST,
    HardwareIdentity,
    LifecycleBudgetLedger,
    VLLMCompileContractError,
    VLLMCompilePlan,
    build_plan,
    canonical_decimal,
    canonical_json,
    validate_hardware_identity,
    workload_descriptors,
)

FORBIDDEN_ENVIRONMENT_NAMES = frozenset(
    {
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_PROFILE",
        "MODAL_ENVIRONMENT",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    }
)
OFFICIAL_PRICING_URL = "https://modal.com/pricing"
OFFICIAL_VOLUMES_URL = "https://modal.com/docs/guide/volumes"
PROVIDER_RATE_SOURCE = "modal-cli://billing/rates/1.5.4"
RUNTIME_PINS = {
    "python_version": "3.12",
    "vllm_version": "0.28.0",
    "torch_version": "2.13.0+cu130",
    "cuda_version": "13.0",
}
CELL_FUNCTIONS = (
    "l40s_eager",
    "l40s_compiled",
    "h100_eager",
    "h100_compiled",
)
PROVENANCE_DOMAINS = frozenset(
    {
        "client_observed",
        "vllm",
        "cuda",
        "modal_provider",
        "model_reported",
        "derived",
    }
)
MAX_OUTPUT_BYTES = 65_536
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,31}[a-z0-9])?$")
_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_-]{8,}|"
    r"github_pat_[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{16,}|"
    r"modal[_-]token[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16})",
    re.IGNORECASE,
)
_PRIVATE_ID = re.compile(r"^(?:ap|vo|ta|ct|ac|us)-[A-Za-z0-9_-]{8,}$")

RATE_RESOURCES: dict[str, tuple[str, str]] = {
    "gpu_l40s": ("l40s_gpu_second_usd", "gpu_second"),
    "gpu_h100": ("h100_gpu_second_usd", "gpu_second"),
    "cpu": ("cpu_core_second_usd", "core_second"),
    "memory": ("memory_gib_second_usd", "gib_second"),
    "volume": ("volume_gib_month_usd", "gib_month"),
}
_APP_STATES = frozenset(
    {
        "deployed",
        "ephemeral",
        "ephemeral (detached)",
        "disabled",
        "initializing...",
        "stopped",
        "stopping...",
        "unknown",
    }
)
_OPTIONAL_REASONS = frozenset(
    {"unsupported", "unavailable", "permission_denied", "not_configured"}
)

_HARNESS_WORKLOAD_PAYLOAD = {
    "schema_version": "1",
    "descriptors": [item.to_dict() for item in workload_descriptors()],
    "sampling": {
        "max_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260831,
    },
    "tokenizer": {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "messages": "single_user_message",
    },
}
_HARNESS_OUTPUT_PAYLOAD = {
    "schema_version": "1",
    "request_terminal_required": True,
    "finish_reason_required": True,
    "input_count_source": "persisted_prompt_token_ids",
    "output_count_source": "request_output_token_ids",
    "decoded_output_max_utf8_bytes": MAX_OUTPUT_BYTES,
    "remote_correctness_evaluation": False,
    "resolved_execution_config_required": True,
    "missing_timing_reason_required": True,
    "provenance_domains": sorted(PROVENANCE_DOMAINS),
}
_REQUEST_FIELD_PROVENANCE = {
    "started_at": "client_observed",
    "ended_at": "client_observed",
    "wall_clock_seconds": "client_observed",
    "input_token_count": "derived",
    "input_token_ids_sha256": "derived",
    "output_token_count": "derived",
    "output_tokens_per_second": "derived",
    "output_rate_basis": "derived",
    "output_token_ids": "model_reported",
    "decoded_output": "model_reported",
    "finish_reason": "model_reported",
    "ttft_seconds": "vllm",
    "correctness": "derived",
}


class ModalOrchestratorError(RuntimeError):
    """A refused or incomplete execution, with teardown evidence attached."""

    def __init__(
        self,
        message: str,
        *,
        original: BaseException | None = None,
        teardown: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.original = original
        self.teardown = dict(teardown or {})


@dataclass(frozen=True)
class RawJSON:
    """Transient provider JSON and the digest of its exact response bytes."""

    payload: Any
    response_sha256: str

    @classmethod
    def from_bytes(cls, raw: bytes) -> RawJSON:
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                parse_float=Decimal,
                parse_int=Decimal,
            )
        except (UnicodeDecodeError, ValueError) as exc:
            raise ModalOrchestratorError("provider returned invalid JSON") from exc
        return cls(payload, _sha256_bytes(raw))


@dataclass(frozen=True)
class OptionalProviderJSON:
    response: RawJSON | None
    unavailable_reason: str | None


class Provider(Protocol):
    def version(self) -> str: ...

    def authenticate(self) -> str: ...

    def billing_rates(self) -> RawJSON: ...

    def billing_summary(self) -> OptionalProviderJSON: ...

    def app_inventory(self) -> RawJSON: ...

    def volume_inventory(self) -> RawJSON: ...

    def container_inventory(self) -> RawJSON: ...

    def secret_inventory(self) -> RawJSON: ...

    def create_volume(self, name: str) -> None: ...

    def stop_app(self, name: str) -> None: ...

    def delete_volume(self, name: str, *, allow_missing: bool) -> None: ...


class GitInspector(Protocol):
    def root(self, workspace: Path) -> Path: ...

    def head(self, workspace: Path) -> str: ...

    def is_clean(self, workspace: Path) -> bool: ...


@dataclass(frozen=True)
class PagePolicySnapshot:
    pricing_status: int | None
    pricing_sha256: str | None
    volumes_status: int | None
    volumes_sha256: str | None
    failure_reason: str | None = None


class PagePolicy(Protocol):
    def fetch(self) -> PagePolicySnapshot: ...


class HarnessLoader(Protocol):
    def load(self, environment: Mapping[str, str]) -> Any: ...


@dataclass(frozen=True)
class ExecutionConfig:
    approval_path: Path
    approval_sha256: str
    git_head: str
    workspace_path: Path
    output_dir: Path
    ledger_path: Path
    experiment_id: str


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_regular(path: Path, *, maximum: int = 1_048_576) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ModalOrchestratorError(f"{path.name} must be a regular non-symlink file")
    if path.stat().st_size > maximum:
        raise ModalOrchestratorError(f"{path.name} exceeds its size limit")
    return path.read_bytes()


def _reject_credential_environment(environ: Mapping[str, str]) -> None:
    present = sorted(
        name for name in environ if name.upper() in FORBIDDEN_ENVIRONMENT_NAMES
    )
    if present:
        raise ModalOrchestratorError(
            "forbidden credential/profile environment names are present: "
            + ", ".join(present)
        )


def _validate_local(
    config: ExecutionConfig,
    environ: Mapping[str, str],
    git: GitInspector,
) -> str:
    _reject_credential_environment(environ)
    if not _GIT_HEAD.fullmatch(config.git_head):
        raise ModalOrchestratorError("git head must be exact 40-hex")
    if not _SAFE_ID.fullmatch(config.experiment_id):
        raise ModalOrchestratorError("experiment id is not a safe unique identifier")
    if _PRIVATE_ID.fullmatch(config.experiment_id):
        raise ModalOrchestratorError("experiment id resembles a private provider id")
    workspace = config.workspace_path
    if not workspace.is_dir() or workspace.is_symlink():
        raise ModalOrchestratorError(
            "workspace must be an existing non-symlink directory"
        )
    workspace = workspace.resolve()
    if git.root(workspace).resolve() != workspace:
        raise ModalOrchestratorError("workspace must be the exact repository root")
    if git.head(workspace) != config.git_head or not git.is_clean(workspace):
        raise ModalOrchestratorError(
            "workspace must be clean at the exact approved HEAD"
        )
    output = config.output_dir
    if (
        not output.is_dir()
        or output.is_symlink()
        or _inside(output, workspace)
        or any(output.iterdir())
    ):
        raise ModalOrchestratorError(
            "output directory must be empty, non-symlink, and outside the repository"
        )
    ledger = config.ledger_path
    if _inside(ledger, workspace) or ledger.exists():
        raise ModalOrchestratorError("ledger must be a new path outside the repository")
    if not ledger.parent.is_dir() or ledger.parent.is_symlink():
        raise ModalOrchestratorError("ledger parent must be an existing safe directory")
    if not _SHA256.fullmatch(config.approval_sha256):
        raise ModalOrchestratorError(
            "approval hash must be exact sha256 lowercase hexadecimal"
        )
    if config.approval_sha256 != APPROVED_PLAN_SHA256:
        raise ModalOrchestratorError(
            "approval hash is not the coordinator-approved plan"
        )
    if _sha256_bytes(_read_regular(config.approval_path)) != config.approval_sha256:
        raise ModalOrchestratorError("coordinator-approved plan artifact hash mismatch")
    return config.approval_sha256


def _validate_page_policy(
    snapshot: PagePolicySnapshot,
) -> dict[str, Any]:
    if snapshot.failure_reason is not None:
        raise ModalOrchestratorError("official page policy fetch failed")
    if snapshot.pricing_status != 200 or snapshot.volumes_status != 200:
        raise ModalOrchestratorError("official page policy status is not successful")
    if not _SHA256.fullmatch(str(snapshot.pricing_sha256)) or not _SHA256.fullmatch(
        str(snapshot.volumes_sha256)
    ):
        raise ModalOrchestratorError("official page policy response hashes are invalid")
    return {
        "pricing": {
            "status": snapshot.pricing_status,
            "sha256": snapshot.pricing_sha256,
        },
        "volumes": {
            "status": snapshot.volumes_status,
            "sha256": snapshot.volumes_sha256,
        },
    }


def parse_billing_rates(response: RawJSON) -> dict[str, str]:
    """Parse only the exact Modal 1.5.4 five-resource rate schema."""

    payload = response.payload
    if not isinstance(payload, dict):
        raise ModalOrchestratorError("billing rates must be an object")
    parsed: dict[str, str] = {}
    for resource, (field, _unit) in RATE_RESOURCES.items():
        value = payload.get(resource)
        if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
            raise ModalOrchestratorError(f"billing price is invalid for {resource}")
        try:
            decimal = Decimal(str(value))
        except InvalidOperation as exc:
            raise ModalOrchestratorError("billing price is not decimal") from exc
        if not decimal.is_finite() or decimal <= 0:
            raise ModalOrchestratorError("billing price must be finite and positive")
        parsed[field] = canonical_decimal(decimal)
    return parsed


@dataclass(frozen=True)
class _InventoryItem:
    name: str
    status: str
    experiment_tag: str | None


def _parse_inventory(response: RawJSON, kind: str) -> tuple[_InventoryItem, ...]:
    payload = response.payload
    if not isinstance(payload, list):
        raise ModalOrchestratorError(f"{kind} inventory is ambiguous")
    items: list[_InventoryItem] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ModalOrchestratorError(f"{kind} inventory is ambiguous")
        if kind == "app":
            name, status = row.get("description"), row.get("state")
            if not isinstance(status, str) or status not in _APP_STATES:
                raise ModalOrchestratorError("app inventory is ambiguous")
        elif kind == "volume":
            name, status = row.get("name"), "running"
        elif kind == "container":
            name, status = row.get("app_name"), "running"
        elif kind == "secret":
            name, status = row.get("name"), "running"
        else:
            raise ModalOrchestratorError("unknown inventory kind")
        if not isinstance(name, str) or not name:
            raise ModalOrchestratorError(f"{kind} inventory is ambiguous")
        items.append(_InventoryItem(name, status, None))
    return tuple(items)


def _inventory_facts(items: Sequence[_InventoryItem]) -> dict[str, Any]:
    statuses = {
        status: sum(item.status == status for item in items)
        for status in sorted({item.status for item in items})
    }
    return {"count": len(items), "status_counts": statuses}


def _experiment_inventory_facts(
    items: Sequence[_InventoryItem],
    *,
    app_name: str,
    volume_name: str,
    experiment_id: str,
) -> dict[str, Any]:
    scoped = [
        item
        for item in items
        if item.name in {app_name, volume_name}
        or item.experiment_tag == experiment_id
        or experiment_id in item.name
    ]
    return _inventory_facts(scoped)


def _reject_stale(
    inventories: Mapping[str, Sequence[_InventoryItem]],
    *,
    app_name: str,
    volume_name: str,
    experiment_id: str,
) -> None:
    for items in inventories.values():
        for item in items:
            if (
                item.name in {app_name, volume_name}
                or item.experiment_tag == experiment_id
                or experiment_id in item.name
            ):
                raise ModalOrchestratorError("stale experiment resource already exists")


def _sanitize_optional_billing(value: OptionalProviderJSON) -> dict[str, Any]:
    if value.response is None:
        if value.unavailable_reason not in _OPTIONAL_REASONS:
            raise ModalOrchestratorError(
                "optional billing unavailable reason is invalid"
            )
        return {
            "facts": None,
            "unavailable_reason": value.unavailable_reason,
            "unsupported_fields": [
                "credits_usd",
                "budget_usd",
                "spend_limit_usd",
            ],
        }
    if value.unavailable_reason is not None or not isinstance(
        value.response.payload, dict
    ):
        raise ModalOrchestratorError("optional billing response is ambiguous")
    allowed = (
        "metered_cost",
        "billed_cost",
    )
    facts: dict[str, str | None] = {}
    for field in allowed:
        raw = value.response.payload.get(field)
        if raw is None:
            facts[field] = None
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int, Decimal)):
            raise ModalOrchestratorError("optional billing numeric fact is invalid")
        try:
            parsed = Decimal(str(raw))
        except InvalidOperation as exc:
            raise ModalOrchestratorError(
                "optional billing numeric fact is invalid"
            ) from exc
        if not parsed.is_finite() or parsed < 0:
            raise ModalOrchestratorError("optional billing numeric fact is invalid")
        facts[field] = canonical_decimal(parsed)
    facts.update(
        {
            "credits_usd": None,
            "budget_usd": None,
            "spend_limit_usd": None,
        }
    )
    return {
        "facts": facts,
        "unavailable_reason": None,
        "unsupported_fields": [
            "credits_usd",
            "budget_usd",
            "spend_limit_usd",
        ],
    }


def _scan_persistable(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_persistable(key)
            _scan_persistable(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scan_persistable(item)
    elif isinstance(value, str):
        if _SHA256.fullmatch(value):
            return
        if _CREDENTIAL.search(value) or _PRIVATE_ID.fullmatch(value):
            raise ModalOrchestratorError(
                "credential-shaped token or private provider id refused"
            )


def _persist_verified(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    _scan_persistable(payload)
    sealed = dict(payload)
    sealed["artifact_sha256"] = _sha256_json(sealed)
    rendered = canonical_json(sealed)
    atomic_write_text(path, rendered)
    reread = _read_regular(path, maximum=8 * 1024 * 1024)
    if reread != rendered.encode():
        raise ModalOrchestratorError("immediate artifact byte verification failed")
    parsed = json.loads(reread)
    expected = parsed.pop("artifact_sha256", None)
    if expected != _sha256_json(parsed):
        raise ModalOrchestratorError("immediate artifact checksum verification failed")
    return sealed


def _persist_remote(path: Path, payload: Mapping[str, Any], *, seal_field: str) -> None:
    """Persist remote evidence byte-for-byte canonically after its own seal."""

    _scan_persistable(payload)
    _verify_remote_seal(payload, seal_field)
    rendered = canonical_json(payload)
    atomic_write_text(path, rendered)
    reread = _read_regular(path, maximum=8 * 1024 * 1024)
    if reread != rendered.encode():
        raise ModalOrchestratorError("remote artifact byte verification failed")
    parsed = json.loads(reread)
    if not isinstance(parsed, dict):
        raise ModalOrchestratorError("remote artifact reread is invalid")
    _verify_remote_seal(parsed, seal_field)


def _reserve_first_pass(
    config: ExecutionConfig, plan: VLLMCompilePlan, app_name: str, volume_name: str
) -> tuple[LifecycleBudgetLedger, dict[str, Any]]:
    ledger = LifecycleBudgetLedger.initialize(
        config.ledger_path,
        plan=plan,
        git_head=config.git_head,
        workspace_path=config.workspace_path,
    )
    argv_by_line = {
        "image-allowance": ("modal", "image", "build", OFFICIAL_VLLM_IMAGE_DIGEST),
        "staging": (app_name, "stage_qwen3", "remote"),
        "cell-l40s-eager": (app_name, "l40s_eager", "remote_gen"),
        "cell-l40s-compiled": (app_name, "l40s_compiled", "remote_gen"),
        "cell-h100-eager": (app_name, "h100_eager", "remote_gen"),
        "cell-h100-compiled": (app_name, "h100_compiled", "remote_gen"),
        "storage": ("modal", "volume", "retain", volume_name, "4-days"),
    }
    for line in plan.lines:
        ledger.reserve(
            f"first-{line.line_id}",
            line_id=line.line_id,
            ceiling_usd=line.amount_usd,
            argv=argv_by_line[line.line_id],
            reserved_at=_now(),
        )
    snapshot = ledger.snapshot()
    if snapshot["reserved_usd"] != canonical_decimal(plan.first_pass_usd):
        raise ModalOrchestratorError("first-pass reservation total does not match plan")
    expected_remaining = plan.full_retry_usd + plan.contingency_usd
    if snapshot["remaining_usd"] != canonical_decimal(expected_remaining):
        raise ModalOrchestratorError("retry and contingency lifecycle is not preserved")
    if plan.first_pass_usd + expected_remaining != HARD_CAP_USD:
        raise ModalOrchestratorError("lifecycle reservation does not preserve USD 28")
    return ledger, snapshot


def _harness_environment(
    plan: VLLMCompilePlan, experiment_id: str, volume_name: str
) -> dict[str, str]:
    return {
        "LLMTRACEFX_QWEN3_COMPILE_PLAN_JSON": plan.to_json(),
        "LLMTRACEFX_QWEN3_COMPILE_APP_NAME": "qwen3-compile",
        "LLMTRACEFX_QWEN3_COMPILE_VOLUME_NAME": volume_name,
        "LLMTRACEFX_QWEN3_COMPILE_EXPERIMENT_TAG": experiment_id,
        "LLMTRACEFX_QWEN3_COMPILE_WORKLOAD_SHA256": _sha256_json(
            _HARNESS_WORKLOAD_PAYLOAD
        ),
        "LLMTRACEFX_QWEN3_COMPILE_OUTPUT_SHA256": _sha256_json(_HARNESS_OUTPUT_PAYLOAD),
    }


def _reproduction_command(config: ExecutionConfig) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "llmtracefx.optimizer.lab.qwen3_8b.modal_orchestrator",
        "--approval",
        "<approved-plan-path>",
        "--approval-sha256",
        config.approval_sha256,
        "--git-head",
        config.git_head,
        "--workspace",
        "<repository-root>",
        "--output-dir",
        "<private-output-directory>",
        "--ledger",
        "<private-ledger-path>",
        "--experiment-id",
        config.experiment_id,
    ]


def _verify_remote_seal(payload: Mapping[str, Any], field: str) -> None:
    material = dict(payload)
    expected = material.pop(field, None)
    if not isinstance(expected, str) or expected != _sha256_json(material):
        raise ModalOrchestratorError(f"remote {field} does not verify")


def _finite_metric(value: Any, *, optional: bool = True) -> float | None:
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not float(value) >= 0
        or not float(value) < float("inf")
    ):
        raise ModalOrchestratorError("remote metric is not finite and non-negative")
    return float(value)


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ModalOrchestratorError("remote timestamp is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ModalOrchestratorError("remote timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ModalOrchestratorError("remote timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _expected_model_inventory() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "data" / "qwen3-8b-conversion-manifest-v1.json"
    try:
        payload = json.loads(_read_regular(path).decode("utf-8"))
        inventory = payload["source"]["files"]
    except (KeyError, UnicodeDecodeError, ValueError) as exc:
        raise ModalOrchestratorError("packaged model inventory is invalid") from exc
    if not isinstance(inventory, list):
        raise ModalOrchestratorError("packaged model inventory is invalid")
    return sorted(inventory, key=lambda item: item["path"])


def _validate_staging(receipt: Any, plan: VLLMCompilePlan) -> dict[str, Any]:
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "1":
        raise ModalOrchestratorError("staging receipt schema is invalid")
    if set(receipt) != {
        "schema_version",
        "plan_sha256",
        "workload_sha256",
        "output_contract_sha256",
        "runtime_sha256",
        "image_sha256",
        "image_digest",
        "model_id",
        "model_revision",
        "model_file_count",
        "model_bytes",
        "inventory",
        "prompts",
        "prompt_ids_sha256",
        "staged_at",
        "receipt_sha256",
    }:
        raise ModalOrchestratorError("staging receipt fields are invalid")
    _verify_remote_seal(receipt, "receipt_sha256")
    _utc_timestamp(receipt.get("staged_at"))
    required = {
        "plan_sha256": plan.content_sha256,
        "workload_sha256": _sha256_json(_HARNESS_WORKLOAD_PAYLOAD),
        "output_contract_sha256": _sha256_json(_HARNESS_OUTPUT_PAYLOAD),
        "runtime_sha256": _sha256_json(RUNTIME_PINS),
        "image_sha256": _sha256_json(
            {"reference": ("vllm/vllm-openai:v0.28.0@" + OFFICIAL_VLLM_IMAGE_DIGEST)}
        ),
        "image_digest": OFFICIAL_VLLM_IMAGE_DIGEST,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_count": EXPECTED_MODEL_FILE_COUNT,
        "model_bytes": EXPECTED_MODEL_BYTES,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise ModalOrchestratorError(f"staging receipt {field} mismatch")
    prompts = receipt.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 6:
        raise ModalOrchestratorError("staging prompt receipt is incomplete")
    if not _SHA256.fullmatch(str(receipt.get("prompt_ids_sha256"))):
        raise ModalOrchestratorError("staging prompt ID collection hash is invalid")
    expected_prompts = {
        f"{item.context_tier}/{item.workload_id}": item.prompt_sha256
        for item in workload_descriptors()
    }
    seen: set[str] = set()
    for item in prompts:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "key",
                "prompt_sha256",
                "prompt_token_ids_sha256",
                "prompt_token_ids",
                "input_token_count",
                "decoded_prompt_sha256",
            }
            or not isinstance(item.get("key"), str)
            or item["key"] in seen
            or not _SHA256.fullmatch(str(item.get("prompt_token_ids_sha256")))
            or not _SHA256.fullmatch(str(item.get("decoded_prompt_sha256")))
            or not isinstance(item.get("prompt_token_ids"), list)
            or not item["prompt_token_ids"]
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in item["prompt_token_ids"]
            )
            or _sha256_json(item["prompt_token_ids"])
            != item.get("prompt_token_ids_sha256")
            or isinstance(item.get("input_token_count"), bool)
            or not isinstance(item.get("input_token_count"), int)
            or item["input_token_count"] <= 0
            or item["input_token_count"] != len(item["prompt_token_ids"])
            or expected_prompts.get(item["key"]) != item.get("prompt_sha256")
        ):
            raise ModalOrchestratorError("staging prompt receipt is invalid")
        seen.add(item["key"])
    if seen != set(expected_prompts):
        raise ModalOrchestratorError("staging prompt keys do not match the workload")
    staged_prompt_payload = {
        "schema_version": "1",
        "workload_sha256": _sha256_json(_HARNESS_WORKLOAD_PAYLOAD),
        "prompts": {item["key"]: item["prompt_token_ids"] for item in prompts},
    }
    if _sha256_json(staged_prompt_payload) != receipt.get("prompt_ids_sha256"):
        raise ModalOrchestratorError(
            "staging prompt ID collection hash does not verify"
        )
    inventory = receipt.get("inventory")
    if inventory != _expected_model_inventory():
        raise ModalOrchestratorError(
            "staging model inventory differs from the public pin"
        )
    return dict(receipt)


def _validate_cell_terminal(
    event: Any,
    *,
    cell_index: int,
    receipt: Mapping[str, Any],
    plan: VLLMCompilePlan,
) -> dict[str, Any]:
    if not isinstance(event, dict) or event.get("event") != "cell_terminal":
        raise ModalOrchestratorError("cell terminal event is invalid")
    if event.get("provenance") not in PROVENANCE_DOMAINS:
        raise ModalOrchestratorError("cell terminal provenance is invalid")
    record = event.get("record")
    if not isinstance(record, dict) or record.get("schema_version") != "1":
        raise ModalOrchestratorError("cell terminal record schema is invalid")
    _verify_remote_seal(record, "cell_sha256")
    if set(record) != {
        "schema_version",
        "cell",
        "plan_sha256",
        "staging_receipt_sha256",
        "workload_sha256",
        "output_contract_sha256",
        "runtime_sha256",
        "image_sha256",
        "hardware",
        "runtime",
        "resolved_execution_config",
        "initialization_started_at",
        "initialization_ready_at",
        "compilation_seconds",
        "compilation_seconds_unobservable_reason",
        "cuda_graph_seconds",
        "cuda_graph_seconds_unobservable_reason",
        "peak_gpu_memory_mib",
        "requests",
        "correctness_evaluated_remotely",
        "terminal",
        "cell_sha256",
    }:
        raise ModalOrchestratorError("cell terminal record fields are invalid")
    cell = CELLS[cell_index]
    if record.get("cell") != cell.to_dict():
        raise ModalOrchestratorError("cell terminal identity mismatch")
    if (
        record.get("plan_sha256") != plan.content_sha256
        or record.get("staging_receipt_sha256") != receipt.get("receipt_sha256")
        or record.get("workload_sha256") != _sha256_json(_HARNESS_WORKLOAD_PAYLOAD)
        or record.get("output_contract_sha256") != _sha256_json(_HARNESS_OUTPUT_PAYLOAD)
        or record.get("runtime_sha256") != _sha256_json(RUNTIME_PINS)
        or record.get("image_sha256")
        != _sha256_json(
            {"reference": "vllm/vllm-openai:v0.28.0@" + OFFICIAL_VLLM_IMAGE_DIGEST}
        )
        or record.get("terminal") is not True
        or record.get("correctness_evaluated_remotely") is not False
    ):
        raise ModalOrchestratorError("cell terminal contract binding mismatch")
    hardware = record.get("hardware")
    if not isinstance(hardware, dict) or set(hardware) != {
        "gpu_name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "gpu_count",
    }:
        raise ModalOrchestratorError("cell hardware evidence is missing")
    try:
        validate_hardware_identity(
            CELLS[cell_index],
            HardwareIdentity(hardware["gpu_name"], hardware["gpu_count"]),
        )
    except (KeyError, VLLMCompileContractError) as exc:
        raise ModalOrchestratorError("cell hardware identity is invalid") from exc
    if (
        not isinstance(hardware["driver_version"], str)
        or not hardware["driver_version"]
        or _finite_metric(hardware["memory_total_mib"], optional=False) is None
        or hardware["memory_total_mib"] <= 0
        or _finite_metric(hardware["memory_used_mib"], optional=False) is None
        or hardware["memory_used_mib"] < 0
        or hardware["memory_used_mib"] > hardware["memory_total_mib"]
    ):
        raise ModalOrchestratorError("cell hardware evidence is invalid")
    if record.get("runtime") != RUNTIME_PINS:
        raise ModalOrchestratorError("cell runtime identity is invalid")
    expected_execution_config = (
        {
            "enforce_eager": False,
            "compilation_mode": "VLLM_COMPILE",
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
        }
        if cell.compile_enabled
        else {
            "enforce_eager": True,
            "compilation_mode": "NONE",
            "cuda_graph_mode": "NONE",
        }
    )
    if record.get("resolved_execution_config") != expected_execution_config:
        raise ModalOrchestratorError("cell resolved execution config is invalid")
    initialized = _utc_timestamp(record.get("initialization_started_at"))
    ready = _utc_timestamp(record.get("initialization_ready_at"))
    if ready < initialized:
        raise ModalOrchestratorError("cell initialization boundaries are reversed")
    compilation_seconds = _finite_metric(record.get("compilation_seconds"))
    cuda_graph_seconds = _finite_metric(record.get("cuda_graph_seconds"))
    compilation_reason = record.get("compilation_seconds_unobservable_reason")
    cuda_graph_reason = record.get("cuda_graph_seconds_unobservable_reason")
    if (compilation_seconds is None) != (
        isinstance(compilation_reason, str) and bool(compilation_reason)
    ) or (cuda_graph_seconds is None) != (
        isinstance(cuda_graph_reason, str) and bool(cuda_graph_reason)
    ):
        raise ModalOrchestratorError("cell component timing observability is invalid")
    if not cell.compile_enabled and (
        compilation_reason != "not_applicable_eager_mode"
        or cuda_graph_reason != "not_applicable_eager_mode"
    ):
        raise ModalOrchestratorError("eager cell component timing reason is invalid")
    peak_memory = _finite_metric(record.get("peak_gpu_memory_mib"), optional=False)
    if peak_memory is None or peak_memory <= 0:
        raise ModalOrchestratorError("cell peak GPU memory is missing")
    requests = record.get("requests")
    descriptors = workload_descriptors()
    if not isinstance(requests, list) or len(requests) != len(descriptors):
        raise ModalOrchestratorError("cell must contain exactly 12 requests")
    prompt_counts = {
        item["key"]: item["input_token_count"] for item in receipt["prompts"]
    }
    prompt_hashes = {
        item["key"]: item["prompt_token_ids_sha256"] for item in receipt["prompts"]
    }
    for request, descriptor in zip(requests, descriptors, strict=True):
        output_ids = (
            request.get("output_token_ids") if isinstance(request, dict) else None
        )
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        if (
            not isinstance(request, dict)
            or set(request)
            != set(descriptor.to_dict())
            | {
                "terminal",
                "started_at",
                "ended_at",
                "wall_clock_seconds",
                "input_token_count",
                "input_token_ids_sha256",
                "output_token_count",
                "output_tokens_per_second",
                "output_rate_basis",
                "output_token_ids",
                "decoded_output",
                "finish_reason",
                "ttft_seconds",
                "evaluator_input",
                "correctness",
                "provenance",
                "field_provenance",
            }
            or request.get("request_id") != descriptor.request_id
            or request.get("terminal") is not True
            or request.get("finish_reason") not in {"stop", "length"}
            or request.get("correctness") is not None
            or request.get("input_token_count") != prompt_counts.get(key)
            or request.get("input_token_ids_sha256") != prompt_hashes.get(key)
            or not isinstance(output_ids, list)
            or not output_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in output_ids
            )
            or request.get("output_token_count") != len(output_ids)
            or len(output_ids) > 96
            or not isinstance(request.get("decoded_output"), str)
            or len(request["decoded_output"].encode()) > MAX_OUTPUT_BYTES
            or request.get("provenance") != "model_reported"
            or request.get("evaluator_input")
            != {
                "workload_id": descriptor.workload_id,
                "context_tier": descriptor.context_tier,
                "decoded_output": request.get("decoded_output"),
                "output_token_ids": output_ids,
            }
        ):
            raise ModalOrchestratorError("cell contains an invalid terminal request")
        wall_clock = _finite_metric(request.get("wall_clock_seconds"), optional=False)
        output_rate = _finite_metric(
            request.get("output_tokens_per_second"), optional=False
        )
        _finite_metric(request.get("ttft_seconds"))
        if (
            wall_clock is None
            or wall_clock <= 0
            or output_rate is None
            or output_rate <= 0
            or request.get("output_rate_basis")
            != "output_tokens_per_complete_response_second"
        ):
            raise ModalOrchestratorError(
                "cell latency or output-rate evidence is invalid"
            )
        started = _utc_timestamp(request.get("started_at"))
        ended = _utc_timestamp(request.get("ended_at"))
        if ended < started:
            raise ModalOrchestratorError("request timestamps are reversed")
        provenance = request.get("field_provenance")
        if provenance != _REQUEST_FIELD_PROVENANCE:
            raise ModalOrchestratorError("request field provenance is invalid")
    return dict(record)


def _sanitized_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "reason": "execution_failed"}


def _original_context(error: BaseException) -> BaseException:
    original = error
    seen: set[int] = set()
    while original.__context__ is not None and id(original) not in seen:
        seen.add(id(original))
        original = original.__context__
    return original


@contextmanager
def _temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_no_live_experiment(
    inventories: Mapping[str, Sequence[_InventoryItem]],
    *,
    app_name: str,
    volume_name: str,
    experiment_id: str,
) -> None:
    live_statuses = {
        "running",
        "deployed",
        "ephemeral",
        "ephemeral (detached)",
        "initializing...",
        "stopping...",
    }
    for items in inventories.values():
        for item in items:
            belongs = (
                item.name in {app_name, volume_name}
                or item.experiment_tag == experiment_id
                or experiment_id in item.name
            )
            if belongs and item.status in live_statuses:
                raise ModalOrchestratorError("live experiment resource remains")


def _app_is_live(items: Sequence[_InventoryItem], app_name: str) -> bool:
    return any(
        item.name == app_name
        and item.status
        in {"deployed", "ephemeral", "ephemeral (detached)", "initializing..."}
        for item in items
    )


class SubprocessGitInspector:
    def _run(self, workspace: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(workspace), *args],
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        return completed.stdout.strip()

    def root(self, workspace: Path) -> Path:
        return Path(self._run(workspace, "rev-parse", "--show-toplevel"))

    def head(self, workspace: Path) -> str:
        return self._run(workspace, "rev-parse", "HEAD")

    def is_clean(self, workspace: Path) -> bool:
        return (
            self._run(
                workspace,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                ".",
                ":(exclude).agent-traces/**",
            )
            == ""
        )


class SubprocessModalProvider:
    """Modal 1.5.4 CLI adapter; raw stdout never leaves this object as text."""

    def __init__(self, executable: str = "modal") -> None:
        self.executable = executable

    def _run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [self.executable, *argv],
            check=True,
            capture_output=True,
            shell=False,
            timeout=60,
        )

    def _json(self, *argv: str) -> RawJSON:
        return RawJSON.from_bytes(self._run(argv).stdout)

    def authenticate(self) -> str:
        profile = self._run(("profile", "current")).stdout
        if not profile.strip():
            raise ModalOrchestratorError(
                "Modal current profile identity is unavailable"
            )
        self._run(("app", "list", "--json"))
        return _sha256_bytes(profile)

    def version(self) -> str:
        output = (
            self._run(("--version",)).stdout.decode("utf-8", errors="strict").strip()
        )
        if output != "modal client version: 1.5.4":
            raise ModalOrchestratorError("Modal CLI must be exactly version 1.5.4")
        return "1.5.4"

    def billing_rates(self) -> RawJSON:
        return self._json("billing", "rates", "--json")

    def billing_summary(self) -> OptionalProviderJSON:
        try:
            return OptionalProviderJSON(
                self._json("billing", "summary", "--json"), None
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            ModalOrchestratorError,
        ):
            return OptionalProviderJSON(None, "unsupported")

    def app_inventory(self) -> RawJSON:
        return self._json("app", "list", "--json")

    def volume_inventory(self) -> RawJSON:
        return self._json("volume", "list", "--json")

    def container_inventory(self) -> RawJSON:
        return self._json("container", "list", "--json")

    def secret_inventory(self) -> RawJSON:
        return self._json("secret", "list", "--json")

    def create_volume(self, name: str) -> None:
        self._run(("volume", "create", name))

    def stop_app(self, name: str) -> None:
        self._run(("app", "stop", name, "--yes"))

    def delete_volume(self, name: str, *, allow_missing: bool) -> None:
        argv = ["volume", "delete", name, "--yes"]
        if allow_missing:
            argv.append("--allow-missing")
        self._run(argv)


class URLPagePolicy:
    def _fetch(self, url: str) -> tuple[int, str]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LLMTraceFX-evidence/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ModalOrchestratorError("official page exceeds size bound")
            return int(response.status), _sha256_bytes(raw)

    def fetch(self) -> PagePolicySnapshot:
        try:
            pricing_status, pricing_hash = self._fetch(OFFICIAL_PRICING_URL)
            volumes_status, volumes_hash = self._fetch(OFFICIAL_VOLUMES_URL)
        except Exception:  # noqa: BLE001 - caller receives only a safe reason
            return PagePolicySnapshot(None, None, None, None, "unavailable")
        return PagePolicySnapshot(
            pricing_status,
            pricing_hash,
            volumes_status,
            volumes_hash,
        )


class ImportHarnessLoader:
    def load(self, environment: Mapping[str, str]) -> ModuleType:
        saved = {name: os.environ.get(name) for name in environment}
        os.environ.update(environment)
        try:
            return importlib.import_module("llmtracefx.deploy.modal_qwen3_compile_app")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def execute(
    config: ExecutionConfig,
    *,
    provider: Provider,
    page_policy: PagePolicy,
    harness_loader: HarnessLoader,
    git: GitInspector,
    environ: Mapping[str, str] | None = None,
    today: Callable[[], date] = date.today,
) -> dict[str, Any]:
    """Execute one sequential first pass and prove teardown."""

    environment = os.environ if environ is None else environ
    approval_sha256 = _validate_local(config, environment, git)
    page_facts = _validate_page_policy(page_policy.fetch())
    app_name = f"qwen3-compile-{config.experiment_id}"
    volume_name = f"qwen3-compile-volume-{config.experiment_id}"
    if app_name == volume_name or len(app_name) > 63 or len(volume_name) > 63:
        raise ModalOrchestratorError("derived app/volume names are unsafe")

    provider_version = provider.version()
    if provider_version != "1.5.4":
        raise ModalOrchestratorError("Modal provider adapter version is not pinned")
    profile_identity_sha256 = provider.authenticate()
    if not _SHA256.fullmatch(profile_identity_sha256):
        raise ModalOrchestratorError("Modal profile identity digest is invalid")
    authenticated_at = _now()
    rate_response = provider.billing_rates()
    rates = parse_billing_rates(rate_response)
    retrieval_date = today().isoformat()
    try:
        plan = build_plan(
            prices=rates,
            effective_date=retrieval_date,
            price_source=PROVIDER_RATE_SOURCE,
            price_source_sha256=rate_response.response_sha256,
            image_digest=OFFICIAL_VLLM_IMAGE_DIGEST,
            runtime_pins=RUNTIME_PINS,
            as_of_date=retrieval_date,
        )
    except VLLMCompileContractError as exc:
        raise ModalOrchestratorError("live pricing refused the lifecycle plan") from exc
    billing_before = _sanitize_optional_billing(provider.billing_summary())
    before = {
        "apps": _parse_inventory(provider.app_inventory(), "app"),
        "volumes": _parse_inventory(provider.volume_inventory(), "volume"),
        "containers": _parse_inventory(provider.container_inventory(), "container"),
        "secrets": _parse_inventory(provider.secret_inventory(), "secret"),
    }
    _reject_stale(
        before,
        app_name=app_name,
        volume_name=volume_name,
        experiment_id=config.experiment_id,
    )
    _, ledger_snapshot = _reserve_first_pass(config, plan, app_name, volume_name)

    state: dict[str, Any] = {
        "schema_version": "1",
        "experiment_id": config.experiment_id,
        "plan_sha256": plan.content_sha256,
        "approval_sha256": approval_sha256,
        "git_head": config.git_head,
        "modal_cli_version": provider_version,
        "profile_identity_sha256": profile_identity_sha256,
        "authenticated_at": authenticated_at,
        "page_policy": page_facts,
        "billing_before": billing_before,
        "provider_inventory_before": {
            key: _inventory_facts(value) for key, value in before.items()
        },
        "ledger": {
            "reserved_usd": ledger_snapshot["reserved_usd"],
            "remaining_usd": ledger_snapshot["remaining_usd"],
            "revision": ledger_snapshot["revision"],
        },
        "invocations": [],
        "status": "running",
    }
    _persist_verified(config.output_dir / "execution-state.json", state)
    _persist_verified(
        config.output_dir / "evidence-contract-input.json",
        {
            "schema_version": "1",
            "experiment_id": config.experiment_id,
            "git_head": config.git_head,
            "approved_plan_sha256": approval_sha256,
            "reproduction_command_argv": _reproduction_command(config),
            "plan": plan.to_dict(),
        },
    )
    _persist_verified(
        config.output_dir / "pricing-snapshot-input.json",
        {
            "schema_version": "1",
            "retrieved_date": retrieval_date,
            "pricing_page": page_facts["pricing"],
            "volumes_page": page_facts["volumes"],
            "rates_response_sha256": rate_response.response_sha256,
        },
    )
    _persist_verified(config.output_dir / "billing-before-input.json", billing_before)
    _persist_verified(
        config.output_dir / "ledger-projection-input.json",
        {
            "schema_version": "1",
            "plan_sha256": plan.content_sha256,
            "revision": ledger_snapshot["revision"],
            "reserved_usd": ledger_snapshot["reserved_usd"],
            "remaining_usd": ledger_snapshot["remaining_usd"],
            "lines": [
                {
                    "line_id": line.line_id,
                    "reserved_usd": canonical_decimal(line.amount_usd),
                }
                for line in plan.lines
            ],
        },
    )

    volume_creation_attempted = False
    app_started = False
    original: BaseException | None = None
    teardown: dict[str, Any] = {
        "schema_version": "1",
        "experiment_id": config.experiment_id,
        "complete": False,
        "steps": [],
        "billing_after": None,
        "billing_after_unavailable_reason": "not_configured",
        "billing_unsupported_fields": [
            "credits_usd",
            "budget_usd",
            "spend_limit_usd",
        ],
        "secrets_created": 0,
        "credentials_to_revoke": [],
        "post_delete_storage_billing_days_accounted": 4,
    }
    try:
        volume_creation_attempted = True
        provider.create_volume(volume_name)
        harness_environment = _harness_environment(
            plan, config.experiment_id, volume_name
        )
        with _temporary_environment(harness_environment):
            harness = harness_loader.load(harness_environment)
        if getattr(harness, "APP_NAME", None) != app_name:
            raise ModalOrchestratorError("harness app identity mismatch")
        app_run_started = _now()
        with harness.app.run(name=app_name, detach=False, interactive=False):
            app_started = True
            state["app_run_started_at"] = app_run_started
            state["app_run_ready_at"] = _now()
            _persist_verified(config.output_dir / "execution-state.json", state)
            stage_started = _now()
            receipt = _validate_staging(harness.stage_qwen3.remote(), plan)
            stage_ended = _now()
            _persist_remote(
                config.output_dir / "staging-receipt.json",
                receipt,
                seal_field="receipt_sha256",
            )
            state["invocations"].append(
                {
                    "function": "stage_qwen3",
                    "started_at": stage_started,
                    "ended_at": stage_ended,
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
            _persist_verified(config.output_dir / "execution-state.json", state)

            for index, function_name in enumerate(CELL_FUNCTIONS):
                invocation: dict[str, Any] = {
                    "function": function_name,
                    "started_at": _now(),
                    "events": [],
                    "ended_at": None,
                }
                terminal_events: list[dict[str, Any]] = []
                remote_function = getattr(harness, function_name)
                for event in remote_function.remote_gen():
                    if (
                        not isinstance(event, dict)
                        or event.get("provenance") not in PROVENANCE_DOMAINS
                    ):
                        raise ModalOrchestratorError(
                            "remote event provenance is invalid"
                        )
                    observed = {"received_at": _now(), "event": event}
                    invocation["events"].append(observed)
                    _persist_verified(
                        config.output_dir / f"{function_name}-lifecycle.json",
                        invocation,
                    )
                    if event.get("event") == "cell_terminal":
                        terminal_events.append(event)
                invocation["ended_at"] = _now()
                if len(terminal_events) != 1:
                    raise ModalOrchestratorError(
                        "cell must yield exactly one terminal event"
                    )
                received_events = [
                    observed["event"] for observed in invocation["events"]
                ]
                if (
                    not received_events
                    or received_events[0].get("event") != "container_started"
                    or received_events[-1].get("event") != "cell_terminal"
                ):
                    raise ModalOrchestratorError(
                        "cell lifecycle boundaries are invalid"
                    )
                event_names = [item.get("event") for item in received_events]
                if (
                    event_names.count("hardware_validated") != 1
                    or event_names.count("initialization_started") != 1
                    or event_names.count("initialization_ready") != 1
                    or event_names.count("request_terminal") != 12
                ):
                    raise ModalOrchestratorError(
                        "cell lifecycle event inventory is incomplete"
                    )
                request_event_ids = [
                    item.get("request", {}).get("request_id")
                    for item in received_events
                    if item.get("event") == "request_terminal"
                    and isinstance(item.get("request"), dict)
                ]
                if request_event_ids != [
                    item.request_id for item in workload_descriptors()
                ]:
                    raise ModalOrchestratorError(
                        "cell request lifecycle order is invalid"
                    )
                terminal = _validate_cell_terminal(
                    terminal_events[0],
                    cell_index=index,
                    receipt=receipt,
                    plan=plan,
                )
                _persist_remote(
                    config.output_dir / f"{function_name}-terminal.json",
                    terminal,
                    seal_field="cell_sha256",
                )
                _persist_verified(
                    config.output_dir / f"{function_name}-lifecycle.json",
                    invocation,
                )
                state["invocations"].append(invocation)
                _persist_verified(config.output_dir / "execution-state.json", state)
    except BaseException as exc:
        original = _original_context(exc)
    finally:
        if app_started or volume_creation_attempted:
            if app_started:
                try:
                    current_apps = _parse_inventory(provider.app_inventory(), "app")
                    if _app_is_live(current_apps, app_name):
                        provider.stop_app(app_name)
                        status = "complete"
                    else:
                        status = "already_stopped"
                    teardown["steps"].append(
                        {"operation": "stop_app", "status": status}
                    )
                except BaseException as exc:
                    teardown["steps"].append(
                        {
                            "operation": "stop_app",
                            "status": "failed",
                            "error": _sanitized_error(exc),
                        }
                    )
            else:
                teardown["steps"].append(
                    {"operation": "stop_app", "status": "not_required"}
                )
            try:
                provider.delete_volume(volume_name, allow_missing=True)
                teardown["steps"].append(
                    {"operation": "delete_volume", "status": "complete"}
                )
            except BaseException as exc:
                teardown["steps"].append(
                    {
                        "operation": "delete_volume",
                        "status": "failed",
                        "error": _sanitized_error(exc),
                    }
                )
            try:
                after = {
                    "apps": _parse_inventory(provider.app_inventory(), "app"),
                    "volumes": _parse_inventory(provider.volume_inventory(), "volume"),
                    "containers": _parse_inventory(
                        provider.container_inventory(), "container"
                    ),
                    "secrets": _parse_inventory(provider.secret_inventory(), "secret"),
                }
                _require_no_live_experiment(
                    after,
                    app_name=app_name,
                    volume_name=volume_name,
                    experiment_id=config.experiment_id,
                )
                teardown["provider_inventory_after"] = {
                    key: _experiment_inventory_facts(
                        value,
                        app_name=app_name,
                        volume_name=volume_name,
                        experiment_id=config.experiment_id,
                    )
                    for key, value in after.items()
                }
                teardown["inventory_status"] = "complete"
            except BaseException as exc:
                teardown["inventory_status"] = "incomplete"
                teardown["inventory_error"] = _sanitized_error(exc)
            try:
                billing_after = _sanitize_optional_billing(provider.billing_summary())
                teardown["billing_after"] = billing_after["facts"]
                teardown["billing_after_unavailable_reason"] = billing_after[
                    "unavailable_reason"
                ]
                teardown["billing_unsupported_fields"] = billing_after[
                    "unsupported_fields"
                ]
            except BaseException as exc:
                teardown["billing_after"] = None
                teardown["billing_after_unavailable_reason"] = "unavailable"
                teardown["billing_error"] = _sanitized_error(exc)
        teardown["complete"] = (
            bool(volume_creation_attempted)
            and teardown.get("inventory_status") == "complete"
            and all(
                step["status"] in {"complete", "not_required"}
                or step["status"] == "already_stopped"
                for step in teardown["steps"]
            )
        )
        try:
            _persist_verified(config.output_dir / "teardown-report.json", teardown)
        except BaseException as exc:
            teardown["complete"] = False
            teardown["report_persistence"] = "failed"
            teardown["report_persistence_error"] = _sanitized_error(exc)
            if original is None:
                original = exc

    if original is not None:
        state["status"] = "failed"
        state["error"] = _sanitized_error(original)
        state["teardown_complete"] = teardown["complete"]
        _persist_verified(config.output_dir / "execution-state.json", state)
        raise ModalOrchestratorError(
            "execution failed; see sanitized state and teardown report",
            original=original,
            teardown=teardown,
        ) from original
    if not teardown["complete"]:
        state["status"] = "incomplete"
        state["teardown_complete"] = False
        _persist_verified(config.output_dir / "execution-state.json", state)
        raise ModalOrchestratorError("teardown could not be proven", teardown=teardown)
    state["status"] = "complete"
    state["teardown_complete"] = True
    _persist_verified(config.output_dir / "execution-state.json", state)
    return state


def offline_preflight(
    config: ExecutionConfig,
    *,
    harness_loader: HarnessLoader,
    git: GitInspector,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate local bindings and harness registration without provider access."""

    environment = os.environ if environ is None else environ
    approval_sha256 = _validate_local(config, environment, git)
    today = date.today().isoformat()
    rates = {name: canonical_decimal(value) for name, value in CURRENT_RATES.items()}
    plan = build_plan(
        prices=rates,
        effective_date=today,
        price_source="approved-baseline://modal-public-pricing",
        price_source_sha256=_sha256_json(rates),
        image_digest=OFFICIAL_VLLM_IMAGE_DIGEST,
        runtime_pins=RUNTIME_PINS,
        as_of_date=today,
    )
    volume_name = f"qwen3-compile-volume-{config.experiment_id}"
    harness_environment = _harness_environment(plan, config.experiment_id, volume_name)
    with _temporary_environment(harness_environment):
        harness = harness_loader.load(harness_environment)
    expected_app = f"qwen3-compile-{config.experiment_id}"
    if getattr(harness, "APP_NAME", None) != expected_app:
        raise ModalOrchestratorError("offline harness app identity mismatch")
    if len(getattr(harness, "CELL_FUNCTIONS", ())) != len(CELLS):
        raise ModalOrchestratorError("offline harness cell inventory is incomplete")
    return {
        "schema_version": "1",
        "approval_sha256": approval_sha256,
        "git_head": config.git_head,
        "plan_sha256": plan.content_sha256,
        "harness_app": expected_app,
        "cells": [cell.cell_id for cell in CELLS],
        "provider_accessed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen3-vllm-modal-run")
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--offline-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExecutionConfig(
        approval_path=args.approval,
        approval_sha256=args.approval_sha256,
        git_head=args.git_head,
        workspace_path=args.workspace,
        output_dir=args.output_dir,
        ledger_path=args.ledger,
        experiment_id=args.experiment_id,
    )
    try:
        if args.offline_only:
            result = offline_preflight(
                config,
                harness_loader=ImportHarnessLoader(),
                git=SubprocessGitInspector(),
            )
            print(canonical_json(result))
        else:
            execute(
                config,
                provider=SubprocessModalProvider(),
                page_policy=URLPagePolicy(),
                harness_loader=ImportHarnessLoader(),
                git=SubprocessGitInspector(),
            )
    except (ModalOrchestratorError, VLLMCompileContractError) as exc:
        outcome = (
            "failed after provider execution"
            if isinstance(exc, ModalOrchestratorError) and exc.teardown
            else "refused before provider execution"
        )
        print(f"qwen3-vllm-modal-run: {outcome}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
