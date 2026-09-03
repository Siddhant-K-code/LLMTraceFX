"""Fail-closed Modal harness for the approved Qwen3-8B vLLM experiment.

This is an internal ``modal run`` entrypoint, not an importable deployment
library.  Importing it registers resources only after every experiment pin has
been supplied and checked.  The helpers are intentionally dependency-injected
so their safety properties can be tested without Modal, CUDA, vLLM, or network
access.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    CELLS,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    ExperimentCell,
    HardwareIdentity,
    VLLMCompileContractError,
    VLLMCompilePlan,
    canonical_json,
    validate_hardware_identity,
    validate_model_identity,
    workload_descriptors,
)
from llmtracefx.optimizer.workloads.catalog import workload_by_id
from llmtracefx.optimizer.workloads.materialize import materialize_prompt
from llmtracefx.optimizer.workloads.schema import ContextTier

IMAGE_REFERENCE = (
    "vllm/vllm-openai:v0.28.0@"
    "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)
IMAGE_DIGEST = IMAGE_REFERENCE.rsplit("@", 1)[1]
MOUNT_PATH = "/qwen3-compile"
MODEL_DIRECTORY = "model-b968826d9c46dd6066d109eabc6255188de91218"
STAGING_RECEIPT = "staging-receipt.json"
PROMPT_IDS_FILE = "prompt-token-ids.json"
MAX_DECODED_OUTPUT_BYTES = 65_536

PLAN_PATH_ENV = "LLMTRACEFX_QWEN3_COMPILE_PLAN_PATH"
PLAN_JSON_ENV = "LLMTRACEFX_QWEN3_COMPILE_PLAN_JSON"
APP_NAME_ENV = "LLMTRACEFX_QWEN3_COMPILE_APP_NAME"
VOLUME_NAME_ENV = "LLMTRACEFX_QWEN3_COMPILE_VOLUME_NAME"
EXPERIMENT_TAG_ENV = "LLMTRACEFX_QWEN3_COMPILE_EXPERIMENT_TAG"
WORKLOAD_HASH_ENV = "LLMTRACEFX_QWEN3_COMPILE_WORKLOAD_SHA256"
OUTPUT_HASH_ENV = "LLMTRACEFX_QWEN3_COMPILE_OUTPUT_SHA256"

_SAFE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROVENANCE_DOMAINS = frozenset(
    {
        "client_observed",
        "vllm",
        "cuda",
        "modal_provider",
        "model_reported",
        "derived",
    }
)

SAMPLING_CONTRACT: dict[str, Any] = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 20260831,
}
TOKENIZER_CONTRACT: dict[str, Any] = {
    "tokenize": True,
    "add_generation_prompt": True,
    "enable_thinking": False,
    "messages": "single_user_message",
}
OUTPUT_CONTRACT: dict[str, Any] = {
    "schema_version": "1",
    "request_terminal_required": True,
    "finish_reason_required": True,
    "input_count_source": "persisted_prompt_token_ids",
    "output_count_source": "request_output_token_ids",
    "decoded_output_max_utf8_bytes": MAX_DECODED_OUTPUT_BYTES,
    "remote_correctness_evaluation": False,
    "resolved_execution_config_required": True,
    "missing_timing_reason_required": True,
    "provenance_domains": sorted(_PROVENANCE_DOMAINS),
}


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


WORKLOAD_CONTRACT: dict[str, Any] = {
    "schema_version": "1",
    "descriptors": [item.to_dict() for item in workload_descriptors()],
    "sampling": SAMPLING_CONTRACT,
    "tokenizer": TOKENIZER_CONTRACT,
}
WORKLOAD_CONTRACT_SHA256 = _sha256_json(WORKLOAD_CONTRACT)
OUTPUT_CONTRACT_SHA256 = _sha256_json(OUTPUT_CONTRACT)
IMAGE_CONTRACT_SHA256 = _sha256_json({"reference": IMAGE_REFERENCE})


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "")
    if not isinstance(value, str) or not value.strip():
        raise VLLMCompileContractError(f"required environment pin {name} is missing")
    return value.strip()


def _safe_name(value: str, *, field: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise VLLMCompileContractError(
            f"{field} must be a safe lowercase Modal name (1-63 characters)"
        )
    return value


def _sanitize_tag(value: str) -> str:
    sanitized = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")
    return _safe_name(sanitized, field="experiment tag")


def _load_plan(environ: Mapping[str, str]) -> VLLMCompilePlan:
    path_value = environ.get(PLAN_PATH_ENV, "").strip()
    json_value = environ.get(PLAN_JSON_ENV, "").strip()
    if bool(path_value) == bool(json_value):
        raise VLLMCompileContractError(
            f"supply exactly one of {PLAN_PATH_ENV} or {PLAN_JSON_ENV}"
        )
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            raise VLLMCompileContractError("plan path must be absolute")
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise VLLMCompileContractError(
                f"cannot read canonical plan: {exc}"
            ) from exc
    else:
        payload = json_value
    plan = VLLMCompilePlan.from_json(payload)
    if payload != plan.to_json():
        raise VLLMCompileContractError("plan JSON must use exact canonical encoding")
    if plan.image_digest != IMAGE_DIGEST:
        raise VLLMCompileContractError("plan does not select the approved amd64 image")
    pins = plan.runtime_pins.to_dict()
    if pins["vllm_version"] != "0.28.0":
        raise VLLMCompileContractError("runtime must pin vLLM exactly 0.28.0")
    return plan


def _load_import_config(
    environ: Mapping[str, str],
) -> tuple[VLLMCompilePlan, str, str, str]:
    plan = _load_plan(environ)
    app_base = _safe_name(_required(environ, APP_NAME_ENV), field="app name")
    volume_name = _safe_name(_required(environ, VOLUME_NAME_ENV), field="volume name")
    tag = _sanitize_tag(_required(environ, EXPERIMENT_TAG_ENV))
    app_name = _safe_name(f"{app_base}-{tag}", field="tagged app name")
    if app_name == volume_name:
        raise VLLMCompileContractError("app and volume names must be unique")
    workload_hash = _required(environ, WORKLOAD_HASH_ENV)
    output_hash = _required(environ, OUTPUT_HASH_ENV)
    if (
        not _SHA256.fullmatch(workload_hash)
        or workload_hash != WORKLOAD_CONTRACT_SHA256
    ):
        raise VLLMCompileContractError("workload contract hash is absent or incorrect")
    if not _SHA256.fullmatch(output_hash) or output_hash != OUTPUT_CONTRACT_SHA256:
        raise VLLMCompileContractError("output contract hash is absent or incorrect")
    return plan, app_name, volume_name, tag


# Deliberately before importing Modal: a missing pin cannot register any resource.
PLAN, APP_NAME, VOLUME_NAME, EXPERIMENT_TAG = _load_import_config(os.environ)

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - only a real entrypoint failure
    raise SystemExit(
        "Install the Modal dependency with `uv sync --extra modal` or "
        "`pip install 'llmtracefx[modal]'`."
    ) from exc

BAKED_ENVIRONMENT = {
    PLAN_JSON_ENV: PLAN.to_json(),
    APP_NAME_ENV: APP_NAME.rsplit(f"-{EXPERIMENT_TAG}", 1)[0],
    VOLUME_NAME_ENV: VOLUME_NAME,
    EXPERIMENT_TAG_ENV: EXPERIMENT_TAG,
    WORKLOAD_HASH_ENV: WORKLOAD_CONTRACT_SHA256,
    OUTPUT_HASH_ENV: OUTPUT_CONTRACT_SHA256,
    "PYTHONPATH": "/opt/llmtracefx",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}

app = modal.App(
    APP_NAME,
    tags={
        "experiment": EXPERIMENT_TAG,
        "project": "llmtracefx-vllm-compile",
    },
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.from_registry(
        IMAGE_REFERENCE,
        setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3 /usr/local/bin/python"],
    )
    .pip_install("typing_extensions==4.15.0")
    .entrypoint([])
    .env(BAKED_ENVIRONMENT)
    .add_local_dir(
        str(Path(__file__).resolve().parents[1]),
        remote_path="/opt/llmtracefx/llmtracefx",
        copy=True,
    )
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _event(event: str, provenance: str, **values: Any) -> dict[str, Any]:
    if provenance not in _PROVENANCE_DOMAINS:
        raise VLLMCompileContractError("invalid provenance domain")
    return {"event": event, "provenance": provenance, "observed_at": _now(), **values}


def _seal(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[hash_field] = _sha256_json(result)
    return result


def _verify_seal(payload: Mapping[str, Any], hash_field: str) -> None:
    expected = payload.get(hash_field)
    unsealed = dict(payload)
    unsealed.pop(hash_field, None)
    if not isinstance(expected, str) or expected != _sha256_json(unsealed):
        raise VLLMCompileContractError(f"{hash_field} verification failed")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    data = canonical_json(payload).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _conversion_manifest() -> dict[str, Any]:
    path = (
        Path(__file__).parents[1]
        / "optimizer"
        / "lab"
        / "qwen3_8b"
        / "data"
        / "qwen3-8b-conversion-manifest-v1.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise VLLMCompileContractError("packaged conversion manifest is invalid")
    return payload


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_inventory(model_path: Path) -> list[dict[str, Any]]:
    source = _conversion_manifest().get("source")
    if not isinstance(source, dict) or source.get("official_id") != MODEL_ID:
        raise VLLMCompileContractError("conversion manifest source identity is invalid")
    if source.get("official_revision") != MODEL_REVISION:
        raise VLLMCompileContractError("conversion manifest revision is invalid")
    expected_items = source.get("files")
    if source.get("expected_source_bytes") != EXPECTED_MODEL_BYTES:
        raise VLLMCompileContractError("conversion manifest byte total is invalid")
    if not isinstance(expected_items, list):
        raise VLLMCompileContractError("conversion manifest inventory is invalid")
    expected = {
        item["path"]: item
        for item in expected_items
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if len(expected) != EXPECTED_MODEL_FILE_COUNT:
        raise VLLMCompileContractError("conversion manifest must contain 15 files")
    observed: set[str] = set()
    for path in model_path.rglob("*"):
        if path.is_symlink():
            raise VLLMCompileContractError(
                f"model inventory cannot contain symlinks: "
                f"{path.relative_to(model_path).as_posix()}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise VLLMCompileContractError("model inventory must contain regular files")
        relative = path.relative_to(model_path).as_posix()
        if relative.startswith(".cache/huggingface/"):
            continue
        if relative not in expected:
            raise VLLMCompileContractError(f"unexpected model file: {relative}")
        observed.add(relative)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        raise VLLMCompileContractError(f"model inventory is incomplete: {missing}")
    verified: list[dict[str, Any]] = []
    total = 0
    for relative in sorted(expected):
        pin = expected[relative]
        path = model_path / relative
        size = path.stat().st_size
        digest = _hash_file(path)
        if size != pin.get("size_bytes") or digest != pin.get("sha256"):
            raise VLLMCompileContractError(
                f"model file verification failed: {relative}"
            )
        total += size
        verified.append({"path": relative, "size_bytes": size, "sha256": digest})
    if total != EXPECTED_MODEL_BYTES:
        raise VLLMCompileContractError("verified model byte total is incorrect")
    return verified


def _materialize_token_ids(
    model_path: Path, tokenizer_factory: Callable[..., Any]
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    tokenizer = tokenizer_factory(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    descriptors = workload_descriptors()
    token_ids: dict[str, list[int]] = {}
    prompt_records: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if descriptor.repetition != 1:
            continue
        prompt = materialize_prompt(
            workload_by_id(descriptor.workload_id),
            ContextTier(descriptor.context_tier),
        )
        if prompt.prompt_hash != descriptor.prompt_sha256:
            raise VLLMCompileContractError(
                f"source prompt hash mismatch for {descriptor.request_id}"
            )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise VLLMCompileContractError("tokenizer did not return exact token IDs")
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        token_ids[key] = ids
        decoded_prompt = tokenizer.decode(ids, skip_special_tokens=False)
        if not isinstance(decoded_prompt, str) or not decoded_prompt:
            raise VLLMCompileContractError(
                "tokenizer did not decode the staged prompt token IDs"
            )
        prompt_records.append(
            {
                "key": key,
                "prompt_sha256": descriptor.prompt_sha256,
                "prompt_token_ids_sha256": _sha256_json(ids),
                "prompt_token_ids": ids,
                "input_token_count": len(ids),
                "decoded_prompt_sha256": "sha256:"
                + hashlib.sha256(decoded_prompt.encode("utf-8")).hexdigest(),
            }
        )
    if len(token_ids) != 6:
        raise VLLMCompileContractError("exactly six tokenized prompts are required")
    return prompt_records, token_ids


def _stage_impl(
    *,
    snapshot_download: Callable[..., str],
    tokenizer_factory: Callable[..., Any],
    mount_path: Path,
) -> dict[str, Any]:
    model_path = mount_path / MODEL_DIRECTORY
    downloaded = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_path),
        token=False,
    )
    if Path(downloaded).resolve() != model_path.resolve():
        raise VLLMCompileContractError(
            "download materialized outside the bound directory"
        )
    inventory = _verify_model_inventory(model_path)
    validate_model_identity(
        observed_revision=MODEL_REVISION,
        observed_file_count=len(inventory),
        observed_bytes=sum(item["size_bytes"] for item in inventory),
    )
    prompts, token_ids = _materialize_token_ids(model_path, tokenizer_factory)
    token_payload = _seal(
        {
            "schema_version": "1",
            "workload_sha256": WORKLOAD_CONTRACT_SHA256,
            "prompts": token_ids,
        },
        "prompt_ids_sha256",
    )
    _atomic_json(mount_path / PROMPT_IDS_FILE, token_payload)
    receipt = _seal(
        {
            "schema_version": "1",
            "plan_sha256": PLAN.content_sha256,
            "workload_sha256": WORKLOAD_CONTRACT_SHA256,
            "output_contract_sha256": OUTPUT_CONTRACT_SHA256,
            "runtime_sha256": _sha256_json(PLAN.runtime_pins.to_dict()),
            "image_sha256": IMAGE_CONTRACT_SHA256,
            "image_digest": IMAGE_DIGEST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_file_count": len(inventory),
            "model_bytes": sum(item["size_bytes"] for item in inventory),
            "inventory": inventory,
            "prompts": prompts,
            "prompt_ids_sha256": token_payload["prompt_ids_sha256"],
            "staged_at": _now(),
        },
        "receipt_sha256",
    )
    _atomic_json(mount_path / STAGING_RECEIPT, receipt)
    volume.commit()
    return receipt


@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    cpu=4,
    memory=32 * 1024,
    timeout=2700,
    retries=0,
    max_containers=1,
    min_containers=0,
    single_use_containers=True,
)
@modal.concurrent(max_inputs=1)
def stage_qwen3() -> dict[str, Any]:
    """Download and attest the exact public revision on CPU only."""

    _verify_runtime(_observe_runtime())
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    from vllm.config import CompilationConfig  # type: ignore[import-not-found]
    from vllm.config.compilation import (  # type: ignore[import-not-found]
        CompilationMode,
        CUDAGraphMode,
    )

    _ = CompilationConfig, CompilationMode, CUDAGraphMode
    return _stage_impl(
        snapshot_download=snapshot_download,
        tokenizer_factory=AutoTokenizer.from_pretrained,
        mount_path=Path(MOUNT_PATH),
    )


def _run_command(argv: Sequence[str]) -> str:
    completed = subprocess.run(
        list(argv),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout


def _query_hardware(command_runner: Callable[[Sequence[str]], str]) -> dict[str, Any]:
    output = command_runner(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        )
    )
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise VLLMCompileContractError("nvidia-smi must report exactly one GPU")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 4 or not fields[0]:
        raise VLLMCompileContractError("nvidia-smi hardware identity is incomplete")

    def optional_number(value: str) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def optional_text(value: str) -> str | None:
        return None if value.lower() in {"", "n/a", "[n/a]", "not supported"} else value

    driver_version = optional_text(fields[1])
    memory_total_mib = optional_number(fields[2])
    memory_used_mib = optional_number(fields[3])
    if (
        driver_version is None
        or memory_total_mib is None
        or memory_total_mib <= 0
        or memory_used_mib is None
        or memory_used_mib < 0
        or memory_used_mib > memory_total_mib
    ):
        raise VLLMCompileContractError("nvidia-smi hardware identity is incomplete")
    return {
        "gpu_name": fields[0],
        "gpu_count": 1,
        "driver_version": driver_version,
        "memory_total_mib": memory_total_mib,
        "memory_used_mib": memory_used_mib,
    }


def _observe_runtime() -> dict[str, str | None]:
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError):
        torch = None
    return {
        "python_version": ".".join(str(item) for item in sys.version_info[:2]),
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": importlib.metadata.version("torch"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "typing_extensions_version": importlib.metadata.version("typing_extensions"),
    }


def _verify_runtime(observed: Mapping[str, str | None]) -> None:
    expected = PLAN.runtime_pins.to_dict()
    if dict(observed) != expected:
        raise VLLMCompileContractError(
            f"runtime pins differ from plan: expected {expected!r}, observed {dict(observed)!r}"
        )


def _read_canonical(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, ValueError) as exc:
        raise VLLMCompileContractError(
            f"cannot read staged artifact {path.name}"
        ) from exc
    if not isinstance(payload, dict) or canonical_json(payload) != text:
        raise VLLMCompileContractError(f"{path.name} is not canonical JSON")
    return payload


def _load_staging(mount_path: Path) -> tuple[dict[str, Any], dict[str, list[int]]]:
    receipt = _read_canonical(mount_path / STAGING_RECEIPT)
    _verify_seal(receipt, "receipt_sha256")
    expected = {
        "plan_sha256": PLAN.content_sha256,
        "workload_sha256": WORKLOAD_CONTRACT_SHA256,
        "output_contract_sha256": OUTPUT_CONTRACT_SHA256,
        "runtime_sha256": _sha256_json(PLAN.runtime_pins.to_dict()),
        "image_sha256": IMAGE_CONTRACT_SHA256,
        "image_digest": IMAGE_DIGEST,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_count": EXPECTED_MODEL_FILE_COUNT,
        "model_bytes": EXPECTED_MODEL_BYTES,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise VLLMCompileContractError(f"staging receipt {field} is stale")
    prompt_payload = _read_canonical(mount_path / PROMPT_IDS_FILE)
    _verify_seal(prompt_payload, "prompt_ids_sha256")
    if prompt_payload.get("prompt_ids_sha256") != receipt.get("prompt_ids_sha256"):
        raise VLLMCompileContractError("persisted prompt token IDs are stale")
    prompts = prompt_payload.get("prompts")
    if not isinstance(prompts, dict):
        raise VLLMCompileContractError("persisted prompt token IDs are missing")
    result: dict[str, list[int]] = {}
    for key, ids in prompts.items():
        if (
            not isinstance(key, str)
            or not isinstance(ids, list)
            or not ids
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise VLLMCompileContractError("persisted prompt token IDs are invalid")
        result[key] = ids
    if len(result) != 6:
        raise VLLMCompileContractError("persisted prompt token set is incomplete")
    return receipt, result


class _MemorySampler:
    def __init__(self, command_runner: Callable[[Sequence[str]], str]) -> None:
        self._runner = command_runner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_mib: float | None = None

    def _observe(self) -> None:
        output = self._runner(
            (
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            )
        ).strip()
        try:
            value = float(output)
        except (TypeError, ValueError) as exc:
            raise VLLMCompileContractError("GPU memory sample is invalid") from exc
        if not math.isfinite(value) or value < 0:
            raise VLLMCompileContractError("GPU memory sample is invalid")
        self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            try:
                self._observe()
            except (
                OSError,
                subprocess.SubprocessError,
                TypeError,
                ValueError,
                VLLMCompileContractError,
            ):
                continue

    def start(self) -> None:
        self._observe()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        try:
            self._observe()
        except (
            OSError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
            VLLMCompileContractError,
        ):
            pass


def _metric_ttft(metrics: Any) -> float | None:
    direct = getattr(metrics, "first_token_latency", None)
    if (
        isinstance(direct, (int, float))
        and not isinstance(direct, bool)
        and math.isfinite(direct)
        and direct >= 0
    ):
        return float(direct)
    return None


def _request_record(
    *,
    descriptor: Any,
    ids: list[int],
    response: Any,
    started_at: str,
    ended_at: str,
    elapsed: float,
) -> dict[str, Any]:
    if getattr(response, "finished", None) is not True:
        raise VLLMCompileContractError("vLLM request did not reach a terminal state")
    outputs = getattr(response, "outputs", None)
    if not isinstance(outputs, Sequence) or len(outputs) != 1:
        raise VLLMCompileContractError("vLLM must return exactly one completion")
    completion = outputs[0]
    finish_reason = getattr(completion, "finish_reason", None)
    if finish_reason not in {"stop", "length"}:
        raise VLLMCompileContractError(
            "terminal completion has no valid stop or length finish reason"
        )
    output_ids = getattr(completion, "token_ids", None)
    if (
        not isinstance(output_ids, Sequence)
        or isinstance(output_ids, (str, bytes))
        or not output_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in output_ids
        )
    ):
        raise VLLMCompileContractError(
            "terminal completion has invalid output token IDs"
        )
    text = getattr(completion, "text", None)
    if (
        not isinstance(text, str)
        or len(text.encode("utf-8")) > MAX_DECODED_OUTPUT_BYTES
    ):
        raise VLLMCompileContractError("decoded output is absent or exceeds its bound")
    return {
        **descriptor.to_dict(),
        "terminal": True,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_seconds": elapsed,
        "input_token_count": len(ids),
        "input_token_ids_sha256": _sha256_json(ids),
        "output_token_count": len(output_ids),
        "output_tokens_per_second": (
            len(output_ids) / elapsed if elapsed > 0 else None
        ),
        "output_rate_basis": "output_tokens_per_complete_response_second",
        "output_token_ids": list(output_ids),
        "decoded_output": text,
        "finish_reason": finish_reason,
        "ttft_seconds": _metric_ttft(getattr(response, "metrics", None)),
        "evaluator_input": {
            "workload_id": descriptor.workload_id,
            "context_tier": descriptor.context_tier,
            "decoded_output": text,
            "output_token_ids": list(output_ids),
        },
        "correctness": None,
        "provenance": "model_reported",
        "field_provenance": {
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
        },
    }


def _construct_llm(vllm_module: Any, cell: ExperimentCell, model_path: Path) -> Any:
    from vllm.config import CompilationConfig
    from vllm.config.compilation import (
        CompilationMode,
        CUDAGraphMode,
    )

    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "disable_log_stats": False,
        "gpu_memory_utilization": 0.85,
    }
    if cell.compile_enabled:
        kwargs.update(
            enforce_eager=False,
            compilation_config=CompilationConfig(
                mode=CompilationMode.VLLM_COMPILE,
                cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            ),
        )
    else:
        kwargs.update(
            enforce_eager=True,
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE,
                cudagraph_mode=CUDAGraphMode.NONE,
            ),
        )
    return vllm_module.LLM(**kwargs)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    if isinstance(value, str) and value:
        return value.rsplit(".", 1)[-1]
    raise VLLMCompileContractError("resolved vLLM mode is unavailable")


def _resolved_execution_config(llm: Any, cell: ExperimentCell) -> dict[str, Any]:
    config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    model_config = getattr(config, "model_config", None)
    compilation = getattr(config, "compilation_config", None)
    resolved = {
        "enforce_eager": getattr(model_config, "enforce_eager", None),
        "compilation_mode": _enum_name(getattr(compilation, "mode", None)),
        "cuda_graph_mode": _enum_name(getattr(compilation, "cudagraph_mode", None)),
    }
    expected = (
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
    if resolved != expected:
        raise VLLMCompileContractError(
            f"resolved vLLM execution config mismatch: {resolved!r} != {expected!r}"
        )
    return resolved


def _observed_compilation_seconds(llm: Any) -> float | None:
    """Read the stable v0.28 CompilationConfig accumulator when exposed."""

    config = getattr(getattr(llm, "llm_engine", None), "vllm_config", None)
    compilation = getattr(config, "compilation_config", None)
    value = getattr(compilation, "compilation_time", None)
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    ):
        return float(value)
    return None


def _run_cell(
    cell: ExperimentCell,
    *,
    mount_path: Path = Path(MOUNT_PATH),
    command_runner: Callable[[Sequence[str]], str] = _run_command,
    runtime_observer: Callable[[], Mapping[str, str | None]] = _observe_runtime,
    vllm_module: Any | None = None,
    llm_factory: Callable[[Any, ExperimentCell, Path], Any] = _construct_llm,
    sampling_factory: Callable[..., Any] | None = None,
    tokens_prompt_factory: Callable[..., Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    yield _event("container_started", "modal_provider", cell_id=cell.cell_id)
    os.environ.update(
        {
            "DO_NOT_TRACK": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
        }
    )
    hardware = _query_hardware(command_runner)
    validate_hardware_identity(
        cell, HardwareIdentity(hardware["gpu_name"], hardware["gpu_count"])
    )
    yield _event("hardware_validated", "cuda", cell_id=cell.cell_id, **hardware)

    observed_runtime = dict(runtime_observer())
    _verify_runtime(observed_runtime)
    receipt, prompt_ids = _load_staging(mount_path)
    if vllm_module is None:
        vllm_module = importlib.import_module("vllm")
    if sampling_factory is None:
        sampling_factory = vllm_module.SamplingParams
    if tokens_prompt_factory is None:
        from vllm.inputs import TokensPrompt  # type: ignore[import-not-found]

        tokens_prompt_factory = TokensPrompt

    sampler = _MemorySampler(command_runner)
    sampler.start()
    initialization_started = _now()
    yield _event(
        "initialization_started",
        "client_observed",
        cell_id=cell.cell_id,
        compilation_seconds=None,
        cuda_graph_seconds=None,
    )
    try:
        llm = llm_factory(vllm_module, cell, mount_path / MODEL_DIRECTORY)
        resolved_execution_config = _resolved_execution_config(llm, cell)
        initialization_ready = _now()
        yield _event(
            "initialization_ready",
            "vllm",
            cell_id=cell.cell_id,
            compilation_seconds=None,
            cuda_graph_seconds=None,
        )
        sampling = sampling_factory(**SAMPLING_CONTRACT)
        records: list[dict[str, Any]] = []
        for descriptor in workload_descriptors():
            key = f"{descriptor.context_tier}/{descriptor.workload_id}"
            ids = prompt_ids[key]
            started_at = _now()
            started = time.monotonic()
            generated = llm.generate(
                [tokens_prompt_factory(prompt_token_ids=list(ids))],
                sampling,
                use_tqdm=False,
            )
            elapsed = time.monotonic() - started
            ended_at = _now()
            if not isinstance(generated, Sequence) or len(generated) != 1:
                raise VLLMCompileContractError("vLLM returned an invalid request batch")
            record = _request_record(
                descriptor=descriptor,
                ids=ids,
                response=generated[0],
                started_at=started_at,
                ended_at=ended_at,
                elapsed=elapsed,
            )
            records.append(record)
            yield _event(
                "request_terminal",
                "model_reported",
                cell_id=cell.cell_id,
                request=record,
            )
    finally:
        sampler.stop()

    if len(records) != 12 or any(not item["terminal"] for item in records):
        raise VLLMCompileContractError(
            "cell did not produce 12 complete terminal requests"
        )
    if sampler.peak_mib is None or sampler.peak_mib <= 0:
        raise VLLMCompileContractError("peak GPU memory is unavailable")
    compilation_seconds = (
        _observed_compilation_seconds(llm) if cell.compile_enabled else None
    )
    compilation_seconds_unobservable_reason = (
        None
        if compilation_seconds is not None
        else (
            "vllm_compilation_time_not_exposed_or_nonpositive"
            if cell.compile_enabled
            else "not_applicable_eager_mode"
        )
    )
    terminal = _seal(
        {
            "schema_version": "1",
            "cell": cell.to_dict(),
            "plan_sha256": PLAN.content_sha256,
            "staging_receipt_sha256": receipt["receipt_sha256"],
            "workload_sha256": WORKLOAD_CONTRACT_SHA256,
            "output_contract_sha256": OUTPUT_CONTRACT_SHA256,
            "runtime_sha256": _sha256_json(observed_runtime),
            "image_sha256": IMAGE_CONTRACT_SHA256,
            "hardware": hardware,
            "runtime": observed_runtime,
            "resolved_execution_config": resolved_execution_config,
            "initialization_started_at": initialization_started,
            "initialization_ready_at": initialization_ready,
            "compilation_seconds": compilation_seconds,
            "compilation_seconds_unobservable_reason": (
                compilation_seconds_unobservable_reason
            ),
            "cuda_graph_seconds": None,
            "cuda_graph_seconds_unobservable_reason": (
                "stable_component_timing_not_exposed"
                if cell.compile_enabled
                else "not_applicable_eager_mode"
            ),
            "peak_gpu_memory_mib": sampler.peak_mib,
            "requests": records,
            "correctness_evaluated_remotely": False,
            "terminal": True,
        },
        "cell_sha256",
    )
    yield _event("cell_terminal", "derived", record=terminal)


_GPU_COMMON: dict[str, Any] = {
    "image": image,
    "volumes": {MOUNT_PATH: volume},
    "cpu": 4,
    "memory": 32 * 1024,
    "timeout": 2700,
    "max_containers": 1,
    "min_containers": 0,
    "single_use_containers": True,
    "enable_memory_snapshot": False,
    "block_network": True,
    "restrict_modal_access": True,
}


@app.function(gpu="L40S", **_GPU_COMMON)
@modal.concurrent(max_inputs=1)
def l40s_eager() -> Generator[dict[str, Any], None, None]:
    yield from _run_cell(CELLS[0])


@app.function(gpu="L40S", **_GPU_COMMON)
@modal.concurrent(max_inputs=1)
def l40s_compiled() -> Generator[dict[str, Any], None, None]:
    yield from _run_cell(CELLS[1])


@app.function(gpu="H100!", **_GPU_COMMON)
@modal.concurrent(max_inputs=1)
def h100_eager() -> Generator[dict[str, Any], None, None]:
    yield from _run_cell(CELLS[2])


@app.function(gpu="H100!", **_GPU_COMMON)
@modal.concurrent(max_inputs=1)
def h100_compiled() -> Generator[dict[str, Any], None, None]:
    yield from _run_cell(CELLS[3])


STAGE_FUNCTION = stage_qwen3
CELL_FUNCTIONS = (
    l40s_eager,
    l40s_compiled,
    h100_eager,
    h100_compiled,
)
