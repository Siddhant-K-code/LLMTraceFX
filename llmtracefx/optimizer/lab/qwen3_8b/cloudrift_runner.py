"""Bounded CloudRift RTX 4090 runner for the Qwen3-8B compile experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    VLLMCompileContractError,
    canonical_json,
    workload_descriptors,
)
from llmtracefx.optimizer.workloads.catalog import workload_by_id
from llmtracefx.optimizer.workloads.materialize import materialize_prompt
from llmtracefx.optimizer.workloads.schema import ContextTier

BASE_IMAGE_REFERENCE = (
    "vllm/vllm-openai:v0.28.0@"
    "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)
DERIVED_IMAGE_ID = (
    "sha256:fd34de17a99d2890ed1031fd32fff4c74837bbc92df7dcb955caf610266cffb3"
)
RUNTIME_PINS = {
    "python_version": "3.12",
    "vllm_version": "0.28.0",
    "torch_version": "2.13.0+cu130",
    "cuda_version": "13.0",
    "transformers_version": "5.15.1",
    "typing_extensions_version": "4.15.0",
}
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_MEMORY_MIB = 24564
EXPECTED_DRIVER = "580.159.03"
MODEL_DIRECTORY = "model-b968826d9c46dd6066d109eabc6255188de91218"
STAGING_FILE = "staging-receipt.json"
PROMPT_FILE = "prompt-token-ids.json"
MAX_OUTPUT_BYTES = 65_536
SAMPLING = {
    "max_tokens": 96,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 20260831,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = _sha256_json(result)
    return result


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    expected = value.get(field)
    material = dict(value)
    material.pop(field, None)
    if expected != _sha256_json(material):
        raise VLLMCompileContractError(f"{field} verification failed")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    data = canonical_json(value).encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict) or canonical_json(value) != text:
        raise VLLMCompileContractError(f"{path.name} is not canonical JSON")
    return value


def _manifest() -> dict[str, Any]:
    path = Path(__file__).parent / "data" / "qwen3-8b-conversion-manifest-v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VLLMCompileContractError("model manifest is invalid")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(model_path: Path) -> list[dict[str, Any]]:
    source = _manifest().get("source")
    if (
        not isinstance(source, dict)
        or source.get("official_id") != MODEL_ID
        or source.get("official_revision") != MODEL_REVISION
        or source.get("expected_source_bytes") != EXPECTED_MODEL_BYTES
    ):
        raise VLLMCompileContractError("model manifest identity is invalid")
    files = source.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_MODEL_FILE_COUNT:
        raise VLLMCompileContractError("model manifest must contain exactly 15 files")
    expected = {item["path"]: item for item in files}
    observed = {
        path.relative_to(model_path).as_posix()
        for path in model_path.rglob("*")
        if path.is_file() and ".cache/huggingface/" not in path.as_posix()
    }
    if observed != set(expected):
        raise VLLMCompileContractError("downloaded model inventory is incomplete")
    verified: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = model_path / relative
        item = expected[relative]
        if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
            raise VLLMCompileContractError(f"model file size mismatch: {relative}")
        digest = _hash_file(path)
        if digest != item["sha256"]:
            raise VLLMCompileContractError(f"model file hash mismatch: {relative}")
        verified.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    if sum(item["size_bytes"] for item in verified) != EXPECTED_MODEL_BYTES:
        raise VLLMCompileContractError("model byte total is invalid")
    return verified


def _tokenize(model_path: Path) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    prompt_ids: dict[str, list[int]] = {}
    records: list[dict[str, Any]] = []
    for descriptor in workload_descriptors():
        if descriptor.repetition != 1:
            continue
        prompt = materialize_prompt(
            workload_by_id(descriptor.workload_id),
            ContextTier(descriptor.context_tier),
        )
        if prompt.prompt_hash != descriptor.prompt_sha256:
            raise VLLMCompileContractError("source prompt hash mismatch")
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(item, bool) or not isinstance(item, int) for item in ids)
        ):
            raise VLLMCompileContractError("tokenizer did not return flat token IDs")
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        prompt_ids[key] = ids
        records.append(
            {
                "key": key,
                "prompt_sha256": descriptor.prompt_sha256,
                "input_token_count": len(ids),
                "prompt_token_ids_sha256": _sha256_json(ids),
            }
        )
    if len(records) != 6:
        raise VLLMCompileContractError("tokenizer did not produce six prompts")
    return records, prompt_ids


def _runtime() -> dict[str, str | None]:
    torch = importlib.import_module("torch")
    return {
        "python_version": ".".join(str(part) for part in sys.version_info[:2]),
        "vllm_version": importlib.metadata.version("vllm"),
        "torch_version": importlib.metadata.version("torch"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "transformers_version": importlib.metadata.version("transformers"),
        "typing_extensions_version": importlib.metadata.version("typing_extensions"),
    }


def _verify_runtime() -> dict[str, str | None]:
    observed = _runtime()
    if observed != RUNTIME_PINS:
        raise VLLMCompileContractError(
            f"runtime mismatch: expected {RUNTIME_PINS!r}, observed {observed!r}"
        )
    return observed


def _command(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), check=True, capture_output=True, text=True, shell=False
    ).stdout


def _hardware() -> dict[str, Any]:
    fields = (
        _command(
            (
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,uuid",
                "--format=csv,noheader,nounits",
            )
        )
        .strip()
        .split(", ")
    )
    if len(fields) != 5:
        raise VLLMCompileContractError("GPU identity is incomplete")
    name, driver, total, used, gpu_uuid = fields
    if (
        name != EXPECTED_GPU_NAME
        or driver != EXPECTED_DRIVER
        or int(total) != EXPECTED_MEMORY_MIB
        or not gpu_uuid.startswith("GPU-")
    ):
        raise VLLMCompileContractError("GPU identity differs from approved VM")
    return {
        "gpu_name": name,
        "gpu_count": 1,
        "driver_version": driver,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "gpu_uuid_sha256": "sha256:" + hashlib.sha256(gpu_uuid.encode()).hexdigest(),
    }


class _MemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.peak_mib: int | None = None

    def _observe(self) -> None:
        value = int(
            _command(
                (
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                )
            ).strip()
        )
        self.peak_mib = value if self.peak_mib is None else max(self.peak_mib, value)

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            try:
                self._observe()
            except (OSError, ValueError, subprocess.SubprocessError):
                continue

    def start(self) -> None:
        self._observe()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._observe()


def tokenizer_canary(model_path: Path, output: Path) -> None:
    from huggingface_hub import snapshot_download

    files = _manifest()["source"]["files"]
    allow = [
        item["path"] for item in files if not item["path"].endswith(".safetensors")
    ]
    started_at = _now()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_path),
        allow_patterns=allow,
        token=False,
    )
    records, ids = _tokenize(model_path)
    _atomic_json(
        output,
        _seal(
            {
                "schema_version": "1",
                "kind": "cloudrift_tokenizer_canary",
                "started_at": started_at,
                "ended_at": _now(),
                "runtime": _verify_runtime(),
                "prompts": records,
                "prompt_ids_sha256": _sha256_json(ids),
                "terminal": True,
            },
            "artifact_sha256",
        ),
    )


def stage(model_path: Path, state_path: Path) -> None:
    from huggingface_hub import snapshot_download

    started_at = _now()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_path),
        token=False,
    )
    inventory = _inventory(model_path)
    prompts, ids = _tokenize(model_path)
    prompt_payload = _seal({"schema_version": "1", "prompts": ids}, "prompt_ids_sha256")
    _atomic_json(state_path / PROMPT_FILE, prompt_payload)
    _atomic_json(
        state_path / STAGING_FILE,
        _seal(
            {
                "schema_version": "1",
                "provider": "cloudrift",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_file_count": len(inventory),
                "model_bytes": sum(item["size_bytes"] for item in inventory),
                "inventory": inventory,
                "prompts": prompts,
                "prompt_ids_sha256": prompt_payload["prompt_ids_sha256"],
                "runtime": _verify_runtime(),
                "runtime_image": {
                    "base_reference": BASE_IMAGE_REFERENCE,
                    "derived_image_id": DERIVED_IMAGE_ID,
                    "overlay": ["typing_extensions==4.15.0"],
                },
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        ),
    )


def _metric_ttft(metrics: Any) -> float | None:
    direct = getattr(metrics, "first_token_latency", None)
    if isinstance(direct, (int, float)) and math.isfinite(direct) and direct >= 0:
        return float(direct)
    first = getattr(metrics, "first_token_ts", None)
    arrival = getattr(metrics, "arrival_time", None)
    if isinstance(first, (int, float)) and isinstance(arrival, (int, float)):
        value = float(first) - float(arrival)
        return value if math.isfinite(value) and value >= 0 else None
    return None


def _resolved(llm: Any, compiled: bool) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    observed = {
        "enforce_eager": config.model_config.enforce_eager,
        "compilation_mode": compilation.mode.name,
        "cuda_graph_mode": compilation.cudagraph_mode.name,
    }
    expected = (
        {
            "enforce_eager": False,
            "compilation_mode": "VLLM_COMPILE",
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
        }
        if compiled
        else {
            "enforce_eager": True,
            "compilation_mode": "NONE",
            "cuda_graph_mode": "NONE",
        }
    )
    if observed != expected:
        raise VLLMCompileContractError(
            f"resolved execution config mismatch: {observed!r}"
        )
    return observed


def _verify_staging_binding(
    staging: Mapping[str, Any],
    prompts: Mapping[str, Any],
    model_path: Path,
) -> None:
    if staging["model_revision"] != MODEL_REVISION:
        raise VLLMCompileContractError("staging receipt is stale")
    if staging["prompt_ids_sha256"] != prompts["prompt_ids_sha256"]:
        raise VLLMCompileContractError("staging and prompt receipts differ")
    if staging["inventory"] != _inventory(model_path):
        raise VLLMCompileContractError("live model inventory differs from staging")


def run_cell(mode: str, model_path: Path, state_path: Path, output: Path) -> None:
    from vllm import LLM, SamplingParams  # type: ignore[import-not-found]
    from vllm.config import CompilationConfig  # type: ignore[import-not-found]
    from vllm.config.compilation import (  # type: ignore[import-not-found]
        CompilationMode,
        CUDAGraphMode,
    )
    from vllm.inputs import TokensPrompt  # type: ignore[import-not-found]

    compiled = mode == "compiled"
    host_invocation_started_at = os.environ.get("CLOUDRIFT_HOST_INVOCATION_STARTED_AT")
    if host_invocation_started_at is None:
        raise VLLMCompileContractError("host invocation lifecycle boundary is missing")
    datetime.fromisoformat(host_invocation_started_at.replace("Z", "+00:00"))
    runtime = _verify_runtime()
    hardware = _hardware()
    staging = _read_json(state_path / STAGING_FILE)
    prompts = _read_json(state_path / PROMPT_FILE)
    _verify_seal(staging, "receipt_sha256")
    _verify_seal(prompts, "prompt_ids_sha256")
    _verify_staging_binding(staging, prompts, model_path)
    prompt_ids = prompts["prompts"]
    maximum = max(len(ids) for ids in prompt_ids.values()) + SAMPLING["max_tokens"]
    process_started_at = _now()
    sampler = _MemorySampler()
    sampler.start()
    initialization_started_at = _now()
    try:
        llm = LLM(
            model=str(model_path),
            trust_remote_code=False,
            dtype="bfloat16",
            max_model_len=maximum,
            max_num_seqs=1,
            gpu_memory_utilization=0.94,
            enable_prefix_caching=False,
            disable_custom_all_reduce=True,
            disable_log_stats=False,
            seed=SAMPLING["seed"],
            enforce_eager=not compiled,
            compilation_config=CompilationConfig(
                mode=(
                    CompilationMode.VLLM_COMPILE if compiled else CompilationMode.NONE
                ),
                cudagraph_mode=(
                    CUDAGraphMode.FULL_AND_PIECEWISE if compiled else CUDAGraphMode.NONE
                ),
            ),
        )
        resolved = _resolved(llm, compiled)
        initialization_ready_at = _now()
        sampling = SamplingParams(**SAMPLING)
        records: list[dict[str, Any]] = []
        for descriptor in workload_descriptors():
            key = f"{descriptor.context_tier}/{descriptor.workload_id}"
            ids = prompt_ids[key]
            started_at = _now()
            before = time.monotonic()
            generated = llm.generate(
                [TokensPrompt(prompt_token_ids=list(ids))],
                sampling,
                use_tqdm=False,
            )
            latency = time.monotonic() - before
            ended_at = _now()
            if len(generated) != 1 or not generated[0].finished:
                raise VLLMCompileContractError("request did not complete")
            response = generated[0]
            if len(response.outputs) != 1:
                raise VLLMCompileContractError("request returned multiple completions")
            completion = response.outputs[0]
            output_ids = list(completion.token_ids)
            if completion.finish_reason not in {"stop", "length"} or not output_ids:
                raise VLLMCompileContractError("request terminal state is invalid")
            if len(completion.text.encode()) > MAX_OUTPUT_BYTES:
                raise VLLMCompileContractError("decoded output exceeds bound")
            record = {
                **descriptor.to_dict(),
                "started_at": started_at,
                "ended_at": ended_at,
                "latency_seconds": latency,
                "ttft_seconds": _metric_ttft(response.metrics),
                "input_token_count": len(ids),
                "input_token_ids_sha256": _sha256_json(ids),
                "output_token_count": len(output_ids),
                "output_token_ids": output_ids,
                "decoded_output": completion.text,
                "finish_reason": completion.finish_reason,
                "output_tokens_per_second": len(output_ids) / latency,
                "terminal": True,
            }
            records.append(record)
            _atomic_json(
                output.with_name(f".{output.stem}-progress.json"),
                {"schema_version": "1", "mode": mode, "requests": records},
            )
    finally:
        sampler.stop()
    compilation_time = getattr(
        llm.llm_engine.vllm_config.compilation_config, "compilation_time", None
    )
    compilation_seconds = (
        float(compilation_time)
        if compiled
        and isinstance(compilation_time, (int, float))
        and compilation_time > 0
        else None
    )
    terminal = _seal(
        {
            "schema_version": "1",
            "provider": "cloudrift",
            "cell_id": f"rtx4090-{mode}",
            "mode": mode,
            "host_invocation_started_at": host_invocation_started_at,
            "process_started_at": process_started_at,
            "initialization_started_at": initialization_started_at,
            "initialization_ready_at": initialization_ready_at,
            "ended_at": _now(),
            "hardware": hardware,
            "runtime": runtime,
            "runtime_image": {
                "base_reference": BASE_IMAGE_REFERENCE,
                "derived_image_id": DERIVED_IMAGE_ID,
                "overlay": ["typing_extensions==4.15.0"],
            },
            "resolved_execution_config": resolved,
            "compilation_seconds": compilation_seconds,
            "compilation_seconds_unobservable_reason": (
                None
                if compilation_seconds is not None
                else (
                    "vllm_compilation_time_not_exposed_or_nonpositive"
                    if compiled
                    else "not_applicable_eager_mode"
                )
            ),
            "cuda_graph_seconds": None,
            "cuda_graph_seconds_unobservable_reason": (
                "stable_component_timing_not_exposed"
                if compiled
                else "not_applicable_eager_mode"
            ),
            "peak_gpu_memory_mib": sampler.peak_mib,
            "requests": records,
            "terminal": len(records) == 12,
        },
        "cell_sha256",
    )
    if not terminal["terminal"]:
        raise VLLMCompileContractError("cell did not complete 12 requests")
    _atomic_json(output, terminal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("tokenizer-canary", "stage", "eager", "compiled")
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "tokenizer-canary":
        if args.output is None:
            raise SystemExit("--output is required")
        tokenizer_canary(args.model_path, args.output)
    elif args.command == "stage":
        if args.state_path is None:
            raise SystemExit("--state-path is required")
        stage(args.model_path, args.state_path)
    else:
        if args.output is None or args.state_path is None:
            raise SystemExit("--output and --state-path are required")
        run_cell(args.command, args.model_path, args.state_path, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
