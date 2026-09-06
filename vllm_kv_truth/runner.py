"""Pinned in-container runner surface for ``qwen3-8b-vllm-kv-truth-v1``.

This module defines the *exact*, immutable configuration the CloudRift vLLM
KV-cache truth audit plan authorizes: one GPU, ``tensor_parallel_size=1``,
``data_parallel_size=1``, ``max_num_seqs=1``, eager execution, fixed sampling,
and the exact image/runtime/model identity published in PR #61. It also
implements the smallest complete executable protocol runner: strictly
sequential per-request execution against an injected engine/event-capture
pair, the fixed nested "B" (cache-enabled) lane, the "A" (cache-disabled
control) lane, the fixed eviction lane, and a deterministic evidence receipt
writer -- all expressed against small ``Protocol`` interfaces so the full
orchestration is exercised by tests with fakes, independent of whether
``vllm``/``torch``/a GPU are actually present.

Nothing at module import time touches ``vllm``, ``torch``, ``zmq``, or the
GPU: every function that ultimately drives a real engine or a real ZMQ
subscriber imports those lazily, so this module (and every pure/injectable
helper in it) is fully importable and unit-testable on a machine with none of
them installed, exactly like the sibling ``cloudrift_runner``/``vllm_compile``
modules in this package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from llmtracefx.optimizer.collectors._shared import atomic_write_text
from llmtracefx.optimizer.lab.qwen3_8b.cloudrift_runner import (
    BASE_IMAGE_REFERENCE,
    EXPECTED_DRIVER,
    EXPECTED_GPU_NAME,
    EXPECTED_MEMORY_MIB,
    RUNTIME_PINS,
)
from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    VLLM_SOURCE_COMMIT,
)
from llmtracefx.optimizer.schema import SchemaValidationError
from vllm_kv_truth.vllm_live import (
    END_OF_REPLAY_SEQUENCE,
    REQUIRED_ENDPOINT_ROLE,
    REQUIRED_REPLAY_ENDPOINT_ROLE,
    IdentityReceipt,
    KVEventsPublisherAttestation,
    LiveBlockStored,
    LiveKVEventBatch,
    ResolvedVLLMConfig,
    RuntimeAttestation,
    SourceFileDigest,
    canonical_json,
    decode_sequence_frame,
    parse_identity_receipt,
    parse_live_kv_event_batch,
    parse_runtime_attestation,
    required_source_file_digests,
    required_source_file_paths,
    sha256_digest,
)

from .workload import (
    BLOCK_SIZE,
    EVICTION_LANE_REQUESTS,
    NESTED_PROBES,
    NUM_GPU_BLOCKS_OVERRIDE,
    PREFIX_MATCH_UNIT,
    KVTruthProbe,
)

#: The committed, real, per-file Qwen3-8B model hash manifest this module
#: reuses for genuine local model-inventory verification (see
#: :func:`verify_model_inventory`). It belongs to the sibling MLX-conversion
#: subsystem, not this protocol, but records the exact same official
#: ``Qwen/Qwen3-8B`` revision/15-file/16,397,461,266-byte inventory this
#: protocol also pins, so reusing it avoids maintaining a second, redundant
#: copy of the same real upstream facts.
_QWEN3_8B_MODEL_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "llmtracefx"
    / "optimizer"
    / "lab"
    / "qwen3_8b"
    / "data"
    / "qwen3-8b-conversion-manifest-v1.json"
)

#: Environment variables the deploy orchestrator (``vllm_kv_truth_lifecycle``)
#: already injects into every lane container, carrying facts it has already
#: independently verified before ever launching this container (the staged
#: source archive's real ``git rev-parse HEAD``, this run's authorized
#: nonce, and the real Docker image ID the orchestrator itself built and
#: inspected). This runner reads them rather than trusting a hardcoded
#: constant or an arbitrary caller-supplied string, but does not re-derive
#: them from inside the container: there is no way for a built container
#: image to independently know "which commit was I built from" or "what is
#: my own image ID" without a baked-in marker or an externally-injected
#: fact, so binding these happens at the strictly earlier, real-git/real-
#: Docker staging stage the orchestrator already performs.
EXPECTED_REPOSITORY_COMMIT_ENV = "EXPECTED_REPOSITORY_COMMIT"
EXPECTED_EXPERIMENT_NONCE_ENV = "EXPECTED_EXPERIMENT_NONCE"
RUNNER_COMMIT_MARKER_PATH = Path("/opt/llmtracefx/RUNNER_COMMIT")

#: The derived runtime image is built, by the deploy orchestrator, from
#: whichever repository HEAD it just staged and verified -- a future
#: merged/tested commit this module cannot know in advance -- so unlike the
#: immutable upstream ``BASE_IMAGE_REFERENCE`` digest pin, there is no fixed
#: historical constant this protocol can compare a derived image ID against.
#: The orchestrator instead builds the derived overlay first, inspects the
#: real Docker image ID Docker assigned it, and injects that value into
#: this container's ``EXPECTED_IMAGE_ID`` environment variable only *after*
#: the build (see ``vllm_kv_truth_lifecycle.build_docker_run_argv``); this
#: runner reads and validates that value rather than trusting any
#: preregistered/historical image ID.
EXPECTED_IMAGE_ID_ENV = "EXPECTED_IMAGE_ID"
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

PROTOCOL_ID = "qwen3-8b-vllm-kv-truth-v1"

#: The plan-provenance repository commit this protocol was originally
#: approved against. This is historical documentation only -- it is never
#: compared against a live attestation, because it can never bind a future
#: post-merge execution HEAD. Every live run instead binds
#: ``repository_commit`` from the ``EXPECTED_REPOSITORY_COMMIT`` environment
#: variable the deploy orchestrator injects, which carries the run's actual
#: staged source HEAD after the orchestrator's own real
#: ``git rev-parse``/archive-hash verification (see
#: :func:`protocol_reasons_for_attestation`); this constant only remains a
#: same-value fallback for local/dev use when that variable is unset.
PLAN_APPROVED_REPOSITORY_COMMIT = "596d1a97ea2fab4dfc20e9262134925986257c09"

# -- Fixed runtime protocol (plan: "Fixed runtime protocol and VRAM fit") ----
TENSOR_PARALLEL_SIZE = 1
DATA_PARALLEL_SIZE = 1
MAX_NUM_SEQS = 1
MAX_MODEL_LEN = 1024
GPU_MEMORY_UTILIZATION = 0.90
CACHE_DTYPE = "bfloat16"
PREFIX_CACHING_HASH_ALGO = "sha256_cbor"
KV_EVENTS_USE_INT_BLOCK_HASHES = "0"
ENFORCE_EAGER = True
ENABLE_PREFIX_CACHING = True
ENABLE_KV_CACHE_EVENTS = True
SPECULATIVE_CONFIG_ENABLED = False
LORA_ENABLED = False
MULTIMODAL_ENABLED = False

# -- Fixed, deterministic sampling (plan: "16-token math") -------------------
SAMPLING_SEED = 20260906
MAX_TOKENS = 2
TEMPERATURE = 0.0
TOP_P = 1.0
SAMPLING_PARAMS: dict[str, Any] = {
    "max_tokens": MAX_TOKENS,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "seed": SAMPLING_SEED,
}

MINIMUM_FREE_VRAM_MIB_RESERVE = 1024

# -- Fixed loopback KV-event publisher endpoints -----------------------------
#: Container-internal loopback only; never forwarded, tunneled, or exposed to
#: the host over SSH. The pub and replay sockets need distinct ports because
#: they are different ZMQ socket types (PUB vs ROUTER).
KV_EVENTS_LOOPBACK_PORT = 57003
KV_EVENTS_REPLAY_PORT = 57004
KV_EVENTS_TOPIC = PROTOCOL_ID

# -- Fixed environment controls (plan: hash determinism + no network) -------
#: Every value the pinned runner must set in its own process environment
#: before constructing the engine. ``VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0``
#: keeps raw 256-bit block hashes instead of vLLM's default 64-bit truncated
#: integers; ``PYTHONHASHSEED`` must be fixed for any hashing that depends on
#: Python's built-in ``hash()``; ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``
#: forbid any network fetch once the model has been verified locally.
REQUIRED_ENVIRONMENT_VARIABLES: dict[str, str] = {
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": KV_EVENTS_USE_INT_BLOCK_HASHES,
    "PYTHONHASHSEED": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


class KVTruthProtocolError(ValueError):
    """Raised when an attestation, engine result, or receipt violates the
    exact fixed ``qwen3-8b-vllm-kv-truth-v1`` protocol.

    Every raise site names the exact mismatched field; there is no fallback
    or partial acceptance path.
    """


def environment_reasons(env: Mapping[str, str]) -> tuple[str, ...]:
    """Return every required environment variable that ``env`` gets wrong."""

    reasons: list[str] = []
    for name, required_value in REQUIRED_ENVIRONMENT_VARIABLES.items():
        if env.get(name) != required_value:
            reasons.append(f"environment_variable_{name.lower()}_mismatch")
    return tuple(reasons)


def assert_offline_environment(env: Mapping[str, str]) -> None:
    """Refuse (raise) unless every required environment variable is set exactly."""

    reasons = environment_reasons(env)
    if reasons:
        raise KVTruthProtocolError(
            "process environment does not satisfy the qwen3-8b-vllm-kv-truth-v1 "
            "offline/determinism contract: " + ", ".join(reasons)
        )


def expected_repository_commit(environ: Mapping[str, str] | None = None) -> str:
    """Return the exact repository commit a live attestation must bind to.

    Prefers the deploy orchestrator's ``EXPECTED_REPOSITORY_COMMIT``
    environment variable -- the run's actual staged source HEAD, already
    independently verified by the orchestrator's own real
    ``git rev-parse``/archive-hash check before this container was ever
    launched -- falling back to :data:`PLAN_APPROVED_REPOSITORY_COMMIT` only
    when that variable is unset, so local/dev tests and tooling that never
    go through the orchestrator keep working unchanged.
    """

    import os

    env = os.environ if environ is None else environ
    return env.get(EXPECTED_REPOSITORY_COMMIT_ENV) or PLAN_APPROVED_REPOSITORY_COMMIT


def expected_image_id(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the exact derived runtime image ID a live attestation must bind to.

    Unlike :func:`expected_repository_commit`, there is no historical
    constant this can ever fall back to: the derived overlay image is built
    from whichever repository HEAD the deploy orchestrator just staged and
    verified -- a future post-merge commit this module cannot know in
    advance -- so its resulting image ID cannot be preregistered without
    silently validating a stale build. The orchestrator therefore builds
    the derived image first, inspects the real Docker image ID it produced,
    and injects that value into this container's ``EXPECTED_IMAGE_ID``
    environment variable only after the build. This function reads and
    strictly validates that value's shape (``sha256:`` followed by exactly
    64 lowercase hex characters) and returns ``None`` -- never raises -- if
    it is missing or malformed, so callers that need to enumerate every
    protocol violation (:func:`protocol_reasons_for_attestation`) can turn
    that into a normal, deterministic reason rather than an exception.
    """

    import os

    env = os.environ if environ is None else environ
    raw = env.get(EXPECTED_IMAGE_ID_ENV)
    if raw is None or not _IMAGE_ID_PATTERN.fullmatch(raw):
        return None
    return raw


def protocol_reasons_for_attestation(
    attestation: RuntimeAttestation,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every way ``attestation`` fails this exact protocol's pins.

    Complements (does not replace)
    ``vllm_live.parse_runtime_attestation``'s generic vLLM-commit/seal
    checks: this only checks the additional, protocol-specific constants --
    the exact image, model, runtime pins, GPU identity, and resolved engine
    configuration published in PR #61 and this plan. Every applicable check
    runs and appends at most one reason, in a fixed order, so the result is
    a deterministic, complete refusal list rather than the first mismatch.
    ``repository_commit`` is checked against :func:`expected_repository_commit`
    (the orchestrator-verified staged HEAD), never a fixed historical
    constant, so this check can bind any future post-merge execution HEAD.
    Likewise ``image_id`` is checked against :func:`expected_image_id` (the
    orchestrator-verified, freshly built derived overlay image ID), never a
    fixed historical constant, so this check can bind any future
    post-merge-built derived image.
    """

    if not isinstance(attestation, RuntimeAttestation):
        raise KVTruthProtocolError(
            "attestation must be a validated RuntimeAttestation, not "
            f"{type(attestation).__name__}"
        )

    reasons: list[str] = []
    if attestation.protocol_id != PROTOCOL_ID:
        reasons.append("protocol_id_mismatch")
    if attestation.image_repository_digest != BASE_IMAGE_REFERENCE:
        reasons.append("base_image_reference_mismatch")
    live_expected_image_id = expected_image_id(environ)
    if live_expected_image_id is None:
        reasons.append("expected_image_id_env_missing_or_malformed")
    elif attestation.image_id != live_expected_image_id:
        reasons.append("derived_image_id_mismatch")
    if attestation.python_version != RUNTIME_PINS["python_version"]:
        reasons.append("python_version_mismatch")
    if attestation.torch_version != RUNTIME_PINS["torch_version"]:
        reasons.append("torch_version_mismatch")
    if attestation.cuda_runtime_version != RUNTIME_PINS["cuda_version"]:
        reasons.append("cuda_runtime_version_mismatch")
    if attestation.transformers_version != RUNTIME_PINS["transformers_version"]:
        reasons.append("transformers_version_mismatch")
    if (
        attestation.typing_extensions_version
        != RUNTIME_PINS["typing_extensions_version"]
    ):
        reasons.append("typing_extensions_version_mismatch")
    if attestation.vllm_commit != VLLM_SOURCE_COMMIT:
        reasons.append("vllm_commit_mismatch")
    if attestation.gpu_name != EXPECTED_GPU_NAME:
        reasons.append("gpu_name_mismatch")
    if attestation.gpu_memory_mib != EXPECTED_MEMORY_MIB:
        reasons.append("gpu_memory_mismatch")
    if attestation.cuda_driver_version != EXPECTED_DRIVER:
        reasons.append("cuda_driver_version_mismatch")
    if attestation.model_id != MODEL_ID:
        reasons.append("model_id_mismatch")
    if attestation.model_revision != MODEL_REVISION:
        reasons.append("model_revision_mismatch")
    if attestation.repository_commit != expected_repository_commit(environ):
        reasons.append("repository_commit_mismatch")

    config = attestation.resolved_config
    if config.max_model_len != MAX_MODEL_LEN:
        reasons.append("max_model_len_mismatch")
    if config.max_num_seqs != MAX_NUM_SEQS:
        reasons.append("max_num_seqs_mismatch")
    if config.tensor_parallel_size != TENSOR_PARALLEL_SIZE:
        reasons.append("tensor_parallel_size_mismatch")
    if config.data_parallel_size != DATA_PARALLEL_SIZE:
        reasons.append("data_parallel_size_mismatch")
    if config.block_size != BLOCK_SIZE:
        reasons.append("block_size_mismatch")
    if config.prefix_match_unit != PREFIX_MATCH_UNIT:
        reasons.append("prefix_match_unit_mismatch")
    if config.num_gpu_blocks_override != NUM_GPU_BLOCKS_OVERRIDE:
        reasons.append("num_gpu_blocks_override_mismatch")
    if config.gpu_memory_utilization != GPU_MEMORY_UTILIZATION:
        reasons.append("gpu_memory_utilization_mismatch")
    if config.cache_dtype != CACHE_DTYPE:
        reasons.append("cache_dtype_mismatch")
    if config.prefix_caching_hash_algo != PREFIX_CACHING_HASH_ALGO:
        reasons.append("prefix_caching_hash_algo_mismatch")
    if config.kv_events_use_int_block_hashes != KV_EVENTS_USE_INT_BLOCK_HASHES:
        reasons.append("kv_events_use_int_block_hashes_mismatch")
    if not config.enforce_eager:
        reasons.append("enforce_eager_disabled")
    if config.speculative_config_enabled != SPECULATIVE_CONFIG_ENABLED:
        reasons.append("speculative_config_enabled_mismatch")
    if config.lora_enabled != LORA_ENABLED:
        reasons.append("lora_enabled_mismatch")
    if config.multimodal_enabled != MULTIMODAL_ENABLED:
        reasons.append("multimodal_enabled_mismatch")

    kv_events = attestation.kv_events_config
    if kv_events.topic != KV_EVENTS_TOPIC:
        reasons.append("kv_events_topic_mismatch")
    return tuple(reasons)


def assert_protocol_attestation(
    attestation: RuntimeAttestation, *, environ: Mapping[str, str] | None = None
) -> None:
    """Refuse (raise) unless ``attestation`` matches every fixed protocol pin."""

    reasons = protocol_reasons_for_attestation(attestation, environ=environ)
    if reasons:
        raise KVTruthProtocolError(
            "attestation does not match the qwen3-8b-vllm-kv-truth-v1 protocol: "
            + ", ".join(reasons)
        )


def kv_events_config_kwargs() -> dict[str, Any]:
    """Return the exact ``KVEventsConfig`` field values for this protocol.

    A plain dict of the real ``vllm.config.kv_events.KVEventsConfig`` field
    names, so this stays importable/testable without ``vllm`` installed;
    :func:`build_llm` converts it into a real ``KVEventsConfig`` instance
    immediately before constructing the engine.
    """

    return {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://127.0.0.1:{KV_EVENTS_LOOPBACK_PORT}",
        "replay_endpoint": f"tcp://127.0.0.1:{KV_EVENTS_REPLAY_PORT}",
        "buffer_steps": 10_000,
        "hwm": 100_000,
        "max_queue_size": 100_000,
        "topic": KV_EVENTS_TOPIC,
    }


def build_engine_kwargs(
    attestation: RuntimeAttestation,
    *,
    model_path: str,
    cache_enabled: bool = True,
) -> dict[str, Any]:
    """Return the exact ``vllm.LLMEngine`` constructor kwargs for this protocol.

    Refuses (raises :class:`KVTruthProtocolError`) unless ``attestation``
    matches every fixed pin above. ``model_path`` must be the verified local
    filesystem path the 15-file inventory check already ran against -- never
    the remote ``model_id`` -- so the engine never touches the network even
    if offline enforcement were somehow bypassed. There is no parameter to
    override any identity or config value with a caller-supplied string:
    every returned value is either one of this module's own fixed constants,
    the caller-supplied verified local path, or a value copied verbatim from
    the attestation's already-verified resolved configuration.

    ``cache_enabled`` is the one deliberate per-lane deviation from the
    attested "this environment supports full caching" identity: the
    attestation itself always describes a cache/KV-eventing-capable
    environment (``vllm_live.parse_runtime_attestation`` requires
    ``enable_prefix_caching``/``enable_kv_cache_events`` to be true
    unconditionally), but the plan's "A" lane is a cache-*disabled*
    output-identity control, so ``run_a_lane`` must actually get a
    cache-disabled engine, not merely an unread config flag. When ``False``,
    this disables prefix caching and omits KV-event publishing entirely
    (a cache-disabled engine cannot publish cache events).
    """

    assert_protocol_attestation(attestation)
    if not isinstance(model_path, str) or not model_path.strip():
        raise KVTruthProtocolError("model_path must be a non-empty local path")
    config = attestation.resolved_config
    return {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": False,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "block_size": BLOCK_SIZE,
        "prefix_match_unit": PREFIX_MATCH_UNIT,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS_OVERRIDE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "enforce_eager": ENFORCE_EAGER,
        "enable_prefix_caching": bool(cache_enabled),
        "prefix_caching_hash_algo": PREFIX_CACHING_HASH_ALGO,
        "dtype": config.cache_dtype,
        "kv_cache_dtype": config.cache_dtype,
        "seed": SAMPLING_SEED,
        # RequestOutput.metrics is populated only when vLLM request stats are
        # enabled. This protocol records those version-bound timing fields.
        "disable_log_stats": False,
        "kv_events_config": kv_events_config_kwargs() if cache_enabled else None,
    }


# ---------------------------------------------------------------------------
# Real, in-container-only system-fact collection
# ---------------------------------------------------------------------------
#
# Every function below performs actual local measurement -- file hashing,
# ``importlib.metadata`` package introspection, an ``nvidia-smi`` subprocess
# call -- instead of trusting a caller-supplied string. Nothing here imports
# ``vllm``/``torch``/``zmq`` at module scope; ``torch`` is imported lazily
# only inside :func:`collect_runtime_versions` (needed for
# ``torch.version.cuda``, which ``importlib.metadata`` cannot report),
# matching this module's existing lazy-import discipline for ``vllm``.


def _hash_file(path: Path) -> str:
    """Return the raw (unprefixed) SHA-256 hex digest of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_prefixed_file(path: Path) -> str:
    return "sha256:" + _hash_file(path)


def _run_command(argv: Sequence[str]) -> str:
    return subprocess.run(
        list(argv), check=True, capture_output=True, text=True, shell=False
    ).stdout


def collect_runtime_versions() -> dict[str, str]:
    """Return the actually-installed python/vllm/torch/cuda/... versions.

    Every value is read from the real running interpreter and its installed
    distributions (``importlib.metadata``) or from the imported ``torch``
    module's own ``version.cuda`` attribute -- never copied from a
    caller-supplied or hardcoded string.
    """

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KVTruthProtocolError(
            "torch is not importable in this environment; refusing to "
            "collect real runtime identity"
        ) from exc
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not cuda_version:
        raise KVTruthProtocolError(
            "torch.version.cuda is not set; refusing to attest a CPU-only "
            "torch build for a GPU-only protocol"
        )
    try:
        vllm_version = importlib.metadata.version("vllm")
        torch_version = importlib.metadata.version("torch")
        transformers_version = importlib.metadata.version("transformers")
        typing_extensions_version = importlib.metadata.version("typing_extensions")
    except importlib.metadata.PackageNotFoundError as exc:
        raise KVTruthProtocolError(
            f"a required package is not installed: {exc}"
        ) from exc
    return {
        "python_version": ".".join(str(part) for part in sys.version_info[:2]),
        "vllm_version": vllm_version,
        "torch_version": torch_version,
        "cuda_runtime_version": cuda_version,
        "transformers_version": transformers_version,
        "typing_extensions_version": typing_extensions_version,
    }


def collect_gpu_facts(
    *, run_command: Callable[[Sequence[str]], str] = _run_command
) -> dict[str, Any]:
    """Return real, locally-queried GPU facts via ``nvidia-smi``.

    ``gpu_uuid_commitment`` is a SHA-256 commitment of the raw GPU UUID,
    never the raw UUID itself, matching
    ``IdentityReceipt.gpu_uuid_commitment``'s bare-hex commitment contract
    (the plan requires the GPU identity to be committed to, not published in
    the clear). Refuses unless exactly one GPU is visible, matching this
    protocol's fixed one-GPU pin.
    """

    output = run_command(
        (
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,uuid,compute_cap",
            "--format=csv,noheader,nounits",
        )
    )
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        raise KVTruthProtocolError(
            f"nvidia-smi reported {len(lines)} GPUs; this protocol requires "
            "exactly one visible GPU"
        )
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 5:
        raise KVTruthProtocolError("nvidia-smi did not report a complete GPU identity")
    name, driver, memory_total, gpu_uuid, compute_cap = fields
    if not gpu_uuid:
        raise KVTruthProtocolError("nvidia-smi reported an empty GPU UUID")
    return {
        "gpu_name": name,
        "cuda_driver_version": driver,
        "gpu_memory_mib": int(memory_total),
        "gpu_compute_capability": compute_cap,
        "gpu_uuid_commitment": hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest(),
    }


def collect_source_file_digests(
    vllm_package_root: Path,
) -> tuple[SourceFileDigest, ...]:
    """Hash the pinned vLLM 0.28.0 critical source files from the actually
    installed package tree and independently compare each recomputed digest
    against the committed manifest's real expected SHA-256 value
    (:func:`~vllm_kv_truth.vllm_live.required_source_file_digests`,
    fetched once from the pinned upstream commit -- source fetch only, no
    provider/model/image access). ``matches_manifest`` therefore reflects a
    genuine byte-content comparison against real expected values, not merely
    "the path was present and hashed successfully".
    """

    expected_digests = required_source_file_digests()
    digests: list[SourceFileDigest] = []
    for relative_path in sorted(required_source_file_paths()):
        path = vllm_package_root / relative_path
        if path.is_symlink() or not path.is_file():
            raise KVTruthProtocolError(
                "required vLLM source file is missing from the installed "
                f"package tree: {relative_path}"
            )
        actual_sha256 = _hash_file(path)
        expected_sha256 = expected_digests.get(relative_path)
        digests.append(
            SourceFileDigest(
                path=relative_path,
                sha256=actual_sha256,
                matches_manifest=(
                    expected_sha256 is not None and actual_sha256 == expected_sha256
                ),
            )
        )
    return tuple(digests)


def collect_installed_distributions_digest() -> str:
    """Return a stable digest of every installed distribution name/version."""

    distributions = sorted(
        (dist.name, dist.version)
        for dist in importlib.metadata.distributions()
        if dist.name
    )
    if not distributions:
        raise KVTruthProtocolError("no installed distributions were discovered")
    return sha256_digest(distributions)


def collect_wheel_record_digest(distribution_name: str = "vllm") -> str:
    """Return a stable digest of the installed vLLM wheel's ``RECORD`` file."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise KVTruthProtocolError(f"{distribution_name} is not installed") from exc
    record = distribution.read_text("RECORD")
    if record is None:
        raise KVTruthProtocolError(f"{distribution_name} has no RECORD file")
    return sha256_digest(sorted(line for line in record.splitlines() if line))


def collect_package_tree_digest(vllm_package_root: Path) -> str:
    """Return a stable content digest of the installed vLLM package tree."""

    if not vllm_package_root.is_dir():
        raise KVTruthProtocolError(
            f"vLLM package root does not exist: {vllm_package_root}"
        )
    files = sorted(
        path
        for path in vllm_package_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    if not files:
        raise KVTruthProtocolError("vLLM package tree is empty")
    inventory: list[dict[str, Any]] = []
    for path in files:
        if path.is_symlink():
            raise KVTruthProtocolError(
                "vLLM package tree contains a symbolic link: "
                f"{path.relative_to(vllm_package_root).as_posix()}"
            )
        inventory.append(
            {
                "path": path.relative_to(vllm_package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
        )
    return sha256_digest(inventory)


def model_path_commitment(model_path: str) -> str:
    """Return a raw (unprefixed) SHA-256 commitment of ``model_path``.

    A commitment, not the raw path itself, matching
    ``IdentityReceipt.model_path_commitment``'s bare-hex contract, since the
    plan requires the local filesystem layout never be published in the
    clear.
    """

    return hashlib.sha256(model_path.encode("utf-8")).hexdigest()


def runner_source_digest() -> str:
    """Return a ``sha256:``-prefixed digest of this runner module's own
    source file, so a tampered runner cannot silently self-attest."""

    return _sha256_prefixed_file(Path(__file__))


def verify_model_inventory(
    model_path: Path,
) -> tuple[tuple[dict[str, Any], ...], str]:
    """Verify every real, local model file against the committed 15-file,
    16,397,461,266-byte Qwen3-8B inventory manifest.

    Reuses the pre-existing ``qwen3-8b-conversion-manifest-v1.json`` (the
    sibling MLX-conversion subsystem's manifest, which happens to record
    the exact same official upstream inventory this protocol also pins)
    rather than maintaining a second, redundant copy of the same real
    facts. Returns the verified per-file records and a stable digest of
    them; raises on any missing file, extra file, size mismatch, or hash
    mismatch -- there is no partial-success return.
    """

    raw = json.loads(_QWEN3_8B_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    source = raw.get("source") if isinstance(raw, dict) else None
    if (
        not isinstance(source, dict)
        or source.get("official_id") != MODEL_ID
        or source.get("official_revision") != MODEL_REVISION
        or source.get("expected_source_bytes") != EXPECTED_MODEL_BYTES
    ):
        raise KVTruthProtocolError("qwen3-8b model manifest identity is invalid")
    files = source.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_MODEL_FILE_COUNT:
        raise KVTruthProtocolError(
            "qwen3-8b model manifest must list exactly "
            f"{EXPECTED_MODEL_FILE_COUNT} files"
        )
    expected = {item["path"]: item for item in files}
    observed = {
        path.relative_to(model_path).as_posix()
        for path in model_path.rglob("*")
        if path.is_file() and ".cache/huggingface/" not in path.as_posix()
    }
    if observed != set(expected):
        raise KVTruthProtocolError(
            "local model inventory does not match the committed manifest exactly"
        )
    verified: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = model_path / relative
        item = expected[relative]
        if path.is_symlink() or path.stat().st_size != item["size_bytes"]:
            raise KVTruthProtocolError(f"model file size mismatch: {relative}")
        digest = _hash_file(path)
        if digest != item["sha256"]:
            raise KVTruthProtocolError(f"model file hash mismatch: {relative}")
        verified.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    total_bytes = sum(item["size_bytes"] for item in verified)
    if total_bytes != EXPECTED_MODEL_BYTES:
        raise KVTruthProtocolError(
            "verified model byte total does not match the committed manifest"
        )
    return tuple(verified), sha256_digest(verified)


def _tokenizer_artifact_digest(verified_files: Sequence[Mapping[str, Any]]) -> str:
    tokenizer_files = sorted(
        (item for item in verified_files if "tokenizer" in item["path"].lower()),
        key=lambda item: item["path"],
    )
    if not tokenizer_files:
        raise KVTruthProtocolError(
            "verified model inventory contains no tokenizer artifacts"
        )
    return sha256_digest(tokenizer_files)


def _discover_vllm_package_root() -> Path:
    try:
        import vllm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KVTruthProtocolError(
            "vllm is not importable in this environment; refusing to "
            "generate a real identity receipt"
        ) from exc
    package_file = getattr(vllm, "__file__", None)
    if not package_file:
        raise KVTruthProtocolError("the installed vllm package has no __file__")
    return Path(package_file).resolve().parent


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def generate_identity_receipt(
    *,
    model_path: str,
    vllm_package_root: Path,
    environ: Mapping[str, str],
    run_command: Callable[[Sequence[str]], str] = _run_command,
    now_fn: Callable[[], str] = _iso_now,
) -> IdentityReceipt:
    """Generate a real, freshly-measured, self-verifying pre-init identity receipt.

    Every field is either directly measured from this process's own live
    environment (package versions, GPU facts via ``nvidia-smi``, source-file
    and model-file hashes) or propagated from an already-independently
    verified upstream fact: ``repository_commit``/``experiment_nonce`` come
    from the deploy orchestrator's ``EXPECTED_REPOSITORY_COMMIT``/
    ``EXPECTED_EXPERIMENT_NONCE`` environment variables, which the
    orchestrator only sets after its own real, remote
    ``git rev-parse``/archive-hash verification of the staged source -- a
    container image cannot independently know "which commit was I built
    from" without a baked-in marker, so binding that specific fact happens
    at the strictly earlier staging stage, not by fabricating a local
    measurement for it. Every other field genuinely comes from this
    process's own live measurement, never from a caller-supplied string
    accepted at face value. The result is round-tripped through
    :func:`parse_identity_receipt` before being returned, so this function
    can never hand back a receipt its own strict parser would reject.
    """

    repository_commit = environ.get(EXPECTED_REPOSITORY_COMMIT_ENV)
    if not repository_commit:
        raise KVTruthProtocolError(
            f"{EXPECTED_REPOSITORY_COMMIT_ENV} is not set; the orchestrator "
            "must inject the verified staged repository HEAD before this "
            "runner can generate an identity receipt"
        )
    experiment_nonce = environ.get(EXPECTED_EXPERIMENT_NONCE_ENV)
    if not experiment_nonce:
        raise KVTruthProtocolError(
            f"{EXPECTED_EXPERIMENT_NONCE_ENV} is not set; the orchestrator "
            "must inject this run's authorized nonce"
        )
    derived_image_id = expected_image_id(environ)
    if derived_image_id is None:
        raise KVTruthProtocolError(
            f"{EXPECTED_IMAGE_ID_ENV} is not set to a well-formed "
            "'sha256:<64 lowercase hex characters>' value; the orchestrator "
            "must build and inspect the derived overlay image, then inject "
            "its real, freshly-produced image id before this runner can "
            "generate an identity receipt -- there is no historical "
            "constant this could fall back to, because the derived image "
            "is built from whichever repository HEAD was actually staged"
        )
    try:
        baked_repository_commit = RUNNER_COMMIT_MARKER_PATH.read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise KVTruthProtocolError(
            "the derived image does not contain its baked RUNNER_COMMIT marker"
        ) from exc
    if (
        re.fullmatch(r"[0-9a-f]{40}", baked_repository_commit) is None
        or baked_repository_commit != repository_commit
    ):
        raise KVTruthProtocolError(
            "the baked RUNNER_COMMIT marker does not match the orchestrator's "
            "verified repository commit"
        )

    runtime_versions = collect_runtime_versions()
    gpu_facts = collect_gpu_facts(run_command=run_command)
    source_file_digests = collect_source_file_digests(vllm_package_root)
    model_root = Path(model_path)
    verified_files, model_inventory_digest = verify_model_inventory(model_root)

    unsealed: dict[str, Any] = {
        "schema_version": "1",
        "protocol_id": PROTOCOL_ID,
        "generated_at": now_fn(),
        "repository_commit": repository_commit,
        "image_repository_digest": BASE_IMAGE_REFERENCE,
        "image_id": derived_image_id,
        "vllm_version": runtime_versions["vllm_version"],
        "vllm_commit": VLLM_SOURCE_COMMIT,
        "python_version": runtime_versions["python_version"],
        "torch_version": runtime_versions["torch_version"],
        "cuda_runtime_version": runtime_versions["cuda_runtime_version"],
        "transformers_version": runtime_versions["transformers_version"],
        "typing_extensions_version": runtime_versions["typing_extensions_version"],
        "cuda_driver_version": gpu_facts["cuda_driver_version"],
        "gpu_name": gpu_facts["gpu_name"],
        "gpu_memory_mib": gpu_facts["gpu_memory_mib"],
        "gpu_compute_capability": gpu_facts["gpu_compute_capability"],
        "gpu_uuid_commitment": gpu_facts["gpu_uuid_commitment"],
        "experiment_nonce": experiment_nonce,
        "installed_distributions_digest": collect_installed_distributions_digest(),
        "wheel_record_digest": collect_wheel_record_digest(),
        "package_tree_digest": collect_package_tree_digest(vllm_package_root),
        "source_file_digests": [item.to_dict() for item in source_file_digests],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_artifact_digest": _tokenizer_artifact_digest(verified_files),
        "model_inventory_digest": model_inventory_digest,
        "model_path_commitment": model_path_commitment(model_path),
        "runner_source_digest": runner_source_digest(),
    }
    seal = hashlib.sha256(canonical_json(unsealed).encode("utf-8")).hexdigest()
    payload = dict(unsealed)
    payload["seal"] = seal
    try:
        return parse_identity_receipt(payload)
    except SchemaValidationError as exc:
        raise KVTruthProtocolError(
            f"generated identity receipt failed self-verification: {exc}"
        ) from exc


def generate_runtime_attestation(
    identity: IdentityReceipt, *, now_fn: Callable[[], str] = _iso_now
) -> RuntimeAttestation:
    """Build the pre-init runtime attestation this protocol requires before
    constructing the engine, from this module's own fixed, exact protocol
    pins -- never a caller-supplied resolved config or event boundary.

    ``resolved_config``/``kv_events_config`` describe the exact
    configuration this runner is *about to request*: real, fixed constants
    already reviewed against the plan, not a measurement (the KV-event
    capture boundary cannot exist yet, since the engine has not been
    constructed and the subscriber has not started; both are left null,
    which :func:`parse_runtime_attestation`'s validation of
    ``kv_events_config`` explicitly allows). Genuine post-init confirmation
    that the constructed engine actually resolved to these exact values
    happens separately, inside :func:`build_llm`, by reading the real
    ``vllm.LLMEngine``'s own resolved configuration back and comparing it
    field-by-field against this same protocol's pins; that check runs
    after this attestation has already gated engine construction, and
    refuses (raises) on any drift instead of merely trusting the request.
    """

    resolved_config = ResolvedVLLMConfig(
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        block_size=BLOCK_SIZE,
        prefix_match_unit=PREFIX_MATCH_UNIT,
        num_gpu_blocks_override=NUM_GPU_BLOCKS_OVERRIDE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        cache_dtype=CACHE_DTYPE,
        prefix_caching_hash_algo=PREFIX_CACHING_HASH_ALGO,
        kv_events_use_int_block_hashes=KV_EVENTS_USE_INT_BLOCK_HASHES,
        pythonhashseed=REQUIRED_ENVIRONMENT_VARIABLES["PYTHONHASHSEED"],
        enable_prefix_caching=True,
        enable_kv_cache_events=True,
        enforce_eager=ENFORCE_EAGER,
        speculative_config_enabled=SPECULATIVE_CONFIG_ENABLED,
        lora_enabled=LORA_ENABLED,
        multimodal_enabled=MULTIMODAL_ENABLED,
        cache_salt_present=False,
        cache_salt=None,
    )
    kv_events_config = KVEventsPublisherAttestation(
        topic=KV_EVENTS_TOPIC,
        endpoint_role=REQUIRED_ENDPOINT_ROLE,
        replay_endpoint_role=REQUIRED_REPLAY_ENDPOINT_ROLE,
        buffer_steps=10_000,
        hwm=100_000,
        max_queue_size=100_000,
        data_parallel_rank=0,
        first_sequence=None,
        last_sequence=None,
        capture_start_monotonic=0.0,
        capture_end_monotonic=0.0,
    )
    attested_at = now_fn()
    unsealed = {
        "identity_seal": identity.seal,
        "resolved_config": resolved_config.to_dict(),
        "kv_events_config": kv_events_config.to_dict(),
        "attested_at": attested_at,
    }
    seal = hashlib.sha256(canonical_json(unsealed).encode("utf-8")).hexdigest()
    payload = {
        "identity": identity.to_dict(),
        "resolved_config": resolved_config.to_dict(),
        "kv_events_config": kv_events_config.to_dict(),
        "attested_at": attested_at,
        "seal": seal,
    }
    try:
        return parse_runtime_attestation(payload)
    except SchemaValidationError as exc:
        raise KVTruthProtocolError(
            f"generated runtime attestation failed self-verification: {exc}"
        ) from exc


def bind_capture_boundary(
    attestation: RuntimeAttestation,
    *,
    first_sequence: int | None,
    last_sequence: int | None,
    capture_start_monotonic: float,
    capture_end_monotonic: float,
) -> RuntimeAttestation:
    """Reseal an attestation with the measured final capture interval."""

    events = attestation.kv_events_config.to_dict()
    events.update(
        {
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "capture_start_monotonic": capture_start_monotonic,
            "capture_end_monotonic": capture_end_monotonic,
        }
    )
    attested_at = _iso_now()
    unsealed = {
        "identity_seal": attestation.identity.seal,
        "resolved_config": attestation.resolved_config.to_dict(),
        "kv_events_config": events,
        "attested_at": attested_at,
    }
    payload = {
        "identity": attestation.identity.to_dict(),
        "resolved_config": attestation.resolved_config.to_dict(),
        "kv_events_config": events,
        "attested_at": attested_at,
        "seal": hashlib.sha256(canonical_json(unsealed).encode("utf-8")).hexdigest(),
    }
    try:
        return parse_runtime_attestation(payload)
    except SchemaValidationError as exc:
        raise KVTruthProtocolError(
            f"final capture attestation failed self-verification: {exc}"
        ) from exc


def _assert_engine_matches_protocol(engine: Any, *, cache_enabled: bool) -> None:
    """Independently verify the real constructed engine's own resolved
    configuration, read back off the live ``vllm.LLMEngine`` object itself
    -- never merely assumed from the request kwargs or the pre-init
    attestation. Any missing attribute is itself treated as a genuine
    runtime validation failure (:class:`KVTruthProtocolError`), not an
    unhandled crash, since a real pinned-version engine is expected to
    expose this config surface.
    """

    try:
        vllm_config = engine.vllm_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        parallel_config = vllm_config.parallel_config
        kv_events_config = vllm_config.kv_events_config
        speculative_config = vllm_config.speculative_config
        lora_config = vllm_config.lora_config
        multimodal_config = model_config.multimodal_config
    except AttributeError as exc:
        raise KVTruthProtocolError(
            "constructed engine does not expose the expected vLLM config "
            f"surface: {exc}"
        ) from exc

    reasons: list[str] = []
    if getattr(model_config, "max_model_len", None) != MAX_MODEL_LEN:
        reasons.append("engine_max_model_len_mismatch")
    if getattr(scheduler_config, "max_num_seqs", None) != MAX_NUM_SEQS:
        reasons.append("engine_max_num_seqs_mismatch")
    if getattr(parallel_config, "tensor_parallel_size", None) != TENSOR_PARALLEL_SIZE:
        reasons.append("engine_tensor_parallel_size_mismatch")
    if getattr(parallel_config, "data_parallel_size", None) != DATA_PARALLEL_SIZE:
        reasons.append("engine_data_parallel_size_mismatch")
    if getattr(cache_config, "block_size", None) != BLOCK_SIZE:
        reasons.append("engine_block_size_mismatch")
    if getattr(cache_config, "prefix_match_unit", None) != PREFIX_MATCH_UNIT:
        reasons.append("engine_prefix_match_unit_mismatch")
    if (
        getattr(cache_config, "num_gpu_blocks_override", None)
        != NUM_GPU_BLOCKS_OVERRIDE
    ):
        reasons.append("engine_num_gpu_blocks_override_mismatch")
    if getattr(cache_config, "gpu_memory_utilization", None) != GPU_MEMORY_UTILIZATION:
        reasons.append("engine_gpu_memory_utilization_mismatch")
    if getattr(cache_config, "cache_dtype", None) != CACHE_DTYPE:
        reasons.append("engine_cache_dtype_mismatch")
    if (
        getattr(cache_config, "prefix_caching_hash_algo", None)
        != PREFIX_CACHING_HASH_ALGO
    ):
        reasons.append("engine_prefix_caching_hash_algo_mismatch")
    if bool(getattr(cache_config, "enable_prefix_caching", None)) != bool(
        cache_enabled
    ):
        reasons.append("engine_enable_prefix_caching_mismatch")
    if bool(getattr(model_config, "enforce_eager", None)) != ENFORCE_EAGER:
        reasons.append("engine_enforce_eager_mismatch")
    if speculative_config is not None:
        reasons.append("engine_speculative_config_enabled")
    if lora_config is not None:
        reasons.append("engine_lora_enabled")
    if multimodal_config is not None:
        reasons.append("engine_multimodal_enabled")
    expected_kv_events = kv_events_config_kwargs() if cache_enabled else None
    if expected_kv_events is None:
        if kv_events_config is not None:
            reasons.append("engine_kv_events_config_unexpected")
    elif kv_events_config is None:
        reasons.append("engine_kv_events_config_missing")
    else:
        for field, expected in expected_kv_events.items():
            if getattr(kv_events_config, field, None) != expected:
                reasons.append(f"engine_kv_events_{field}_mismatch")
    if reasons:
        raise KVTruthProtocolError(
            "constructed engine's real resolved configuration does not "
            "match the qwen3-8b-vllm-kv-truth-v1 protocol pins: " + ", ".join(reasons)
        )


_TIMING_FIELD_NAMES: tuple[str, ...] = (
    "queued_ts",
    "scheduled_ts",
    "first_token_ts",
    "last_token_ts",
    "first_token_latency",
)


def _coerce_optional_float(value: Any) -> float | None:
    """Coerce a raw attribute value to ``float``, refusing to fabricate a
    number from a non-numeric value (``bool`` is deliberately excluded even
    though it is an ``int`` subclass, since a timestamp/duration field is
    never legitimately boolean)."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(frozen=True)
class RequestTiming:
    """Per-request timing read from vLLM 0.28.0's exact
    ``RequestOutput.metrics`` surface (``queued_ts``, ``scheduled_ts``,
    ``first_token_ts``, ``last_token_ts``, ``first_token_latency``) plus a
    best-effort, non-fabricating read of any ``FinishedRequestStats``-shaped
    duration fields exposed alongside it.

    Every field is ``float | None``: a ``None`` value always carries an
    explicit machine-readable reason in ``null_reasons`` (e.g.
    ``"metrics_unavailable"`` when ``output.metrics`` itself is absent, or
    ``"metrics_first_token_ts_unavailable"``/``"..._malformed"`` for a
    single missing/non-numeric field) rather than silently standing in for
    "zero" or "not applicable".
    """

    queued_ts: float | None
    scheduled_ts: float | None
    first_token_ts: float | None
    last_token_ts: float | None
    first_token_latency: float | None
    finished_request_stats: Mapping[str, float] | None
    null_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "queued_ts": self.queued_ts,
            "scheduled_ts": self.scheduled_ts,
            "first_token_ts": self.first_token_ts,
            "last_token_ts": self.last_token_ts,
            "first_token_latency": self.first_token_latency,
            "finished_request_stats": (
                dict(self.finished_request_stats)
                if self.finished_request_stats is not None
                else None
            ),
            "null_reasons": list(self.null_reasons),
        }


def _extract_finished_request_stats(
    output: Any, metrics: Any
) -> Mapping[str, float] | None:
    """Best-effort, non-fabricating extraction of any
    ``FinishedRequestStats``-shaped duration fields exposed on the output
    or its metrics.

    vLLM 0.28.0's exact ``FinishedRequestStats`` attribute surface cannot
    be verified without a real installed package, so this never hardcodes
    an unverified exact attribute name: it structurally introspects
    whichever object it finds for public numeric attributes whose name
    ends in ``_time`` or ``_duration``, returning ``None`` (paired with the
    ``"finished_request_stats_unavailable"`` null reason by the caller) if
    no such object or field exists.
    """

    stats = getattr(output, "finished_request_stats", None)
    if stats is None and metrics is not None:
        stats = getattr(metrics, "finished_request_stats", None)
    if stats is None:
        return None
    collected: dict[str, float] = {}
    for name in dir(stats):
        if name.startswith("_"):
            continue
        if not (name.endswith("_time") or name.endswith("_duration")):
            continue
        value = _coerce_optional_float(getattr(stats, name, None))
        if value is not None:
            collected[name] = value
    return collected or None


def _extract_timing(output: Any) -> RequestTiming:
    metrics = getattr(output, "metrics", None)
    null_reasons: list[str] = []
    values: dict[str, float | None] = {}
    if metrics is None:
        null_reasons.append("metrics_unavailable")
        for name in _TIMING_FIELD_NAMES:
            values[name] = None
    else:
        for name in _TIMING_FIELD_NAMES:
            if not hasattr(metrics, name):
                null_reasons.append(f"metrics_{name}_unavailable")
                values[name] = None
                continue
            raw = getattr(metrics, name)
            coerced = _coerce_optional_float(raw)
            if raw is not None and coerced is None:
                null_reasons.append(f"metrics_{name}_malformed")
            values[name] = coerced
    finished_request_stats = _extract_finished_request_stats(output, metrics)
    if finished_request_stats is None:
        null_reasons.append("finished_request_stats_unavailable")
    return RequestTiming(
        queued_ts=values["queued_ts"],
        scheduled_ts=values["scheduled_ts"],
        first_token_ts=values["first_token_ts"],
        last_token_ts=values["last_token_ts"],
        first_token_latency=values["first_token_latency"],
        finished_request_stats=finished_request_stats,
        null_reasons=tuple(null_reasons),
    )


@dataclass
class _RequestOutputView:
    request_id: str
    prompt_token_ids: Sequence[int]
    output_token_ids: Sequence[int]
    num_cached_tokens: int | None
    num_cache_creation_tokens: int | None
    finished: bool
    finish_reason: str | None
    timing: RequestTiming


def _view_of(output: Any) -> _RequestOutputView:
    """Adapt a real ``vllm.RequestOutput`` (a list-of-completions shape)
    into this module's flat :class:`RequestOutputLike` surface.

    Refuses (raises) unless exactly one completion is present: this
    protocol's fixed sampling contract never sets ``n`` above 1, so more
    than one completion (or zero) means the engine did not honor it.
    Exact per-completion fields (``token_ids``/``finish_reason``) are read
    from ``outputs[0]``, never from a nonexistent top-level
    ``output_token_ids``/``finish_reason``. ``num_cached_tokens`` is
    genuinely ``int | None``: vLLM 0.28.0 reports ``None`` when prefix
    caching is disabled (this protocol's A lane), which is a legitimate,
    non-fatal outcome captured as-is rather than raised on.
    """

    completions = getattr(output, "outputs", None)
    if not isinstance(completions, list) or len(completions) != 1:
        observed = len(completions) if isinstance(completions, list) else "a non-list"
        raise KVTruthProtocolError(
            f"engine returned {observed} completions for one request; this "
            "protocol requires exactly one (n=1) sample per request"
        )
    completion = completions[0]
    num_cached_raw = getattr(output, "num_cached_tokens", None)
    num_cached_tokens = (
        int(num_cached_raw)
        if isinstance(num_cached_raw, int) and not isinstance(num_cached_raw, bool)
        else None
    )
    prompt_token_ids = tuple(getattr(output, "prompt_token_ids", ()) or ())
    output_token_ids = tuple(getattr(completion, "token_ids", ()) or ())
    num_created_raw = getattr(output, "num_cache_creation_tokens", None)
    num_cache_creation_tokens = (
        int(num_created_raw)
        if isinstance(num_created_raw, int) and not isinstance(num_created_raw, bool)
        else None
    )
    return _RequestOutputView(
        request_id=output.request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        num_cached_tokens=num_cached_tokens,
        num_cache_creation_tokens=num_cache_creation_tokens,
        finished=bool(output.finished),
        finish_reason=getattr(completion, "finish_reason", None),
        timing=_extract_timing(output),
    )


class _LLMEngineHandle:
    """Real :class:`EngineHandle` wrapping ``vllm.LLMEngine`` directly (not
    the high-level ``vllm.LLM`` convenience API), so this runner genuinely
    drives per-request ``add_request``/``step()`` dispatch and reads real
    ``RequestOutput``s -- the plan's "genuine offline LLMEngine runner"
    requirement. Safe under a plain ``while``/``step()`` loop because this
    protocol pins ``max_num_seqs=1``: no other request is ever concurrently
    in flight.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def reset_prefix_cache(self) -> bool:
        result = self._engine.reset_prefix_cache()
        return True if result is None else bool(result)

    def generate_one(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
    ) -> RequestOutputLike:
        from vllm import SamplingParams

        params = SamplingParams(**dict(sampling_params))
        # A plain ``{"prompt_token_ids": [...]}`` mapping matches vLLM's
        # ``TokensPrompt`` TypedDict shape structurally, so no import of
        # ``TokensPrompt`` itself (a class-vs-TypedDict API detail this
        # module cannot verify offline) is required to build a valid
        # tokens-prompt argument.
        self._engine.add_request(
            request_id, {"prompt_token_ids": list(prompt_token_ids)}, params
        )
        final_output: Any = None
        while self._engine.has_unfinished_requests():
            for output in self._engine.step():
                if output.request_id == request_id and output.finished:
                    final_output = output
            if final_output is not None:
                break
        if final_output is None:
            raise KVTruthProtocolError(
                f"engine never produced a finished RequestOutput for " f"{request_id!r}"
            )
        view = _view_of(final_output)
        if "metrics_unavailable" in view.timing.null_reasons:
            raise KVTruthProtocolError(
                "vLLM omitted RequestOutput.metrics while request stats were enabled"
            )
        return view


def build_llm(
    attestation: RuntimeAttestation,
    *,
    model_path: str,
    environ: Mapping[str, str] | None = None,
    cache_enabled: bool = True,
) -> EngineHandle:
    """Construct the pinned, real ``vllm.LLMEngine`` for this protocol.

    Enforces the offline/determinism environment contract and this exact
    protocol's identity attestation first, then lazily imports ``vllm`` so
    this module remains importable without it installed; a missing
    ``vllm`` package, or an unsatisfied environment, is treated as a
    refusal (:class:`KVTruthProtocolError`), not an unhandled import error.
    After construction, this independently reads the real engine's own
    resolved configuration back (:func:`_assert_engine_matches_protocol`)
    and refuses unless it matches this protocol's fixed pins exactly -- the
    engine's post-init state is never merely assumed from the pre-init
    attestation or the requested kwargs.
    """

    import os

    assert_offline_environment(os.environ if environ is None else environ)
    kwargs = build_engine_kwargs(
        attestation, model_path=model_path, cache_enabled=cache_enabled
    )
    try:
        from vllm import LLMEngine
        from vllm.config.kv_events import (  # type: ignore[import-not-found]
            KVEventsConfig,
        )
        from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KVTruthProtocolError(
            "vllm is not importable in this environment; refusing to build "
            "the pinned qwen3-8b-vllm-kv-truth-v1 engine"
        ) from exc
    engine_kwargs = dict(kwargs)
    kv_events_config = engine_kwargs.pop("kv_events_config")
    engine_kwargs["kv_events_config"] = (
        KVEventsConfig(**kv_events_config) if kv_events_config is not None else None
    )
    engine_args = EngineArgs(**engine_kwargs)
    engine = LLMEngine.from_engine_args(engine_args)
    _assert_engine_matches_protocol(engine, cache_enabled=cache_enabled)
    return _LLMEngineHandle(engine)


class LiveKVEventSubscriber:
    """Real :class:`EventCapture` wrapping a loopback-only ZMQ SUB socket
    plus a REQ replay-request socket, decoding every multipart frame
    with :func:`vllm_live.decode_sequence_frame`/
    :func:`vllm_live.parse_live_kv_event_batch`.

    ``zmq``/``msgspec`` are imported lazily inside :meth:`start`, never at
    module scope, so this class -- and this module -- stay importable
    offline. This is this repository's best-effort, clearly documented
    reconstruction of vLLM 0.28.0's actual wire protocol at the pinned
    commit (``vllm/distributed/kv_events.py``): the batch multipart
    framing and     the ``END_OF_REPLAY_SEQUENCE`` sentinel it drives are the exact schema
    confirmed against the pinned upstream source and v0.28.0 subscriber
    example. Any wire-shape mismatch fails closed with
    :class:`KVTruthProtocolError`/a schema-validation error inside the
    pinned container -- never a silently fabricated success -- and a
    request whose events never arrive is legitimately reported through
    ``run_request``'s own ``no_kv_events_observed_for_request`` boundary
    reason, not papered over here.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        replay_endpoint: str,
        topic: str,
        recv_timeout_ms: int = 200,
    ) -> None:
        if not endpoint.startswith("tcp://127.0.0.1:"):
            raise KVTruthProtocolError(
                f"KV-event endpoint must be loopback-only, got {endpoint!r}"
            )
        if not replay_endpoint.startswith("tcp://127.0.0.1:"):
            raise KVTruthProtocolError(
                "KV-event replay_endpoint must be loopback-only, got "
                f"{replay_endpoint!r}"
            )
        self._endpoint = endpoint
        self._replay_endpoint = replay_endpoint
        self._topic = topic
        self._recv_timeout_ms = recv_timeout_ms
        self._context: Any = None
        self._sub_socket: Any = None
        self._replay_socket: Any = None
        self._batches: list[LiveKVEventBatch] = []
        self._sequences_seen: set[int] = set()
        self._sequence_payload_digests: dict[int, str] = {}

    def start(self) -> None:
        try:
            import zmq  # type: ignore[import-not-found]
        except ImportError as exc:
            raise KVTruthProtocolError(
                "zmq is not importable in this environment; refusing to "
                "start the live KV-event subscriber"
            ) from exc
        self._context = zmq.Context.instance()
        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._sub_socket.setsockopt(zmq.SUBSCRIBE, self._topic.encode("utf-8"))
        # The pinned vLLM 0.28.0 KV-events publisher is itself a ZMQ PUB
        # socket that *connects out* to its configured endpoint (it expects
        # a long-lived listener already there), rather than binding one --
        # the opposite of the more common PUB-binds/SUB-connects pattern.
        # This SUB socket must therefore bind, not connect, and must do so
        # (via being called from ``run()``) strictly *before* the engine
        # -- and therefore the publisher -- is constructed, or the
        # publisher's connect attempt has nothing to connect to and every
        # KV event published before a late bind is silently lost (the ZMQ
        # slow-joiner problem, made worse here because a late bind means
        # the publisher never even successfully connects in the first
        # place, not merely that early messages are dropped after
        # connecting).
        self._sub_socket.bind(self._endpoint)
        # The replay endpoint is the mirror image: it is *this run's own*
        # ROUTER responder. The exact v0.28.0 subscriber contract uses REQ,
        # which supplies the empty delimiter frame expected by ROUTER.
        self._replay_socket = self._context.socket(zmq.REQ)
        self._replay_socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._replay_socket.connect(self._replay_endpoint)

    def replay_from_start(self) -> None:
        """Request and synchronously drain replay from sequence zero.

        ``run`` invokes this after engine construction but before the first
        cache operation. The SUB socket was already bound before engine
        construction, and replay now closes any publisher-startup race.
        """

        self._replay_socket.send((0).to_bytes(8, "big"))
        self._drain_replay()

    def _drain_replay(self) -> None:
        import zmq

        while True:
            try:
                frames = self._replay_socket.recv_multipart()
            except zmq.Again as exc:
                raise KVTruthProtocolError(
                    "KV-event replay timed out before its end sentinel"
                ) from exc
            if not self._ingest_frames(frames, allow_end_sentinel=True):
                break

    def _ingest_frames(
        self, frames: Sequence[bytes], *, allow_end_sentinel: bool
    ) -> bool:
        """Decode one multipart message; returns ``False`` on the replay
        end-of-buffer sentinel, ``True`` otherwise."""

        import msgspec  # type: ignore[import-not-found]

        if len(frames) != 3:
            raise KVTruthProtocolError(
                "KV-event frame has an unexpected multipart shape: "
                f"{len(frames)} frames"
            )
        topic_bytes, seq_bytes, payload_bytes = frames
        try:
            sequence = decode_sequence_frame(bytes(seq_bytes))
        except SchemaValidationError as exc:
            raise KVTruthProtocolError(
                f"malformed KV-event sequence frame: {exc}"
            ) from exc
        if sequence == END_OF_REPLAY_SEQUENCE:
            if not allow_end_sentinel:
                raise KVTruthProtocolError(
                    "received the end-of-replay sentinel on the live SUB socket"
                )
            if topic_bytes or payload_bytes:
                raise KVTruthProtocolError(
                    "replay end sentinel must have empty topic and payload frames"
                )
            return False
        expected_topic = self._topic.encode("utf-8")
        if bytes(topic_bytes) != expected_topic:
            raise KVTruthProtocolError("KV-event frame topic does not match protocol")
        payload_digest = hashlib.sha256(bytes(payload_bytes)).hexdigest()
        if sequence in self._sequences_seen:
            if self._sequence_payload_digests[sequence] != payload_digest:
                raise KVTruthProtocolError(
                    "duplicate KV-event sequence carried replacement payload bytes"
                )
            return True
        if self._sequences_seen and sequence != max(self._sequences_seen) + 1:
            raise KVTruthProtocolError("KV-event stream contains a sequence gap")
        try:
            payload = msgspec.msgpack.decode(bytes(payload_bytes))
            batch = parse_live_kv_event_batch(
                self._topic.encode("utf-8"), bytes(seq_bytes), payload
            )
        except SchemaValidationError as exc:
            raise KVTruthProtocolError(
                f"malformed KV-event batch payload: {exc}"
            ) from exc
        self._sequences_seen.add(sequence)
        self._sequence_payload_digests[sequence] = payload_digest
        self._batches.append(batch)
        return True

    def drain(self) -> tuple[LiveKVEventBatch, ...]:
        import zmq

        while True:
            try:
                frames = self._sub_socket.recv_multipart()
            except zmq.Again:
                break
            self._ingest_frames(frames, allow_end_sentinel=False)
        drained = tuple(self._batches)
        self._batches.clear()
        return drained

    def drain_before_dispatch(self) -> tuple[LiveKVEventBatch, ...]:
        return self.drain()

    def drain_reset(self) -> tuple[LiveKVEventBatch, ...]:
        batches: list[LiveKVEventBatch] = []
        quiet = 0
        while quiet < 2:
            drained = self.drain()
            if drained:
                batches.extend(drained)
                quiet = 0
            else:
                quiet += 1
            if quiet < 2:
                time.sleep(0.05)
        return tuple(batches)

    def stop(self) -> tuple[int | None, int | None]:
        late_batches = self.drain()
        if late_batches:
            raise KVTruthProtocolError(
                "KV events arrived after the final request capture boundary"
            )
        sequences = sorted(self._sequences_seen)
        first = sequences[0] if sequences else None
        last = sequences[-1] if sequences else None
        if first is not None and first != 0:
            raise KVTruthProtocolError(
                "KV-event capture did not begin at publisher sequence zero"
            )
        if self._sub_socket is not None:
            self._sub_socket.close(linger=0)
        if self._replay_socket is not None:
            self._replay_socket.close(linger=0)
        return (first, last)


def request_ids() -> tuple[str, ...]:
    """Return the strictly sequential, unique fixed request IDs for one B lane.

    One ID per nested probe in ``kv_truth_workload.NESTED_PROBES``, in the
    exact preregistered order; the plan requires unique fixed request IDs
    processed strictly sequentially, never concurrently and never reused.
    """

    return tuple(f"{PROTOCOL_ID}-probe-{probe.name}" for probe in NESTED_PROBES)


def eviction_lane_request_ids() -> tuple[str, ...]:
    return tuple(
        f"{PROTOCOL_ID}-eviction-{index:02d}"
        for index in range(len(EVICTION_LANE_REQUESTS))
    )


@dataclass(frozen=True)
class WorkloadManifest:
    """A canonical, hash-sealed description of the fixed workload.

    Recorded once as part of the evidence bundle so the exact probe order,
    token counts, and independent expectations used by a run can be
    re-derived and verified without re-running the workload module.
    """

    schema_version: str
    protocol_id: str
    block_size: int
    prefix_match_unit: int
    num_gpu_blocks_override: int
    probes: tuple[dict[str, Any], ...]
    eviction_lane_request_lengths: tuple[int, ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "block_size": self.block_size,
            "prefix_match_unit": self.prefix_match_unit,
            "num_gpu_blocks_override": self.num_gpu_blocks_override,
            "probes": list(self.probes),
            "eviction_lane_request_lengths": list(self.eviction_lane_request_lengths),
            "digest": self.digest,
        }


def build_workload_manifest() -> WorkloadManifest:
    """Build the canonical, hash-sealed manifest for the fixed workload."""

    probes = tuple(
        {
            "name": probe.name,
            "scenario": probe.scenario,
            "request_token_count": len(probe.request_tokens),
            "identity_matches": probe.identity_matches,
            "expected_reusable_tokens": probe.expected_reusable_tokens,
            "expected_reusable_blocks": probe.expected_reusable_blocks,
        }
        for probe in NESTED_PROBES
    )
    unsealed = {
        "schema_version": "1",
        "protocol_id": PROTOCOL_ID,
        "block_size": BLOCK_SIZE,
        "prefix_match_unit": PREFIX_MATCH_UNIT,
        "num_gpu_blocks_override": NUM_GPU_BLOCKS_OVERRIDE,
        "probes": list(probes),
        "eviction_lane_request_lengths": [
            len(request) for request in EVICTION_LANE_REQUESTS
        ],
    }
    digest = sha256_digest(unsealed)
    return WorkloadManifest(
        schema_version="1",
        protocol_id=PROTOCOL_ID,
        block_size=BLOCK_SIZE,
        prefix_match_unit=PREFIX_MATCH_UNIT,
        num_gpu_blocks_override=NUM_GPU_BLOCKS_OVERRIDE,
        probes=probes,
        eviction_lane_request_lengths=tuple(
            len(request) for request in EVICTION_LANE_REQUESTS
        ),
        digest=digest,
    )


def no_warmup_canary_reasons() -> tuple[str, ...]:
    """Return every way the fixed configuration would require a hidden warmup.

    A non-empty result means the protocol cannot claim "no hidden generation
    warmups": eager mode must be on (no CUDA graph capture warmup) and no
    speculative decoding may be configured (no draft-model warmup).
    """

    reasons: list[str] = []
    if not ENFORCE_EAGER:
        reasons.append("enforce_eager_disabled_requires_cuda_graph_warmup")
    if SPECULATIVE_CONFIG_ENABLED:
        reasons.append("speculative_config_enabled_requires_draft_warmup")
    return tuple(reasons)


# ---------------------------------------------------------------------------
# Executable protocol surface: engine/event-capture injection points
# ---------------------------------------------------------------------------


class RequestOutputLike(Protocol):
    """The minimal ``vllm.RequestOutput`` surface this runner reads."""

    request_id: str
    prompt_token_ids: Sequence[int]
    output_token_ids: Sequence[int]
    num_cached_tokens: int | None
    num_cache_creation_tokens: int | None
    finished: bool
    finish_reason: str | None
    timing: RequestTiming


class EngineHandle(Protocol):
    """The minimal constructed-engine surface this runner drives.

    A real implementation wraps ``vllm.LLM``: ``reset_prefix_cache`` calls
    ``self.llm_engine.reset_prefix_cache()`` and ``generate_one`` calls
    ``self.generate([...], SamplingParams(...))[0]``. Tests inject a fake
    that returns pre-programmed :class:`RequestOutputLike` values instead.
    """

    def reset_prefix_cache(self) -> bool: ...

    def generate_one(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
    ) -> RequestOutputLike: ...


class EventCapture(Protocol):
    """The minimal KV-event subscriber surface this runner drives.

    A real implementation wraps a ZMQ SUB socket bound to
    ``kv_events_config_kwargs()``'s loopback endpoint, replaying from
    sequence zero before the engine is constructed to avoid the slow-joiner
    gap, and decodes each multipart frame with
    ``vllm_live.parse_live_kv_event_batch``. Tests inject a fake that
    returns pre-programmed batches instead.
    """

    def start(self) -> None: ...

    def replay_from_start(self) -> None: ...

    def drain_before_dispatch(self) -> tuple[LiveKVEventBatch, ...]:
        """Drain events that arrived outside any request boundary."""
        ...

    def drain_reset(self) -> tuple[LiveKVEventBatch, ...]:
        """Drain the bounded reset boundary after ``reset_prefix_cache``."""
        ...

    def drain(self) -> tuple[LiveKVEventBatch, ...]:
        """Return, and clear, every batch received since the last drain."""
        ...

    def stop(self) -> tuple[int | None, int | None]:
        """Stop capturing and return ``(first_sequence, last_sequence)``."""
        ...


@dataclass(frozen=True)
class RequestRecord:
    """One executed request's evidence: engine-attested plus event-bound facts.

    ``event_batches`` are exactly the batches drained between this request's
    dispatch and its bounded quiet-poll boundary; a gap, an unexpected late
    arrival, or an empty capture where one was required invalidates the
    lifecycle per the plan, surfaced here as ``boundary_reasons``.

    ``prompt_token_ids``/``output_token_ids`` are the exact, private token
    arrays for this request. They are never included in any public/redacted
    evidence view -- only this private raw receipt.
    """

    request_id: str
    scenario: str
    prompt_token_count: int
    output_token_count: int
    num_cached_tokens: int | None
    num_cache_creation_tokens: int | None
    finished: bool
    finish_reason: str | None
    event_batches: tuple[LiveKVEventBatch, ...]
    prompt_token_ids: tuple[int, ...] = ()
    output_token_ids: tuple[int, ...] = ()
    timing: RequestTiming | None = None
    boundary_reasons: tuple[str, ...] = ()

    @property
    def boundary_valid(self) -> bool:
        return not self.boundary_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "prompt_token_count": self.prompt_token_count,
            "output_token_count": self.output_token_count,
            "num_cached_tokens": self.num_cached_tokens,
            "num_cache_creation_tokens": self.num_cache_creation_tokens,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "event_batches": [batch.to_dict() for batch in self.event_batches],
            # Private-only: the exact token arrays, never surfaced in any
            # public/redacted evidence view.
            "prompt_token_ids": list(self.prompt_token_ids),
            "output_token_ids": list(self.output_token_ids),
            "timing": self.timing.to_dict() if self.timing is not None else None,
            "boundary_valid": self.boundary_valid,
            "boundary_reasons": list(self.boundary_reasons),
        }


def _boundary_reasons_for(
    *,
    request_id: str,
    expect_events: bool,
    batches: Sequence[LiveKVEventBatch],
    expect_cache_counts: bool,
    num_cached_tokens: int | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if expect_events and not any(batch.events for batch in batches):
        reasons.append("no_kv_events_observed_for_request")
    sequences = [batch.sequence for batch in batches]
    if sequences != sorted(sequences):
        reasons.append("kv_events_out_of_order_for_request")
    if len(set(sequences)) != len(sequences):
        reasons.append("kv_events_duplicated_for_request")
    if expect_cache_counts and num_cached_tokens is None:
        reasons.append("cached_tokens_unavailable")
    if not expect_cache_counts and num_cached_tokens is not None:
        reasons.append("cached_tokens_unexpectedly_present")
    return tuple(reasons)


def run_request(
    engine: EngineHandle,
    capture: EventCapture | None,
    *,
    request_id: str,
    scenario: str,
    prompt_token_ids: Sequence[int],
    sampling_params: Mapping[str, Any] | None = None,
    expect_cache_counts: bool = True,
    quiet_polls: int = 2,
    poll_interval_seconds: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
) -> RequestRecord:
    """Execute exactly one request and bind its KV-event capture boundary.

    Refuses an apparently clean boundary when events are already pending
    before dispatch, then drains after execution until ``quiet_polls``
    consecutive drains come back empty. A gap, duplicate, out-of-order, or
    late event is recorded in ``boundary_reasons`` rather than silently
    accepted.

    ``expect_cache_counts`` distinguishes the cache-disabled A lane (where a
    genuine ``num_cached_tokens is None`` is the expected, non-fatal outcome)
    from the cache-enabled B/eviction lanes (where a ``None`` is itself a
    boundary violation worth flagging, and an unexpectedly-present count on
    a lane that disabled caching is flagged the other way).
    """

    late_batches: tuple[LiveKVEventBatch, ...] = ()
    if capture is not None:
        late_batches = capture.drain_before_dispatch()
    output = engine.generate_one(
        request_id, prompt_token_ids, dict(sampling_params or SAMPLING_PARAMS)
    )
    if output.request_id != request_id:
        raise KVTruthProtocolError(
            f"engine returned output for {output.request_id!r}, expected "
            f"{request_id!r}; ambiguous request/event binding"
        )

    batches: list[LiveKVEventBatch] = []
    if capture is not None:
        quiet = 0
        while quiet < quiet_polls:
            drained = capture.drain()
            if drained:
                batches.extend(drained)
                quiet = 0
            else:
                quiet += 1
            if quiet < quiet_polls:
                sleep(poll_interval_seconds)

    boundary_reasons = list(
        _boundary_reasons_for(
            request_id=request_id,
            expect_events=capture is not None,
            batches=batches,
            expect_cache_counts=expect_cache_counts,
            num_cached_tokens=output.num_cached_tokens,
        )
    )
    if late_batches:
        boundary_reasons.append("late_kv_events_before_request_dispatch")
    timing = getattr(output, "timing", None)
    return RequestRecord(
        request_id=request_id,
        scenario=scenario,
        prompt_token_count=len(output.prompt_token_ids),
        output_token_count=len(output.output_token_ids),
        num_cached_tokens=output.num_cached_tokens,
        num_cache_creation_tokens=output.num_cache_creation_tokens,
        finished=output.finished,
        finish_reason=output.finish_reason,
        event_batches=late_batches + tuple(batches),
        prompt_token_ids=tuple(output.prompt_token_ids),
        output_token_ids=tuple(output.output_token_ids),
        timing=timing,
        boundary_reasons=tuple(boundary_reasons),
    )


@dataclass(frozen=True)
class LaneResult:
    """The complete, ordered record of one executed lane (A, B, or eviction)."""

    lane: str
    records: tuple[RequestRecord, ...]
    reset_event_batches: tuple[LiveKVEventBatch, ...] = ()
    reset_boundary_reasons: tuple[str, ...] = ()

    @property
    def all_boundaries_valid(self) -> bool:
        return not self.reset_boundary_reasons and all(
            record.boundary_valid for record in self.records
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "reset_event_batches": [
                batch.to_dict() for batch in self.reset_event_batches
            ],
            "reset_boundary_reasons": list(self.reset_boundary_reasons),
            "records": [record.to_dict() for record in self.records],
            "all_boundaries_valid": self.all_boundaries_valid,
        }


def _capture_reset_boundary(
    capture: EventCapture,
) -> tuple[tuple[LiveKVEventBatch, ...], tuple[str, ...]]:
    batches = capture.drain_reset()
    reasons: list[str] = []
    sequences = [batch.sequence for batch in batches]
    if not any(
        event.event_type.value == "AllBlocksCleared"
        for batch in batches
        for event in batch.events
    ):
        reasons.append("reset_all_blocks_cleared_event_missing")
    if sequences != sorted(sequences):
        reasons.append("reset_kv_events_out_of_order")
    if len(set(sequences)) != len(sequences):
        reasons.append("reset_kv_events_duplicated")
    return batches, tuple(reasons)


def run_b_lane(engine: EngineHandle, capture: EventCapture) -> LaneResult:
    """Run the fixed cache-enabled nested probe sequence (plan: "B" lane).

    Resets the prefix cache first, matching the plan's "cold seed ... after
    verified reset", then executes each nested probe strictly sequentially
    under its unique fixed request ID, in the exact preregistered order.
    """

    reset_ok = engine.reset_prefix_cache()
    if not reset_ok:
        raise KVTruthProtocolError(
            "reset_prefix_cache() reported failure; refusing to start the B lane"
        )
    reset_batches, reset_reasons = _capture_reset_boundary(capture)
    ids = request_ids()
    records = tuple(
        run_request(
            engine,
            capture,
            request_id=request_id,
            scenario=probe.scenario,
            prompt_token_ids=probe.request_tokens,
        )
        for request_id, probe in zip(ids, NESTED_PROBES, strict=True)
    )
    return LaneResult(
        lane="B",
        records=records,
        reset_event_batches=reset_batches,
        reset_boundary_reasons=reset_reasons,
    )


def run_a_lane(engine: EngineHandle) -> LaneResult:
    """Run the fixed cache-disabled output-identity control (plan: "A" lane).

    Executes the exact same token arrays and sampling contract as the B
    lane, against a fresh engine constructed with caching disabled, to
    establish terminal output-token identity only; no event capture is
    attached because a cache-disabled engine publishes no KV-cache events.
    """

    ids = request_ids()
    records = tuple(
        run_request(
            engine,
            None,
            request_id=request_id,
            scenario=probe.scenario,
            prompt_token_ids=probe.request_tokens,
            expect_cache_counts=False,
        )
        for request_id, probe in zip(ids, NESTED_PROBES, strict=True)
    )
    return LaneResult(lane="A", records=records)


def run_eviction_lane(engine: EngineHandle, capture: EventCapture) -> LaneResult:
    """Run the fixed eviction lane: reset, seed, five fillers, final seed probe."""

    reset_ok = engine.reset_prefix_cache()
    if not reset_ok:
        raise KVTruthProtocolError(
            "reset_prefix_cache() reported failure; refusing to start the "
            "eviction lane"
        )
    reset_batches, reset_reasons = _capture_reset_boundary(capture)
    ids = eviction_lane_request_ids()
    scenarios = (
        "eviction_seed",
        *(f"eviction_filler_{index}" for index in range(5)),
        "eviction_final_probe",
    )
    records = tuple(
        run_request(
            engine,
            capture,
            request_id=request_id,
            scenario=scenario,
            prompt_token_ids=tokens,
        )
        for request_id, scenario, tokens in zip(
            ids, scenarios, EVICTION_LANE_REQUESTS, strict=True
        )
    )
    return LaneResult(
        lane="eviction",
        records=records,
        reset_event_batches=reset_batches,
        reset_boundary_reasons=reset_reasons,
    )


@dataclass(frozen=True)
class ProtocolReceipt:
    """A canonical, hash-sealed receipt covering one lane's execution."""

    schema_version: str
    protocol_id: str
    lane: str
    workload_digest: str
    lane_result: dict[str, Any]
    runtime_attestation: dict[str, Any] | None = None
    digest: str = field(compare=False, default="")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "lane": self.lane,
            "workload_digest": self.workload_digest,
            "lane_result": self.lane_result,
            "runtime_attestation": self.runtime_attestation,
            "digest": self.digest,
        }


def build_protocol_receipt(
    result: LaneResult,
    *,
    runtime_attestation: Mapping[str, Any] | None = None,
) -> ProtocolReceipt:
    """Build the canonical, hash-sealed receipt for one executed lane."""

    lane_result = result.to_dict()
    workload_digest = build_workload_manifest().digest
    unsealed = {
        "schema_version": "1",
        "protocol_id": PROTOCOL_ID,
        "lane": result.lane,
        "workload_digest": workload_digest,
        "lane_result": lane_result,
        "runtime_attestation": (
            dict(runtime_attestation) if runtime_attestation is not None else None
        ),
    }
    return ProtocolReceipt(
        schema_version="1",
        protocol_id=PROTOCOL_ID,
        lane=result.lane,
        workload_digest=workload_digest,
        lane_result=lane_result,
        runtime_attestation=(
            dict(runtime_attestation) if runtime_attestation is not None else None
        ),
        digest=sha256_digest(unsealed),
    )


def write_protocol_receipt(receipt: ProtocolReceipt, output: Path) -> None:
    """Persist ``receipt`` as canonical JSON, refusing to overwrite silently."""

    if output.exists():
        raise KVTruthProtocolError(f"refusing to overwrite existing receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, canonical_json(receipt.to_dict()) + "\n")


def _validate_lane_result_payload(lane: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise KVTruthProtocolError("protocol receipt lane_result is not an object")
    expected_keys = {
        "lane",
        "reset_event_batches",
        "reset_boundary_reasons",
        "records",
        "all_boundaries_valid",
    }
    if set(value) != expected_keys or value["lane"] != lane:
        raise KVTruthProtocolError(
            "protocol receipt lane_result does not match the exact schema/lane"
        )
    reset_batches = value["reset_event_batches"]
    reset_reasons = value["reset_boundary_reasons"]
    records = value["records"]
    if (
        not isinstance(reset_batches, list)
        or not isinstance(reset_reasons, list)
        or not all(isinstance(reason, str) for reason in reset_reasons)
        or not isinstance(records, list)
    ):
        raise KVTruthProtocolError("protocol receipt lane_result fields are malformed")

    parsed_reset = [
        parse_live_kv_event_batch(
            batch["topic"].encode("utf-8"),
            int(batch["sequence"]).to_bytes(8, "big"),
            [batch["ts"], batch["events"], batch["data_parallel_rank"]],
        )
        for batch in reset_batches
    ]
    if lane == "A":
        if parsed_reset or reset_reasons:
            raise KVTruthProtocolError("A lane must not contain a reset event boundary")
        expected_request_ids = request_ids()
    else:
        if reset_reasons or not any(
            event.event_type.value == "AllBlocksCleared"
            for batch in parsed_reset
            for event in batch.events
        ):
            raise KVTruthProtocolError(
                "cache-enabled lane lacks a valid AllBlocksCleared reset boundary"
            )
        expected_request_ids = (
            request_ids() if lane == "B" else eviction_lane_request_ids()
        )
    if len(records) != len(expected_request_ids):
        raise KVTruthProtocolError(
            "protocol receipt has the wrong number of request records"
        )

    record_keys = {
        "request_id",
        "scenario",
        "prompt_token_count",
        "output_token_count",
        "num_cached_tokens",
        "num_cache_creation_tokens",
        "finished",
        "finish_reason",
        "event_batches",
        "prompt_token_ids",
        "output_token_ids",
        "timing",
        "boundary_valid",
        "boundary_reasons",
    }
    derived_valid = not reset_reasons
    for expected_request_id, record in zip(expected_request_ids, records, strict=True):
        if not isinstance(record, dict) or set(record) != record_keys:
            raise KVTruthProtocolError(
                "protocol receipt request record does not match the exact schema"
            )
        if record["request_id"] != expected_request_id:
            raise KVTruthProtocolError(
                "protocol receipt request IDs/order do not match the fixed workload"
            )
        prompt_ids = record["prompt_token_ids"]
        output_ids = record["output_token_ids"]
        reasons = record["boundary_reasons"]
        batches = record["event_batches"]
        if (
            not isinstance(prompt_ids, list)
            or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in prompt_ids
            )
            or not isinstance(output_ids, list)
            or not all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in output_ids
            )
            or record["prompt_token_count"] != len(prompt_ids)
            or record["output_token_count"] != len(output_ids)
            or not isinstance(reasons, list)
            or not all(isinstance(reason, str) for reason in reasons)
            or record["boundary_valid"] != (not reasons)
            or record["finished"] is not True
            or not isinstance(batches, list)
        ):
            raise KVTruthProtocolError("protocol receipt request record is malformed")
        parsed_batches = [
            parse_live_kv_event_batch(
                batch["topic"].encode("utf-8"),
                int(batch["sequence"]).to_bytes(8, "big"),
                [batch["ts"], batch["events"], batch["data_parallel_rank"]],
            )
            for batch in batches
        ]
        if lane == "A":
            if parsed_batches or record["num_cached_tokens"] is not None:
                raise KVTruthProtocolError(
                    "A lane contains cache events or cached-token counts"
                )
        else:
            if (
                not any(batch.events for batch in parsed_batches)
                or not isinstance(record["num_cached_tokens"], int)
                or isinstance(record["num_cached_tokens"], bool)
            ):
                raise KVTruthProtocolError(
                    "cache-enabled request lacks events or cached-token count"
                )
            all_tokens = prompt_ids + output_ids
            stored_events = [
                event
                for batch in parsed_batches
                for event in batch.events
                if isinstance(event, LiveBlockStored)
            ]
            if not stored_events:
                raise KVTruthProtocolError(
                    "cache-enabled request lacks a BlockStored binding event"
                )
            for event in stored_events:
                event_tokens = list(event.token_ids)
                if event.medium != "GPU" or event.group_idx != 0:
                    raise KVTruthProtocolError(
                        "BlockStored event lacks fixed GPU/group metadata"
                    )
                if not any(
                    all_tokens[start : start + len(event_tokens)] == event_tokens
                    for start in range(0, len(all_tokens), event.block_size)
                ):
                    raise KVTruthProtocolError(
                        "BlockStored event token IDs do not bind to its request"
                    )
        derived_valid = derived_valid and not reasons
    if value["all_boundaries_valid"] is not derived_valid:
        raise KVTruthProtocolError(
            "protocol receipt all_boundaries_valid is inconsistent"
        )


def verify_protocol_receipt(path: Path) -> ProtocolReceipt:
    """Load and verify a persisted receipt's seal, refusing on any mismatch."""

    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise KVTruthProtocolError(f"{path} is not a JSON object")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "lane",
        "workload_digest",
        "lane_result",
        "runtime_attestation",
        "digest",
    }
    if set(raw) != expected_keys:
        raise KVTruthProtocolError(
            f"{path} protocol receipt keys do not match the exact schema"
        )
    digest = raw.get("digest")
    unsealed = {key: value for key, value in raw.items() if key != "digest"}
    if sha256_digest(unsealed) != digest:
        raise KVTruthProtocolError(f"{path} digest does not match its contents")
    if raw["schema_version"] != "1" or raw["protocol_id"] != PROTOCOL_ID:
        raise KVTruthProtocolError(f"{path} protocol identity is invalid")
    if raw["lane"] not in {"A", "B", "eviction"}:
        raise KVTruthProtocolError(f"{path} lane is invalid")
    if raw["workload_digest"] != build_workload_manifest().digest:
        raise KVTruthProtocolError(f"{path} workload digest is invalid")
    _validate_lane_result_payload(raw["lane"], raw["lane_result"])
    runtime_attestation = raw["runtime_attestation"]
    if runtime_attestation is not None:
        if not isinstance(runtime_attestation, dict):
            raise KVTruthProtocolError(
                f"{path} runtime_attestation is not an object or null"
            )
        try:
            runtime_attestation = parse_runtime_attestation(
                runtime_attestation
            ).to_dict()
        except SchemaValidationError as exc:
            raise KVTruthProtocolError(
                f"{path} runtime_attestation is invalid: {exc}"
            ) from exc
    return ProtocolReceipt(
        schema_version=raw["schema_version"],
        protocol_id=raw["protocol_id"],
        lane=raw["lane"],
        workload_digest=raw["workload_digest"],
        lane_result=raw["lane_result"],
        runtime_attestation=runtime_attestation,
        digest=raw["digest"],
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

PROG = "llmtracefx-kv-truth-runner"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Run one fixed lane of the qwen3-8b-vllm-kv-truth-v1 protocol "
            "inside the pinned container. Never executable outside a real "
            "GPU container with the exact pinned vLLM installed; every "
            "failure mode is a genuine validation/runtime error, never a "
            "refusal by construction."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--lane", choices=("A", "B", "eviction"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--attestation",
        required=True,
        type=Path,
        help=(
            "path this run writes its freshly generated, self-verifying "
            "pre-init identity receipt to (for evidence collection); never "
            "a caller-supplied attestation to trust"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(
    args: argparse.Namespace,
    *,
    build_attestation: Callable[[Path, str], RuntimeAttestation],
    make_engine: Callable[[RuntimeAttestation, str, bool], EngineHandle],
    make_capture: Callable[[], EventCapture] | None = None,
) -> ProtocolReceipt:
    """Execute one lane end to end and persist its receipt.

    Every real dependency (attestation construction, engine construction,
    event capture) is injected so this function -- the actual dispatch and
    sequencing logic -- is exercised by tests with fakes; only
    ``main`` wires the real, lazily-imported implementations. The "A" lane
    always constructs a cache-*disabled* engine (an output-identity
    control); the "B" and eviction lanes always construct a cache-*enabled*
    engine, matching the plan's requirement that the A lane genuinely
    differ from B/eviction, not merely be labeled differently. For "B" and
    eviction, ``capture.start()`` (which binds the real subscriber's SUB
    socket) is always called *before* ``make_engine`` (which constructs the
    real vLLM KV-events publisher): the pinned publisher connects out to
    this endpoint rather than binding it, so the SUB side must already be
    bound and listening or every event published during/just after engine
    construction is silently lost.
    """

    attestation = build_attestation(args.attestation, args.model_path)
    cache_enabled = args.lane != "A"

    if args.lane == "A":
        # The A lane never uses event capture (it is the cache-disabled
        # control), so there is no publisher/subscriber bind ordering
        # concern here: the engine can be constructed directly.
        engine = make_engine(attestation, args.model_path, cache_enabled)
        result = run_a_lane(engine)
    else:
        if make_capture is None:
            raise KVTruthProtocolError(
                f"lane {args.lane!r} requires an event capture but none was provided"
            )
        # The pinned vLLM 0.28.0 KV-events publisher is itself a ZMQ PUB
        # socket that connects out to its configured endpoint rather than
        # binding one, so the capture's SUB socket must already be bound
        # -- via ``capture.start()`` -- *before* the engine (and therefore
        # the publisher) is constructed, or the publisher's connect
        # attempt finds nothing listening and every KV event it emits
        # during/immediately after engine construction is silently lost.
        capture = make_capture()
        capture_started = time.monotonic()
        capture.start()
        first_sequence: int | None = None
        last_sequence: int | None = None
        try:
            engine = make_engine(attestation, args.model_path, cache_enabled)
            capture.replay_from_start()
            if args.lane == "B":
                result = run_b_lane(engine, capture)
            else:
                result = run_eviction_lane(engine, capture)
        finally:
            first_sequence, last_sequence = capture.stop()
        attestation = bind_capture_boundary(
            attestation,
            first_sequence=first_sequence,
            last_sequence=last_sequence,
            capture_start_monotonic=capture_started,
            capture_end_monotonic=time.monotonic(),
        )

    if not result.all_boundaries_valid:
        raise KVTruthProtocolError(
            f"lane {result.lane!r} contains invalid reset or request boundaries"
        )
    receipt = build_protocol_receipt(result, runtime_attestation=attestation.to_dict())
    write_protocol_receipt(receipt, args.output)
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Real CLI entry point.

    This is genuinely executable inside the pinned in-container GPU
    environment: it generates a real identity receipt and runtime
    attestation from this process's own live installed packages, GPU, and
    verified local model files (:func:`generate_identity_receipt`,
    :func:`generate_runtime_attestation`), constructs a real
    ``vllm.LLMEngine`` (:func:`build_llm`), and -- for the "B"/eviction
    lanes -- a real loopback ZMQ KV-event subscriber
    (:class:`LiveKVEventSubscriber`). It can fail only on an actual
    validation or runtime error (missing/wrong package, GPU, model files,
    or a real ``vllm``/``zmq`` import failure), reported as
    :class:`KVTruthProtocolError` (exit code 1), never by construction.
    """

    import os

    args = build_parser().parse_args(argv)

    def build_attestation(
        attestation_path: Path, model_path: str
    ) -> RuntimeAttestation:
        vllm_package_root = _discover_vllm_package_root()
        identity = generate_identity_receipt(
            model_path=model_path,
            vllm_package_root=vllm_package_root,
            environ=os.environ,
        )
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation = generate_runtime_attestation(identity)
        atomic_write_text(
            attestation_path, canonical_json(attestation.to_dict()) + "\n"
        )
        return attestation

    def make_engine(
        attestation: RuntimeAttestation, model_path: str, cache_enabled: bool
    ) -> EngineHandle:
        return build_llm(
            attestation, model_path=model_path, cache_enabled=cache_enabled
        )

    def make_capture() -> EventCapture:
        config = kv_events_config_kwargs()
        return LiveKVEventSubscriber(
            endpoint=config["endpoint"],
            replay_endpoint=config["replay_endpoint"],
            topic=config["topic"],
        )

    try:
        receipt = run(
            args,
            build_attestation=build_attestation,
            make_engine=make_engine,
            make_capture=make_capture,
        )
        if receipt.runtime_attestation is None:
            raise KVTruthProtocolError("real run did not produce a runtime attestation")
        atomic_write_text(
            args.attestation, canonical_json(receipt.runtime_attestation) + "\n"
        )
    except KVTruthProtocolError as exc:
        print(f"{PROG}: refused: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "CACHE_DTYPE",
    "DATA_PARALLEL_SIZE",
    "ENABLE_KV_CACHE_EVENTS",
    "ENABLE_PREFIX_CACHING",
    "ENFORCE_EAGER",
    "EXPECTED_EXPERIMENT_NONCE_ENV",
    "EXPECTED_IMAGE_ID_ENV",
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MODEL_FILE_COUNT",
    "EXPECTED_REPOSITORY_COMMIT_ENV",
    "EngineHandle",
    "EventCapture",
    "GPU_MEMORY_UTILIZATION",
    "KVTruthProtocolError",
    "KV_EVENTS_LOOPBACK_PORT",
    "KV_EVENTS_REPLAY_PORT",
    "KV_EVENTS_TOPIC",
    "KV_EVENTS_USE_INT_BLOCK_HASHES",
    "KVTruthProbe",
    "LORA_ENABLED",
    "LaneResult",
    "LiveKVEventSubscriber",
    "MAX_MODEL_LEN",
    "MAX_NUM_SEQS",
    "MAX_TOKENS",
    "MINIMUM_FREE_VRAM_MIB_RESERVE",
    "MULTIMODAL_ENABLED",
    "PLAN_APPROVED_REPOSITORY_COMMIT",
    "PREFIX_CACHING_HASH_ALGO",
    "PROTOCOL_ID",
    "ProtocolReceipt",
    "REQUIRED_ENVIRONMENT_VARIABLES",
    "RequestOutputLike",
    "RequestRecord",
    "SAMPLING_PARAMS",
    "SAMPLING_SEED",
    "SPECULATIVE_CONFIG_ENABLED",
    "TEMPERATURE",
    "TENSOR_PARALLEL_SIZE",
    "TOP_P",
    "WorkloadManifest",
    "assert_offline_environment",
    "assert_protocol_attestation",
    "build_engine_kwargs",
    "build_llm",
    "build_parser",
    "build_protocol_receipt",
    "build_workload_manifest",
    "canonical_json",
    "collect_gpu_facts",
    "collect_installed_distributions_digest",
    "collect_package_tree_digest",
    "collect_runtime_versions",
    "collect_source_file_digests",
    "collect_wheel_record_digest",
    "environment_reasons",
    "eviction_lane_request_ids",
    "expected_image_id",
    "expected_repository_commit",
    "generate_identity_receipt",
    "generate_runtime_attestation",
    "kv_events_config_kwargs",
    "main",
    "model_path_commitment",
    "no_warmup_canary_reasons",
    "protocol_reasons_for_attestation",
    "request_ids",
    "run",
    "run_a_lane",
    "run_b_lane",
    "run_eviction_lane",
    "run_request",
    "runner_source_digest",
    "verify_model_inventory",
    "verify_protocol_receipt",
    "write_protocol_receipt",
]
