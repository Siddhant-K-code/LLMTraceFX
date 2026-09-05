"""Modal entrypoint for the Qwen3-8B L4 crossover delta.

This module is the only file in the package that imports the Modal SDK.
Nothing on a planning, verification, or evidence path imports it, which
is what keeps every offline contract testable with no provider package
installed and no possibility of an accidental provider call.

Import order matters. The sealed offline feasibility gate runs before the SDK
import and currently refuses this protocol identity. The dormant declaration
path then imports the SDK with an install message rather than a bare
``ModuleNotFoundError``; probes the pinned API surface; reads run-scoped,
non-credential configuration; and only then declares the app, volume, images,
and Functions from the sealed plan.

There is no web endpoint, no ``modal.Secret``, and no credential of any
kind. Work is invoked over authenticated Modal RPC by the operator's own
profile, and every Function is single-use, one input at a time, one
container at a time, with zero retries and an explicit timeout.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any

from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_cell_runner as runner
from llmtracefx.optimizer.lab.qwen3_8b.modal_l4_crossover import (
    CONTAINER_CACHE_ROOT,
    CONTAINER_OUTPUT_ROOT,
    FUNCTION_SPEC_BY_KEY,
    MODEL_MOUNT_PATH,
    NO_WEB_ENDPOINT_DECORATORS,
    PROTOCOL_ID,
    STAGING_IMAGE_HF_HUB_PIN,
    STAGING_IMAGE_PYTHON_VERSION,
    ModalL4ContractError,
    build_default_plan,
    require_controlled_cell_decode_feasible,
    run_scoped_names,
    verify_sdk_capabilities,
)
from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import BASE_IMAGE_REFERENCE

# This provider entrypoint is a second, independent spend barrier. Even a
# direct ``modal run`` that bypasses the local orchestrator must prove the
# sealed protocol feasible before the Modal SDK is imported or any resource is
# declared. The current L4 identity is infeasible, so production import stops
# here. Tests patch this pure gate only to exercise the dormant declaration
# contract against a fake SDK.
require_controlled_cell_decode_feasible()

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on Modal
    raise SystemExit(
        "The Modal SDK is required to run this entrypoint. It is an optional "
        "dependency of llmtracefx: install the exact tested SDK with "
        "`uv sync --extra modal-l4-execute` (pins modal==1.5.5). Planning, "
        "verification, and evidence do not need it and never import this "
        "module."
    ) from exc

APP_NAME_VAR = "LLMTRACEFX_MODAL_L4_APP_NAME"
VOLUME_NAME_VAR = "LLMTRACEFX_MODAL_L4_VOLUME_NAME"
NONCE_VAR = "LLMTRACEFX_MODAL_L4_NONCE"
PLAN_SHA256_VAR = "LLMTRACEFX_MODAL_L4_PLAN_SHA256"

SDK_CAPABILITIES = verify_sdk_capabilities(modal)

PLAN = build_default_plan()
EXPERIMENT_NONCE = (os.environ.get(NONCE_VAR) or "").strip()
NAMES = run_scoped_names(EXPERIMENT_NONCE)
if (os.environ.get(APP_NAME_VAR) or NAMES["app_name"]) != NAMES["app_name"]:
    raise ModalL4ContractError("app name does not match the run-scoped nonce")
if (os.environ.get(VOLUME_NAME_VAR) or NAMES["volume_name"]) != NAMES["volume_name"]:
    raise ModalL4ContractError("volume name does not match the run-scoped nonce")
if (os.environ.get(PLAN_SHA256_VAR) or PLAN.content_sha256) != PLAN.content_sha256:
    raise ModalL4ContractError("orchestrator plan hash differs from this plan")

app = modal.App(NAMES["app_name"])

# One run-scoped volume. It is created here and deleted by the local
# orchestrator's teardown; nothing else in the workspace shares its name.
# The measured functions receive it read-only through the documented
# ``with_mount_options(read_only=True)`` API (modal 1.5.5); the deprecated
# ``Volume.read_only()`` is deliberately not used.
model_volume = modal.Volume.from_name(NAMES["volume_name"], create_if_missing=True)
read_only_model_volume = model_volume.with_mount_options(read_only=True)

# Staging only needs a hub client. It is deliberately not the runtime
# image: a container that downloads several GiB has no business carrying
# a CUDA userspace, and this one never sees an accelerator.
staging_image = modal.Image.debian_slim(
    python_version=STAGING_IMAGE_PYTHON_VERSION
).pip_install(STAGING_IMAGE_HF_HUB_PIN)

# Everything measured runs on the digest-pinned upstream image the
# protocol froze. The tag carries the digest, so a moved tag cannot
# silently change the runtime under a sealed experiment.
runtime_image = modal.Image.from_registry(BASE_IMAGE_REFERENCE)


def _writable_volumes() -> (
    dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
):
    return {MODEL_MOUNT_PATH: model_volume}


def _read_only_volumes() -> (
    dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]
):
    return {MODEL_MOUNT_PATH: read_only_model_volume}


def _image(spec_key: str) -> Any:
    spec = FUNCTION_SPEC_BY_KEY[spec_key]
    return staging_image if spec.image == "staging" else runtime_image


def _volumes(
    spec_key: str,
) -> dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount]:
    spec = FUNCTION_SPEC_BY_KEY[spec_key]
    return (
        _writable_volumes()
        if spec.model_volume_mode == "read_write"
        else _read_only_volumes()
    )


@app.function(
    image=_image("stage"),
    volumes=_volumes("stage"),
    **FUNCTION_SPEC_BY_KEY["stage"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def stage_model() -> dict[str, Any]:
    """Download the pinned revision onto the run-scoped volume. CPU only."""

    return runner.stage_model(volume_committer=model_volume.commit)


@app.function(
    image=_image("verify"),
    volumes=_volumes("verify"),
    **FUNCTION_SPEC_BY_KEY["verify"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def verify_stage() -> dict[str, Any]:
    """Verify the staged bytes and seal the prompt token arrays. CPU only."""

    return runner.verify_and_seal(volume_committer=model_volume.commit)


@app.function(
    image=_image("eager_canary"),
    volumes=_volumes("eager_canary"),
    **FUNCTION_SPEC_BY_KEY["eager_canary"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def eager_canary(experiment_nonce: str) -> dict[str, Any]:
    """Run the eager memory-gate canary on one L4."""

    return runner.run_canary(
        "eager", experiment_nonce=experiment_nonce, cache_root=CONTAINER_CACHE_ROOT
    )


@app.function(
    image=_image("compiled_canary"),
    volumes=_volumes("compiled_canary"),
    **FUNCTION_SPEC_BY_KEY["compiled_canary"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def compiled_canary(experiment_nonce: str) -> dict[str, Any]:
    """Run the compiled memory-gate canary on one L4."""

    return runner.run_canary(
        "compiled", experiment_nonce=experiment_nonce, cache_root=CONTAINER_CACHE_ROOT
    )


@app.function(
    image=_image("natural_cell"),
    volumes=_volumes("natural_cell"),
    **FUNCTION_SPEC_BY_KEY["natural_cell"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def natural_cell(cell_id: str, experiment_nonce: str) -> dict[str, Any]:
    """Execute one sealed natural-lane cell in a single-use container."""

    return runner.run_measured_cell(
        cell_id,
        experiment_nonce=experiment_nonce,
        cache_root=CONTAINER_CACHE_ROOT,
        output_root=CONTAINER_OUTPUT_ROOT,
    )


@app.function(
    image=_image("controlled_cell"),
    volumes=_volumes("controlled_cell"),
    **FUNCTION_SPEC_BY_KEY["controlled_cell"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def controlled_cell(cell_id: str, experiment_nonce: str) -> dict[str, Any]:
    """Execute one sealed controlled-lane cell in a single-use container."""

    return runner.run_measured_cell(
        cell_id,
        experiment_nonce=experiment_nonce,
        cache_root=CONTAINER_CACHE_ROOT,
        output_root=CONTAINER_OUTPUT_ROOT,
    )


@app.function(
    image=_image("analysis"),
    volumes=_volumes("analysis"),
    **FUNCTION_SPEC_BY_KEY["analysis"].modal_kwargs(),
)
@modal.concurrent(max_inputs=1)
def analysis(cell_ids: list[str]) -> dict[str, Any]:
    """Return a sanitized container-side inventory. CPU only.

    No statistic is computed here. The crossover analysis stays in the
    existing results core, which runs on the operator's machine over the
    receipts this run returned.
    """

    return runner.analysis_inventory(cell_ids)


FUNCTIONS = {
    "stage": stage_model,
    "verify": verify_stage,
    "eager_canary": eager_canary,
    "compiled_canary": compiled_canary,
    "natural_cell": natural_cell,
    "controlled_cell": controlled_cell,
    "analysis": analysis,
}

if set(FUNCTIONS) != set(FUNCTION_SPEC_BY_KEY):
    raise ModalL4ContractError("declared Functions differ from the sealed plan")
if getattr(app, "registered_web_endpoints", ()):
    raise ModalL4ContractError("this app must expose no web endpoint")

APP_TAGS = {
    "llmtracefx_protocol_id": PROTOCOL_ID,
    "llmtracefx_experiment_nonce": EXPERIMENT_NONCE,
    "llmtracefx_plan_sha256": PLAN.content_sha256,
}

__all__ = [
    "APP_TAGS",
    "FUNCTIONS",
    "NAMES",
    "NO_WEB_ENDPOINT_DECORATORS",
    "SDK_CAPABILITIES",
    "app",
    "model_volume",
]
