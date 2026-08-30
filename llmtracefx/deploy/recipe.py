"""The pinned GLM-5.3-Flash serving recipe.

Scope is deliberately one model. GLM-5.3-Flash is the 320B-A18B FP8
checkpoint; the full GLM-5.3 is roughly twice the size and is out of
scope for a credit-funded validation, as is the BF16 variant of Flash.
Both are rejected by name rather than left to fail later against a VRAM
limit, because by then the GPUs are already allocated and billing.

What is pinned here and what is not is a deliberate split:

* Architecture facts are compiled in. They were read from the published
  ``config.json`` and they describe the checkpoint, so they cannot drift
  without the checkpoint itself changing, which is what the revision pin
  detects.
* The model revision, the container digest and every price are *not*
  compiled in. They are mutable, and a stale literal for any of them is
  a silently wrong answer rather than a loud one. Each has to be supplied
  by the caller and is validated for shape before it is used.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import DeploymentPlanError

MODEL_CARD_URL = "https://huggingface.co/zai-org/GLM-5.3-Flash"
MODEL_CONFIG_URL = "https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json"
UPSTREAM_REPO_URL = "https://github.com/zai-org/GLM-5"

SUPPORTED_REPO_ID = "zai-org/GLM-5.3-Flash"

# Repositories refused by name, with the reason the operator needs.
_REFUSED_REPO_IDS: dict[str, str] = {
    "zai-org/GLM-5.3-Flash-BF16": (
        "the BF16 variant is roughly twice the size of the FP8 checkpoint "
        "this harness is scoped to and does not fit the budget it is "
        "designed for"
    ),
    "zai-org/GLM-5.3": (
        "full GLM-5.3 is roughly 704 GiB at FP8 and is out of scope for "
        "this harness, which targets GLM-5.3-Flash only"
    ),
}

_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_IMAGE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*$")

SGLANG = "sglang"
VLLM = "vllm"
SUPPORTED_FRAMEWORKS = (SGLANG, VLLM)

# How each framework accepts the endpoint credential.
#
# This is the deciding difference between them for this harness, and it
# is why vLLM is the default. vLLM reads ``VLLM_API_KEY`` from the
# environment (vllm/envs.py, and the auth middleware in
# vllm/entrypoints/serve/middleware/register.py, read 2026-08-30), so the
# value never appears on a command line. SGLang accepts the key only as
# ``--api-key`` on argv, and its engine logs its own resolved
# configuration with ``logger.info(f"server_args={server_args.resolved_dict()}")``
# (python/sglang/srt/entrypoints/engine.py, read 2026-08-30) with no
# redaction, so the key lands in the container's stdout and from there in
# the platform's log store, as well as being readable from
# /proc/<pid>/cmdline.
CREDENTIAL_TRANSPORT_ENVIRONMENT = "environment"
CREDENTIAL_TRANSPORT_ARGV = "argv"

_CREDENTIAL_TRANSPORTS: dict[str, str] = {
    VLLM: CREDENTIAL_TRANSPORT_ENVIRONMENT,
    SGLANG: CREDENTIAL_TRANSPORT_ARGV,
}

VLLM_API_KEY_ENV_VAR = "VLLM_API_KEY"

DEFAULT_FRAMEWORK = VLLM

# Where the staged weights are mounted inside every container, and the
# port the OpenAI-compatible server binds. Both live here rather than in
# the CLI so the Modal app can read them without importing an argparse
# module.
DEFAULT_WEIGHTS_MOUNT_PATH = "/weights"
DEFAULT_SERVER_PORT = 30000

# Nameplate capacity per accelerator, in GiB, as advertised on Modal's
# GPU guide (https://modal.com/docs/guide/gpu, read 2026-08-30). This is
# advertised capacity, not usable capacity: the runtime, the CUDA
# context and fragmentation all take a share before any weight is
# loaded, which is why the fit check below demands headroom rather than
# merely arithmetic sufficiency.
GPU_VRAM_GIB: dict[str, float] = {
    "H100": 80.0,
    "H200": 141.0,
    "B200": 180.0,
}

# Fraction of total VRAM that must remain after the weights are resident.
# KV cache, activations and the CUDA context all come out of what is
# left, and a plan that fits the weights but nothing else produces a
# server that starts and then fails on the first long request, having
# billed for the whole startup.
MIN_VRAM_HEADROOM_FRACTION = 0.20


@dataclass(frozen=True)
class ModelFacts:
    """Architecture facts read from the published model configuration.

    ``approximate_checkpoint_gib`` is the one soft number here. It is
    used to size the volume and to sanity-check the GPU fit, never to
    make a claim about what was downloaded: the staging manifest records
    the byte counts actually observed on the volume.
    """

    repo_id: str
    total_parameters_b: float
    active_parameters_b: float
    num_hidden_layers: int
    num_routed_experts: int
    num_experts_per_token: int
    max_position_embeddings: int
    quantization: str
    quantization_format: str
    activation_scheme: str
    multimodal: bool
    attention: str
    approximate_checkpoint_gib: float
    config_source: str
    model_card: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GLM_53_FLASH = ModelFacts(
    repo_id=SUPPORTED_REPO_ID,
    total_parameters_b=320.0,
    active_parameters_b=18.0,
    num_hidden_layers=45,
    num_routed_experts=288,
    num_experts_per_token=8,
    max_position_embeddings=1_048_576,
    quantization="fp8",
    quantization_format="e4m3",
    activation_scheme="dynamic",
    multimodal=True,
    attention="hybrid linear attention with periodic sparse attention layers",
    approximate_checkpoint_gib=306.0,
    config_source=MODEL_CONFIG_URL,
    model_card=MODEL_CARD_URL,
)


def require_model_revision(revision: str) -> str:
    """Accept only a full commit SHA as a model revision.

    A branch name is not a revision. ``main`` resolves to whatever was
    pushed most recently, so staging weights at ``main`` produces an
    artifact whose contents cannot be reconstructed later and a manifest
    that records a name rather than a fact. A short SHA is refused for
    the same reason a short hash is refused anywhere else: it is a prefix
    that is not guaranteed to stay unique.
    """
    if not isinstance(revision, str):
        raise DeploymentPlanError("model revision must be a string")
    candidate = revision.strip().lower()
    if not _GIT_SHA_PATTERN.match(candidate):
        raise DeploymentPlanError(
            "model revision must be a full 40-character commit SHA, not "
            f"{revision!r}. Resolve it once and pin it, for example: "
            "curl -s https://huggingface.co/api/models/"
            f"{SUPPORTED_REPO_ID} | python -c "
            "\"import json,sys; print(json.load(sys.stdin)['sha'])\""
        )
    return candidate


def require_supported_repo(repo_id: str) -> str:
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise DeploymentPlanError("repo_id must be a non-empty string")
    candidate = repo_id.strip()
    refusal = _REFUSED_REPO_IDS.get(candidate)
    if refusal is not None:
        raise DeploymentPlanError(f"{candidate} is not supported here: {refusal}")
    if candidate != SUPPORTED_REPO_ID:
        raise DeploymentPlanError(
            f"this harness serves {SUPPORTED_REPO_ID} only, got {candidate!r}"
        )
    return candidate


@dataclass(frozen=True)
class ImageReference:
    """A container image reference, and whether it is actually pinned."""

    reference: str
    name: str
    tag: str | None
    digest: str | None

    @property
    def is_digest_pinned(self) -> bool:
        return self.digest is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "name": self.name,
            "tag": self.tag,
            "digest": self.digest,
            "digest_pinned": self.is_digest_pinned,
        }


def _split_image_reference(candidate: str) -> tuple[str, str | None, str | None]:
    """Split a reference into name, tag and digest, parsing right to left.

    A single regex cannot do this correctly, because ``:`` is overloaded:
    it separates a tag, and it also separates a registry host from its
    port. ``registry.internal:5000/org/img`` has a colon and no tag at
    all. The rule the OCI spec encodes, and the one applied here, is that
    a colon introduces a tag only when no ``/`` follows it; otherwise it
    belongs to the registry host.

    Getting this wrong in the lenient direction would misread a port as a
    version. Getting it wrong in the strict direction, as a naive pattern
    does, refuses every private registry outright.
    """
    remainder, separator, digest_part = candidate.rpartition("@")
    if separator:
        digest: str | None = digest_part
        if not remainder:
            raise DeploymentPlanError(f"malformed image reference: {candidate!r}")
    else:
        digest = None
        remainder = digest_part

    name, colon, tail = remainder.rpartition(":")
    if colon and "/" not in tail:
        tag: str | None = tail
    else:
        name, tag = remainder, None

    if not name:
        raise DeploymentPlanError(f"malformed image reference: {candidate!r}")
    return name, tag, digest


def parse_image_reference(
    reference: str, *, accept_mutable: bool = False
) -> ImageReference:
    """Parse and vet a registry reference.

    ``latest`` is refused outright and with no override, because it is
    not a version at all: two deploys a week apart can differ with
    nothing in the record to show it. A concrete tag without a digest is
    still mutable, since a tag can be repointed, so it is refused too,
    but that refusal has an override for the case where the operator
    knowingly accepts the risk and the plan then records that they did.

    The published GLM-5.3-Flash model card itself shows
    ``lmsysorg/sglang:latest``, so this check is expected to reject a
    copy-pasted upstream command. That is the intended behaviour, not an
    incompatibility: reproducibility is the whole point of the pin.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise DeploymentPlanError("image reference must be a non-empty string")
    candidate = reference.strip()
    if any(character.isspace() for character in candidate):
        raise DeploymentPlanError(f"malformed image reference: {reference!r}")

    name, tag, digest = _split_image_reference(candidate)

    if not _IMAGE_NAME_PATTERN.match(name):
        raise DeploymentPlanError(f"malformed image name in {reference!r}")
    if tag is not None and not _IMAGE_TAG_PATTERN.match(tag):
        raise DeploymentPlanError(f"malformed image tag in {reference!r}")
    if digest is not None and not _DIGEST_PATTERN.match(digest):
        raise DeploymentPlanError(f"malformed image digest in {reference!r}")

    if tag == "latest":
        raise DeploymentPlanError(
            "image tag 'latest' is never reproducible; pin a concrete tag "
            "and, preferably, an @sha256: digest"
        )
    if tag is None and digest is None:
        raise DeploymentPlanError(
            f"image reference {reference!r} has neither a tag nor a digest"
        )
    if digest is None and not accept_mutable:
        raise DeploymentPlanError(
            f"image reference {reference!r} is tag-only and a tag can be "
            "repointed. Supply an @sha256: digest, or pass "
            "--accept-mutable-image to record that you accepted a mutable "
            "image on purpose"
        )
    return ImageReference(reference=candidate, name=name, tag=tag, digest=digest)


@dataclass(frozen=True)
class MemoryFit:
    """Whether the requested accelerators can hold the checkpoint."""

    gpu_type: str
    gpu_count: int
    vram_gib_per_gpu: float
    total_vram_gib: float
    weights_gib: float
    residual_gib: float
    residual_fraction: float
    required_headroom_fraction: float
    fits: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "vram_gib_per_gpu": self.vram_gib_per_gpu,
            "total_vram_gib": round(self.total_vram_gib, 2),
            "weights_gib": round(self.weights_gib, 2),
            "residual_gib": round(self.residual_gib, 2),
            "residual_fraction": round(self.residual_fraction, 4),
            "required_headroom_fraction": self.required_headroom_fraction,
            "fits": self.fits,
            "caveat": self.caveat,
        }


def check_memory_fit(
    *,
    gpu_type: str,
    gpu_count: int,
    facts: ModelFacts = GLM_53_FLASH,
    vram_gib_per_gpu: float | None = None,
) -> MemoryFit:
    """A necessary, and explicitly not sufficient, capacity check.

    No official hardware recommendation for GLM-5.3-Flash was found on
    the model card or in the upstream repository, so this is arithmetic
    over the published checkpoint size and the advertised VRAM, not a
    restatement of vendor guidance. It can prove that a configuration
    cannot work. It cannot prove that one will, which is what the caveat
    on the result says and what the smoke test exists to settle.
    """
    resolved_type = gpu_type.strip()
    if vram_gib_per_gpu is None:
        known = GPU_VRAM_GIB.get(resolved_type.upper())
        if known is None:
            raise DeploymentPlanError(
                f"unknown GPU type {gpu_type!r}; pass an explicit "
                "vram_gib_per_gpu or use one of: " + ", ".join(sorted(GPU_VRAM_GIB))
            )
        vram = known
    else:
        vram = float(vram_gib_per_gpu)
        if vram <= 0:
            raise DeploymentPlanError("vram_gib_per_gpu must be greater than zero")

    total = vram * gpu_count
    weights = facts.approximate_checkpoint_gib
    residual = total - weights
    fraction = residual / total if total > 0 else 0.0
    return MemoryFit(
        gpu_type=resolved_type,
        gpu_count=gpu_count,
        vram_gib_per_gpu=vram,
        total_vram_gib=total,
        weights_gib=weights,
        residual_gib=residual,
        residual_fraction=fraction,
        required_headroom_fraction=MIN_VRAM_HEADROOM_FRACTION,
        fits=residual > 0 and fraction >= MIN_VRAM_HEADROOM_FRACTION,
        caveat=(
            "Advertised VRAM against an approximate checkpoint size. "
            "Passing this check does not prove the model serves at the "
            "requested context length; no official hardware requirement "
            "for GLM-5.3-Flash was published on the model card or in "
            f"{UPSTREAM_REPO_URL} as of 2026-08-30."
        ),
    )


@dataclass(frozen=True)
class ServingRecipe:
    """A fully resolved, reproducible description of how to serve.

    Holds no credential and no price. It is the "what runs" half of a
    plan; the "what may it cost" half lives in the cost envelope.
    """

    framework: str
    framework_version: str
    image: ImageReference
    model_repo_id: str
    model_revision: str
    facts: ModelFacts
    gpu_type: str
    gpu_count: int
    tensor_parallel_size: int
    context_length: int
    served_model_name: str
    port: int
    weights_mount_path: str
    extra_server_args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def credential_transport(self) -> str:
        """How this framework will receive the endpoint key.

        Recorded and adjudicated rather than assumed, because the two
        supported frameworks differ in a way that matters: one takes the
        key from the environment and one takes it on its command line and
        then logs its own configuration.
        """
        return _CREDENTIAL_TRANSPORTS[self.framework]

    @property
    def exposes_credential_on_argv(self) -> bool:
        return self.credential_transport == CREDENTIAL_TRANSPORT_ARGV

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "credential_transport": self.credential_transport,
            "image": self.image.to_dict(),
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "model_facts": self.facts.to_dict(),
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "tensor_parallel_size": self.tensor_parallel_size,
            "context_length": self.context_length,
            "served_model_name": self.served_model_name,
            "port": self.port,
            "weights_mount_path": self.weights_mount_path,
            "extra_server_args": list(self.extra_server_args),
        }

    def local_model_path(self) -> str:
        """Where the staged weights are read from inside the container.

        The server is always pointed at the volume, never at the
        repository id. Passing the repository id would let the serving
        container fall back to downloading over the network at start up,
        on GPUs, which is precisely the expensive mistake the separate
        CPU staging step exists to prevent.
        """
        return f"{self.weights_mount_path.rstrip('/')}/{self.model_revision}"

    def launch_argv(self) -> tuple[str, ...]:
        """The exact server process argv, as a tuple, never a shell string."""
        if self.framework == SGLANG:
            return (
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                self.local_model_path(),
                "--served-model-name",
                self.served_model_name,
                "--host",
                "0.0.0.0",
                "--port",
                str(self.port),
                "--tp",
                str(self.tensor_parallel_size),
                "--context-length",
                str(self.context_length),
                *self.extra_server_args,
            )
        return (
            "vllm",
            "serve",
            self.local_model_path(),
            "--served-model-name",
            self.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--max-model-len",
            str(self.context_length),
            *self.extra_server_args,
        )


def build_recipe(
    *,
    framework: str,
    framework_version: str,
    image_reference: str,
    model_revision: str,
    gpu_type: str,
    gpu_count: int,
    context_length: int,
    weights_mount_path: str,
    port: int,
    repo_id: str = SUPPORTED_REPO_ID,
    served_model_name: str = SUPPORTED_REPO_ID,
    tensor_parallel_size: int | None = None,
    extra_server_args: tuple[str, ...] = (),
    accept_mutable_image: bool = False,
    facts: ModelFacts = GLM_53_FLASH,
) -> ServingRecipe:
    """Validate every field and assemble a recipe, or refuse.

    ``tensor_parallel_size`` defaults to ``gpu_count`` because a
    tensor-parallel degree below the number of allocated GPUs leaves
    accelerators idle and billing, which is a costly default to get
    wrong silently.
    """
    if framework not in SUPPORTED_FRAMEWORKS:
        raise DeploymentPlanError(
            f"framework must be one of {', '.join(SUPPORTED_FRAMEWORKS)}, "
            f"got {framework!r}"
        )
    if not isinstance(framework_version, str) or not framework_version.strip():
        raise DeploymentPlanError(
            "framework_version must be stated so the record says which "
            "build served the request"
        )
    if context_length < 1:
        raise DeploymentPlanError("context_length must be at least 1")
    if context_length > facts.max_position_embeddings:
        raise DeploymentPlanError(
            f"context_length {context_length} exceeds the model's "
            f"max_position_embeddings {facts.max_position_embeddings}"
        )
    if not (1 <= port <= 65535):
        raise DeploymentPlanError(f"port must be in 1..65535, got {port}")
    if not weights_mount_path.startswith("/"):
        raise DeploymentPlanError("weights_mount_path must be an absolute path")

    parallel = gpu_count if tensor_parallel_size is None else tensor_parallel_size
    if parallel < 1:
        raise DeploymentPlanError("tensor_parallel_size must be at least 1")
    if parallel > gpu_count:
        raise DeploymentPlanError(
            f"tensor_parallel_size {parallel} exceeds gpu_count {gpu_count}"
        )

    return ServingRecipe(
        framework=framework,
        framework_version=framework_version.strip(),
        image=parse_image_reference(
            image_reference, accept_mutable=accept_mutable_image
        ),
        model_repo_id=require_supported_repo(repo_id),
        model_revision=require_model_revision(model_revision),
        facts=facts,
        gpu_type=gpu_type.strip(),
        gpu_count=gpu_count,
        tensor_parallel_size=parallel,
        context_length=context_length,
        served_model_name=served_model_name,
        port=port,
        weights_mount_path=weights_mount_path,
        extra_server_args=tuple(extra_server_args),
    )
