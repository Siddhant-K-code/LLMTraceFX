"""Pinning rules: which model, which revision, which image, which GPUs."""

from __future__ import annotations

import pytest
from _fakes import (
    OTHER_REVISION,
    PINNED_IMAGE,
    TAG_ONLY_IMAGE,
    VALID_DIGEST,
    VALID_REVISION,
)

from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.recipe import (
    GLM_53_FLASH,
    MIN_VRAM_HEADROOM_FRACTION,
    SGLANG,
    SUPPORTED_REPO_ID,
    VLLM,
    build_recipe,
    check_memory_fit,
    parse_image_reference,
    require_model_revision,
    require_supported_repo,
)


def recipe(**overrides: object):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "framework": SGLANG,
        "framework_version": "0.5.6",
        "image_reference": PINNED_IMAGE,
        "model_revision": VALID_REVISION,
        "gpu_type": "H200",
        "gpu_count": 4,
        "context_length": 131072,
        "weights_mount_path": "/weights",
        "port": 30000,
    }
    kwargs.update(overrides)
    return build_recipe(**kwargs)  # type: ignore[arg-type]


def test_published_architecture_facts_are_carried_verbatim() -> None:
    assert GLM_53_FLASH.repo_id == SUPPORTED_REPO_ID
    assert GLM_53_FLASH.num_hidden_layers == 45
    assert GLM_53_FLASH.max_position_embeddings == 1_048_576
    assert GLM_53_FLASH.quantization == "fp8"
    assert GLM_53_FLASH.quantization_format == "e4m3"
    assert GLM_53_FLASH.multimodal is True
    assert GLM_53_FLASH.config_source.startswith("https://huggingface.co/")


@pytest.mark.parametrize(
    "bad",
    ["main", "v1.0", VALID_REVISION[:12], VALID_REVISION.upper() + "X", "", "  "],
)
def test_only_a_full_commit_sha_counts_as_a_revision(bad: str) -> None:
    with pytest.raises(DeploymentPlanError, match="40-character commit SHA"):
        require_model_revision(bad)


def test_revision_is_normalised_to_lower_case() -> None:
    assert require_model_revision(VALID_REVISION.upper()) == VALID_REVISION


def test_the_bf16_and_full_variants_are_refused_by_name() -> None:
    with pytest.raises(DeploymentPlanError, match="twice the size"):
        require_supported_repo("zai-org/GLM-5.3-Flash-BF16")
    with pytest.raises(DeploymentPlanError, match="704 GiB"):
        require_supported_repo("zai-org/GLM-5.3")
    with pytest.raises(DeploymentPlanError, match="serves .* only"):
        require_supported_repo("meta-llama/Llama-3-70B")


def test_latest_is_refused_even_with_the_override() -> None:
    with pytest.raises(DeploymentPlanError, match="never reproducible"):
        parse_image_reference("lmsysorg/sglang:latest", accept_mutable=True)


def test_tag_only_image_needs_an_explicit_acceptance() -> None:
    with pytest.raises(DeploymentPlanError, match="accept-mutable-image"):
        parse_image_reference(TAG_ONLY_IMAGE)
    accepted = parse_image_reference(TAG_ONLY_IMAGE, accept_mutable=True)
    assert accepted.is_digest_pinned is False
    assert accepted.tag == "v0.5.6"


def test_digest_pinned_image_is_parsed_into_its_parts() -> None:
    reference = parse_image_reference(PINNED_IMAGE)
    assert reference.name == "lmsysorg/sglang"
    assert reference.tag == "v0.5.6"
    assert reference.digest is not None
    assert reference.is_digest_pinned is True
    assert reference.to_dict()["digest_pinned"] is True


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "image@sha256:tooshort", "image:tag@md5:" + "a" * 32],
)
def test_malformed_image_references_are_refused(bad: str) -> None:
    with pytest.raises(DeploymentPlanError):
        parse_image_reference(bad, accept_mutable=True)


def test_a_registry_port_is_not_mistaken_for_a_tag() -> None:
    """``registry:5000/org/img`` has a colon and no tag.

    Reading the port as a version would accept an untagged reference as
    though it were pinned. Refusing the whole reference, which a naive
    single pattern does, locks out every private registry.
    """
    reference = parse_image_reference(
        f"registry.internal:5000/lmsysorg/sglang:v0.5.6@{VALID_DIGEST}"
    )
    assert reference.name == "registry.internal:5000/lmsysorg/sglang"
    assert reference.tag == "v0.5.6"
    assert reference.digest == VALID_DIGEST

    with pytest.raises(DeploymentPlanError, match="neither a tag nor a digest"):
        parse_image_reference("registry.internal:5000/lmsysorg/sglang")


def test_a_digest_only_reference_is_pinned_without_a_tag() -> None:
    reference = parse_image_reference(f"lmsysorg/sglang@{VALID_DIGEST}")
    assert reference.tag is None
    assert reference.is_digest_pinned is True


def test_whitespace_inside_a_reference_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="malformed"):
        parse_image_reference(f"lmsysorg/sglang:v1 --privileged@{VALID_DIGEST}")


def test_four_h200_clears_the_headroom_requirement_for_the_fp8_checkpoint() -> None:
    fit = check_memory_fit(gpu_type="H200", gpu_count=4)
    assert fit.total_vram_gib == pytest.approx(564.0)
    assert fit.fits is True
    assert fit.residual_fraction > MIN_VRAM_HEADROOM_FRACTION
    assert "does not prove" in fit.caveat


def test_two_h200_cannot_hold_the_checkpoint() -> None:
    fit = check_memory_fit(gpu_type="H200", gpu_count=2)
    assert fit.total_vram_gib == pytest.approx(282.0)
    assert fit.residual_gib < 0
    assert fit.fits is False


def test_enough_room_for_weights_but_not_for_kv_cache_still_fails() -> None:
    # 320 GiB total against roughly 306 GiB of weights: the weights fit
    # and nothing else does.
    fit = check_memory_fit(gpu_type="H200", gpu_count=1, vram_gib_per_gpu=320.0)
    assert fit.residual_gib > 0
    assert fit.fits is False


def test_unknown_gpu_requires_an_explicit_capacity() -> None:
    with pytest.raises(DeploymentPlanError, match="unknown GPU type"):
        check_memory_fit(gpu_type="TPUv5", gpu_count=4)
    fit = check_memory_fit(gpu_type="TPUv5", gpu_count=4, vram_gib_per_gpu=200.0)
    assert fit.fits is True


def test_tensor_parallel_defaults_to_the_gpu_count() -> None:
    assert recipe().tensor_parallel_size == 4


def test_tensor_parallel_cannot_exceed_the_allocated_gpus() -> None:
    with pytest.raises(DeploymentPlanError, match="exceeds gpu_count"):
        recipe(tensor_parallel_size=8)


def test_context_length_cannot_exceed_the_model_maximum() -> None:
    with pytest.raises(DeploymentPlanError, match="max_position_embeddings"):
        recipe(context_length=GLM_53_FLASH.max_position_embeddings + 1)


def test_framework_version_must_be_stated() -> None:
    with pytest.raises(DeploymentPlanError, match="framework_version"):
        recipe(framework_version="  ")


def test_unsupported_framework_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="framework must be one of"):
        recipe(framework="tensorrt")


def test_server_is_pointed_at_the_volume_not_at_the_repository_id() -> None:
    built = recipe()
    argv = built.launch_argv()
    assert built.local_model_path() == f"/weights/{VALID_REVISION}"
    assert built.local_model_path() in argv
    assert SUPPORTED_REPO_ID not in argv[: argv.index("--served-model-name")]


def test_sglang_argv_carries_parallelism_and_context_cap() -> None:
    argv = recipe().launch_argv()
    assert argv[:3] == ("python3", "-m", "sglang.launch_server")
    assert "--tp" in argv and argv[argv.index("--tp") + 1] == "4"
    assert argv[argv.index("--context-length") + 1] == "131072"
    assert argv[argv.index("--port") + 1] == "30000"


def test_vllm_argv_uses_its_own_flag_spellings() -> None:
    argv = recipe(framework=VLLM).launch_argv()
    assert argv[:2] == ("vllm", "serve")
    assert argv[argv.index("--tensor-parallel-size") + 1] == "4"
    assert argv[argv.index("--max-model-len") + 1] == "131072"


def test_launch_argv_never_contains_a_credential_flag() -> None:
    for framework in (SGLANG, VLLM):
        argv = recipe(framework=framework).launch_argv()
        joined = " ".join(argv).casefold()
        assert "--api-key" not in joined
        assert "token" not in joined


def test_recipe_serialises_the_pin_and_the_revision() -> None:
    payload = recipe(model_revision=OTHER_REVISION).to_dict()
    assert payload["model_revision"] == OTHER_REVISION
    assert payload["image"]["digest_pinned"] is True
    assert payload["model_facts"]["quantization"] == "fp8"


def test_relative_mount_path_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="absolute path"):
        recipe(weights_mount_path="weights")
