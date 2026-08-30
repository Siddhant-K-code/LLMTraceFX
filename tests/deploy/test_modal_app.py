"""The Modal entrypoint, imported against a fake SDK.

What this module decides at import is the whole safety story: how many
accelerators the serving function asks for, how long it may run, whether
it scales, and above all that the staging function asks for no
accelerator at all. Those decisions are only visible where the decorators
receive them, so the entrypoint is imported here with a stand-in ``modal``
module that records its arguments.

Nothing here authenticates, connects, or needs the real SDK installed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from _fakes import OTHER_REVISION, VALID_REVISION, imported_app, valid_environ

from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.manifest import (
    MANIFEST_SCHEMA_VERSION,
    StagedFile,
    VerificationIssue,
    WeightStagingManifest,
    WeightVerification,
)
from llmtracefx.deploy.resources import (
    SERVING_CPU_CORES,
    SERVING_MEMORY_GIB,
    STAGING_CPU_CORES,
    VERIFY_CPU_CORES,
)


def test_the_staging_function_is_never_given_an_accelerator() -> None:
    with imported_app(valid_environ()) as (_, fake):
        staging = fake._fake_apps[0].registrations["stage_weights"]
        assert "gpu" not in staging.function_kwargs
        assert staging.function_kwargs["cpu"] == 8.0
        assert staging.function_kwargs["retries"] == 0


def test_staging_and_serving_use_different_images() -> None:
    with imported_app(valid_environ()) as (module, fake):
        registrations = fake._fake_apps[0].registrations
        staging_image = registrations["stage_weights"].function_kwargs["image"]
        serving_image = registrations["serve"].function_kwargs["image"]
        assert staging_image is not serving_image
        assert staging_image.label == "debian_slim"
        assert serving_image.label == "from_registry"
        assert serving_image.kwargs["tag"] == module.RECIPE.image.reference


def test_the_serving_function_requests_exactly_the_planned_accelerators() -> None:
    with imported_app(valid_environ()) as (_, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["gpu"] == "H200:4"


def test_gpu_count_follows_the_environment() -> None:
    with imported_app(
        valid_environ(LLMTRACEFX_GLM_GPU_COUNT="8", LLMTRACEFX_GLM_MAX_USD="20.00")
    ) as (_, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["gpu"] == "H200:8"


def test_the_serving_function_carries_the_timeout_that_was_priced() -> None:
    with imported_app(valid_environ()) as (module, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["timeout"] == 1800
        assert (
            serve.function_kwargs["timeout"] == module.PLAN.envelope.max_runtime_seconds
        )


def test_serving_does_not_scale_or_stay_warm_by_default() -> None:
    with imported_app(valid_environ()) as (_, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["max_containers"] == 1
        assert serve.function_kwargs["min_containers"] == 0
        assert serve.function_kwargs["scaledown_window"] == 300
        assert serve.concurrent_kwargs == {"max_inputs": 1}


def test_the_web_server_port_and_startup_budget_match_the_recipe() -> None:
    with imported_app(valid_environ()) as (module, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.web_server_kwargs["port"] == module.RECIPE.port
        assert serve.web_server_kwargs["startup_timeout"] == 900


def test_the_secret_is_referenced_by_name_with_a_required_key() -> None:
    with imported_app(valid_environ()) as (module, fake):
        secret = fake._fake_secrets[0]
        assert secret.name == "glm-selfhost-api-key"
        assert secret.required_keys == ("GLM_SELFHOST_API_KEY",)
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["secrets"] == [secret]
        assert module.PLAN.endpoint.api_key_env_var == "GLM_SELFHOST_API_KEY"


def test_both_functions_mount_the_same_weights_volume() -> None:
    with imported_app(valid_environ()) as (module, fake):
        volume = fake._fake_volumes[0]
        assert volume.name == module.PLAN.volume_name
        assert volume.kwargs["create_if_missing"] is True
        registrations = fake._fake_apps[0].registrations
        for name in ("stage_weights", "serve"):
            mounts = registrations[name].function_kwargs["volumes"]
            assert mounts == {module.MOUNT_PATH: volume}


def test_the_app_is_named_from_the_plan() -> None:
    with imported_app(valid_environ(LLMTRACEFX_GLM_APP_NAME="custom-app")) as (
        module,
        fake,
    ):
        assert fake._fake_apps[0].name == "custom-app"
        assert module.PLAN.app_name == "custom-app"


def test_the_staging_client_is_pinned() -> None:
    with imported_app(valid_environ()) as (module, fake):
        staging_image = (
            fake._fake_apps[0].registrations["stage_weights"].function_kwargs["image"]
        )
        assert staging_image.pip_packages == [module.DEFAULT_HF_HUB_PIN]
        assert "==" in module.DEFAULT_HF_HUB_PIN


def test_the_serving_image_gets_the_local_package_for_manifest_building() -> None:
    with imported_app(valid_environ()) as (_, fake):
        serving_image = (
            fake._fake_apps[0].registrations["serve"].function_kwargs["image"]
        )
        assert serving_image.env_vars["PYTHONPATH"] == "/opt/llmtracefx"
        assert serving_image.local_dirs == [
            ("llmtracefx", "/opt/llmtracefx/llmtracefx")
        ]


def test_importing_without_the_spending_authority_fails_before_anything_registers() -> (
    None
):
    environ = valid_environ()
    del environ["LLMTRACEFX_GLM_MAX_USD"]
    with pytest.raises(DeploymentPlanError, match="LLMTRACEFX_GLM_MAX_USD"):
        with imported_app(environ):
            pass


def test_importing_an_over_budget_configuration_fails() -> None:
    environ = valid_environ(
        LLMTRACEFX_GLM_MAX_USD="0.01", LLMTRACEFX_GLM_USD_PER_GPU_HOUR="50"
    )
    with pytest.raises(DeploymentPlanError, match="exceeds the authorised budget"):
        with imported_app(environ):
            pass


def test_importing_with_an_unpinned_revision_fails() -> None:
    environ = valid_environ(LLMTRACEFX_GLM_MODEL_REVISION="main")
    with pytest.raises(DeploymentPlanError, match="commit SHA"):
        with imported_app(environ):
            pass


def test_the_module_exposes_the_adjudicated_plan_it_was_built_from() -> None:
    with imported_app(valid_environ()) as (module, _):
        assert module.PLAN.approved is True
        assert module.RECIPE.model_revision == VALID_REVISION
        assert module.MOUNT_PATH == "/weights"


def test_the_entrypoint_is_not_imported_by_the_package() -> None:
    """Importing the library must never pull in the Modal SDK."""
    import sys

    sys.modules.pop("llmtracefx.deploy.modal_glm_app", None)
    sys.modules.pop("modal", None)
    import importlib

    importlib.reload(importlib.import_module("llmtracefx.deploy"))
    assert "modal" not in sys.modules
    assert "llmtracefx.deploy.modal_glm_app" not in sys.modules


def _serve_harness(module: Any, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Neutralise everything in ``serve`` except the decisions under test.

    The manifest is made to match so the staging gate passes, hardware
    probes and the readiness thread are stubbed out because they shell
    out and poll a socket, and Popen is replaced by a recorder so the
    question "did this start a server" has a direct answer.
    """
    launched: list[list[str]] = []

    manifest = WeightStagingManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        completed_at="2026-08-30T00:00:00+00:00",
        repo_id=module.RECIPE.model_repo_id,
        revision=module.RECIPE.model_revision,
        mount_path=module.MOUNT_PATH,
        files=(StagedFile(path="config.json", size_bytes=1024),),
    )
    verification = WeightVerification(
        schema_version=MANIFEST_SCHEMA_VERSION,
        verified_at="2026-08-30T00:00:00+00:00",
        repo_id=module.RECIPE.model_repo_id,
        revision=module.RECIPE.model_revision,
        mount_path=module.MOUNT_PATH,
        files_checked=1,
        bytes_checked=1024,
        hashes_checked=0,
        hashes_available=0,
    )
    monkeypatch.setattr(module, "_read_manifest", lambda revision: manifest)
    monkeypatch.setattr(module, "_read_verification", lambda revision: verification)
    monkeypatch.setattr(module, "_observed_gpus", lambda: ("NVIDIA H200",))
    monkeypatch.setattr(module, "_observed_cuda_version", lambda: "12.9")
    monkeypatch.setattr(module.threading, "Thread", lambda **kwargs: _NoThread())
    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda argv, env=None: launched.append(list(argv)),
    )
    return launched


class _NoThread:
    def start(self) -> None:
        return None


def test_serve_refuses_to_start_without_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty key must be a refusal, not an unauthenticated endpoint.

    Neither framework installs authentication middleware without a
    non-empty key, and this endpoint has no platform auth in front of it,
    so serving anyway would publish the accelerators openly. Modal's
    `required_keys` does not cover it: it asserts presence, not content,
    and `modal secret create NAME VAR="$VAR"` with the export lost
    creates exactly this.
    """
    for value in ("", "   "):
        with imported_app(valid_environ(GLM_SELFHOST_API_KEY=value)) as (module, _):
            launched = _serve_harness(module, monkeypatch)
            with pytest.raises(RuntimeError, match="Refusing to start"):
                module.serve()
            assert launched == []


def test_serve_refuses_when_the_variable_is_absent_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with imported_app(valid_environ()) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        with pytest.raises(RuntimeError, match="GLM_SELFHOST_API_KEY"):
            module.serve()
        assert launched == []


def test_serve_refuses_when_the_weights_are_not_staged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY="k")) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        monkeypatch.setattr(module, "_read_manifest", lambda revision: None)
        with pytest.raises(RuntimeError, match="are not staged"):
            module.serve()
        assert launched == []


def test_serve_keeps_the_key_out_of_the_argv_on_the_default_framework(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-live-SERVE-SENTINEL-2f7c"
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY=secret)) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        module.serve()
        assert len(launched) == 1
        assert secret not in " ".join(launched[0])
        assert "--api-key" not in launched[0]
        # vLLM takes it from the environment, so nothing printed by this
        # module can carry it either.
        assert secret not in capsys.readouterr().out


def test_serve_logs_the_argv_before_sglang_gets_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The recorded argv must be credential free even for sglang."""
    secret = "sk-live-SERVE-SENTINEL-9a1b"
    environ = valid_environ(
        LLMTRACEFX_GLM_FRAMEWORK="sglang",
        LLMTRACEFX_GLM_ACCEPT_ARGV_CREDENTIAL_EXPOSURE="true",
        GLM_SELFHOST_API_KEY=secret,
    )
    with imported_app(environ) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        module.serve()
        printed = capsys.readouterr().out
        assert secret not in printed
        # The framework really does receive it; the guarantee is only
        # that this module did not write it down.
        assert launched[0][-2:] == ["--api-key", secret]


def test_a_remote_container_can_rebuild_the_plan_from_the_image_alone() -> None:
    """The decisive test for the remote import path.

    The entrypoint is imported twice: locally by `modal deploy`, where
    the operator has exported the configuration, and again inside every
    container, where nothing has. Baking the adjudicated values into both
    images is what makes the second import work. This simulates it by
    importing with *only* what the image sets plus the secret, which is
    exactly what a worker sees.
    """
    with imported_app(valid_environ()) as (module, fake):
        baked = dict(module.BAKED_ENVIRONMENT)
        local_plan = module.PLAN
        images = {
            registration.function_kwargs["image"]
            for registration in fake._fake_apps[0].registrations.values()
        }
        # Every image carries it, so no function can be scheduled onto a
        # container that lacks the configuration.
        for image in images:
            assert "LLMTRACEFX_GLM_MAX_USD" in image.env_vars

    # A worker's environment: the image env, plus the Secret, and none of
    # the operator's shell.
    remote_environ = {**baked, "GLM_SELFHOST_API_KEY": "secret-value"}
    assert not any(
        key.startswith("LLMTRACEFX_GLM_")
        for key in valid_environ()
        if key not in remote_environ
    )
    with imported_app(remote_environ) as (module, _):
        remote_plan = module.PLAN
        assert remote_plan.approved is True
        assert remote_plan.recipe.model_revision == local_plan.recipe.model_revision
        assert remote_plan.recipe.gpu_count == local_plan.recipe.gpu_count
        assert remote_plan.controls.timeout_seconds == (
            local_plan.controls.timeout_seconds
        )
        assert remote_plan.controls.deployment_seconds == (
            local_plan.controls.deployment_seconds
        )
        assert remote_plan.envelope.worst_case_usd == pytest.approx(
            local_plan.envelope.worst_case_usd
        )


def test_the_baked_configuration_carries_no_credential() -> None:
    secret = "sk-live-BAKED-SENTINEL-3c8d"
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY=secret)) as (module, _):
        baked = module.BAKED_ENVIRONMENT
        assert secret not in json.dumps(baked)
        # The name is there, so the container knows which variable to read.
        assert baked["LLMTRACEFX_GLM_API_KEY_ENV"] == "GLM_SELFHOST_API_KEY"


def test_an_expired_deployment_refuses_to_allocate_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expiry is what makes the priced window an upper bound.

    Modal's timeout bounds a Function's execution, and this one returns
    as soon as it has launched the server, so without an expiry a request
    arriving at any future time would allocate accelerators again.
    """
    past = "2020-01-01T00:00:00+00:00"
    environ = valid_environ(
        GLM_SELFHOST_API_KEY="k", LLMTRACEFX_GLM_DEPLOY_EXPIRES_AT=past
    )
    with imported_app(environ) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        with pytest.raises(RuntimeError, match="expired"):
            module.serve()
        assert launched == []


def test_an_unparseable_expiry_reads_as_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The safe reading of a corrupt expiry is 'do not start'."""
    environ = valid_environ(
        GLM_SELFHOST_API_KEY="k", LLMTRACEFX_GLM_DEPLOY_EXPIRES_AT="not-a-date"
    )
    with imported_app(environ) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        with pytest.raises(RuntimeError, match="expired"):
            module.serve()
        assert launched == []


def test_a_container_inherits_the_deploy_expiry_rather_than_extending_it() -> None:
    """A cold start must not restart the clock.

    If each container computed its own expiry from its own start time,
    the window would renew on every request and bound nothing.
    """
    future = "2099-01-01T00:00:00+00:00"
    with imported_app(valid_environ(LLMTRACEFX_GLM_DEPLOY_EXPIRES_AT=future)) as (
        module,
        _,
    ):
        assert module.DEPLOY_EXPIRES_AT == future
        assert module.BAKED_ENVIRONMENT["LLMTRACEFX_GLM_DEPLOY_EXPIRES_AT"] == future


def test_serving_requests_the_cpu_and_memory_the_envelope_prices() -> None:
    with imported_app(valid_environ()) as (_, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.function_kwargs["cpu"] == SERVING_CPU_CORES
        assert serve.function_kwargs["memory"] == int(SERVING_MEMORY_GIB * 1024)
        staging = fake._fake_apps[0].registrations["stage_weights"]
        assert staging.function_kwargs["cpu"] == STAGING_CPU_CORES
        verify = fake._fake_apps[0].registrations["verify_weights"]
        assert "gpu" not in verify.function_kwargs
        assert verify.function_kwargs["cpu"] == VERIFY_CPU_CORES


def test_serve_refuses_when_the_weights_were_never_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest alone must not be enough to reach the accelerators.

    The other serve tests stub a passing verification so they can reach
    the code past this gate, which means none of them exercises the gate
    itself. This one removes the record entirely.
    """
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY="k")) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        monkeypatch.setattr(module, "_read_verification", lambda revision: None)
        with pytest.raises(RuntimeError, match="have not passed verification"):
            module.serve()
        assert launched == []


def test_serve_refuses_a_verification_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record that exists but records issues must not satisfy the gate."""
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY="k")) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        broken = WeightVerification(
            schema_version=MANIFEST_SCHEMA_VERSION,
            verified_at="2026-08-30T00:00:00+00:00",
            repo_id=module.RECIPE.model_repo_id,
            revision=module.RECIPE.model_revision,
            mount_path=module.MOUNT_PATH,
            files_checked=1,
            bytes_checked=0,
            hashes_checked=0,
            hashes_available=0,
            issues=(VerificationIssue(path="shard.safetensors", problem="missing"),),
        )
        monkeypatch.setattr(module, "_read_verification", lambda revision: broken)
        with pytest.raises(RuntimeError, match="have not passed verification"):
            module.serve()
        assert launched == []


def test_serve_refuses_a_verification_for_another_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passing record for different weights must not be reused."""
    with imported_app(valid_environ(GLM_SELFHOST_API_KEY="k")) as (module, _):
        launched = _serve_harness(module, monkeypatch)
        foreign = WeightVerification(
            schema_version=MANIFEST_SCHEMA_VERSION,
            verified_at="2026-08-30T00:00:00+00:00",
            repo_id=module.RECIPE.model_repo_id,
            revision=OTHER_REVISION,
            mount_path=module.MOUNT_PATH,
            files_checked=1,
            bytes_checked=1024,
            hashes_checked=0,
            hashes_available=0,
        )
        monkeypatch.setattr(module, "_read_verification", lambda revision: foreign)
        with pytest.raises(RuntimeError, match="have not passed verification"):
            module.serve()
        assert launched == []


def test_unauthenticated_requests_cannot_schedule_a_gpu_container() -> None:
    """The decisive control, asserted where the platform reads it.

    Everything this module checks runs after Modal has already scheduled
    the container, so a Python-side refusal bounds how long a container
    serves, not whether one starts. Proxy auth is enforced at Modal's
    edge, which rejects a request without credentials with a 401 before
    scheduling anything, so this decorator argument is the only thing
    standing between a public request and an allocated accelerator.
    """
    with imported_app(valid_environ()) as (_, fake):
        serve = fake._fake_apps[0].registrations["serve"]
        assert serve.web_server_kwargs["requires_proxy_auth"] is True
        # And it really is the GPU function that carries it.
        assert serve.function_kwargs["gpu"].startswith("H200:")


def test_a_public_endpoint_cannot_be_configured_at_all() -> None:
    """There is deliberately no flag that turns the edge off.

    A public endpoint has no cost bound: any request from anyone
    allocates accelerators before any refusal can run. Rather than offer
    an override, the harness refuses the configuration outright.
    """
    from llmtracefx.deploy.plan import RuntimeControls

    with pytest.raises(DeploymentPlanError, match="cannot be turned off"):
        RuntimeControls(
            timeout_seconds=1800,
            deployment_seconds=1800,
            require_proxy_auth=False,
        )


def test_the_envelope_states_what_it_does_not_cover() -> None:
    with imported_app(valid_environ()) as (module, _):
        payload = module.PLAN.envelope.to_dict()
        assert payload["bounded"] is True
        assert "scheduled before any code" in payload["does_not_cover"]
