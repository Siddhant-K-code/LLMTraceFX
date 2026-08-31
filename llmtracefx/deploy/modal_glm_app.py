"""Modal entrypoint for a budget-guarded GLM-5.3-Flash self-host.

This module is run by ``modal run`` and ``modal deploy``. It is not a
library: nothing else in the package imports it, and the tests never do,
which is what allows the entire planning and budget contract to be tested
with no Modal SDK installed.

Three things happen at import, in this order, and the order matters:

1. The Modal SDK is imported, with a message that says how to install it
   rather than a bare ``ModuleNotFoundError``.
2. The plan is rebuilt from the environment. If any spending authority,
   price or pin is missing, or if the resulting plan is refused, this
   raises and ``modal deploy`` fails before registering anything. There
   is no configuration under which importing this module quietly yields
   a deployable app with default GPU settings.
3. Only then are the app, the volume and the images declared, every one
   of them parameterised from the adjudicated plan rather than from a
   literal written here.

Staging and serving are separate functions with separate images and
separate hardware. ``stage_weights`` has no ``gpu=`` argument at all, so
there is no code path on which several hundred GiB are transferred while
accelerators are allocated and billing.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only on Modal
    raise SystemExit(
        "The Modal SDK is required to run this entrypoint. It is an optional "
        "dependency of llmtracefx: install it with `uv sync --extra modal` or "
        "`pip install 'llmtracefx[modal]'`. Planning does not need it: "
        "`llmtracefx-deploy plan` runs without Modal installed."
    ) from exc

from llmtracefx.deploy.environment import plan_environment, plan_from_environ
from llmtracefx.deploy.manifest import (
    MANIFEST_SCHEMA_VERSION,
    VERIFICATION_FILENAME,
    ServerManifest,
    StagedFile,
    WeightStagingManifest,
    WeightVerification,
    present_env_var_names,
    verify_staged_weights,
)
from llmtracefx.deploy.recipe import (
    SGLANG,
    VLLM_API_KEY_ENV_VAR,
    require_model_revision,
    require_supported_repo,
)
from llmtracefx.deploy.resources import (
    SERVING_CPU_CORES,
    SERVING_MEMORY_GIB,
    STAGING_CPU_CORES,
    STAGING_MEMORY_GIB,
    STAGING_TIMEOUT_SECONDS,
    VERIFY_CPU_CORES,
    VERIFY_MEMORY_GIB,
    VERIFY_TIMEOUT_SECONDS,
)

# Resolved from PyPI on 2026-08-30. Pinned rather than floating because
# the staging step's whole job is to produce a reproducible artifact, and
# a resolver that picks a different client on a later run is one more way
# for two stagings of the same revision to differ.
DEFAULT_HF_HUB_PIN = "huggingface_hub==1.29.0"

MANIFEST_FILENAME = "manifest.json"

# Set on both images so a container rebuilds the same plan the local
# deploy approved. Without it the remote import would look for eleven
# variables that only ever existed in the operator's shell.
DEPLOY_EXPIRES_AT_VAR = "LLMTRACEFX_GLM_DEPLOY_EXPIRES_AT"

PLAN = plan_from_environ(os.environ)
RECIPE = PLAN.recipe
CONTROLS = PLAN.controls
MOUNT_PATH = RECIPE.weights_mount_path

# The wall-clock instant after which no container may start serving.
#
# This is what turns the priced window into an enforced one. Modal's
# timeout bounds a Function's execution, and this server's function
# returns as soon as it has launched the framework, so nothing in the
# platform stops requests from starting fresh containers indefinitely.
# An expiry does: once it passes, every cold start refuses, and the
# accelerators cannot be allocated again by anyone, authenticated or not.
#
# Taken from the environment when already present so that a container
# inherits the deploy-time value rather than extending its own window on
# every cold start.
DEPLOY_EXPIRES_AT = (os.environ.get(DEPLOY_EXPIRES_AT_VAR) or "").strip() or (
    datetime.now(timezone.utc) + timedelta(seconds=CONTROLS.deployment_seconds)
).isoformat(timespec="seconds")

# Everything a remote container needs to rebuild this exact plan. No
# credential appears here: names, prices, revisions and limits only.
BAKED_ENVIRONMENT = {
    **plan_environment(PLAN, as_of=date.today()),
    DEPLOY_EXPIRES_AT_VAR: DEPLOY_EXPIRES_AT,
}

app = modal.App(PLAN.app_name)

weights_volume = modal.Volume.from_name(PLAN.volume_name, create_if_missing=True)

# The serving container receives the endpoint key through a Modal Secret.
# ``required_keys`` makes Modal assert the variable exists at deploy
# time, so a misnamed secret fails during deploy instead of producing a
# server that starts on four accelerators and rejects every request.
endpoint_secret = modal.Secret.from_name(
    PLAN.endpoint.modal_secret_name,
    required_keys=[PLAN.endpoint.api_key_env_var],
)

# Optional: a Hugging Face token, needed only if the repository is gated.
_HF_SECRET_NAME = (os.environ.get("LLMTRACEFX_GLM_HF_SECRET_NAME") or "").strip()
_staging_secrets = [modal.Secret.from_name(_HF_SECRET_NAME)] if _HF_SECRET_NAME else []

staging_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(os.environ.get("LLMTRACEFX_GLM_HF_HUB_PIN") or DEFAULT_HF_HUB_PIN)
    .env({**BAKED_ENVIRONMENT, "PYTHONPATH": "/opt/llmtracefx"})
    .add_local_dir("llmtracefx", remote_path="/opt/llmtracefx/llmtracefx")
)

# The serving image is whatever the plan pinned. It already contains the
# framework and its CUDA userspace; the local package is added only so
# the container can build a manifest with the same code the rest of the
# project uses.
serving_image = (
    modal.Image.from_registry(RECIPE.image.reference)
    .env({**BAKED_ENVIRONMENT, "PYTHONPATH": "/opt/llmtracefx"})
    .add_local_dir("llmtracefx", remote_path="/opt/llmtracefx/llmtracefx")
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _manifest_path(revision: str) -> Path:
    return Path(MOUNT_PATH) / revision / MANIFEST_FILENAME


def _read_manifest(revision: str) -> WeightStagingManifest | None:
    path = _manifest_path(revision)
    if not path.is_file():
        return None
    try:
        return WeightStagingManifest.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return None


@app.function(
    image=staging_image,
    volumes={MOUNT_PATH: weights_volume},
    secrets=_staging_secrets,
    timeout=STAGING_TIMEOUT_SECONDS,
    cpu=STAGING_CPU_CORES,
    memory=int(STAGING_MEMORY_GIB * 1024),
    retries=0,
)
def stage_weights(
    repo_id: str,
    revision: str,
    confirm: str,
    volume_name: str,
) -> dict[str, Any]:
    """Fetch one pinned revision onto the volume. CPU and network only.

    ``confirm`` must repeat ``revision``. A bare yes/no flag would be
    carried unchanged from a previous command line and would confirm
    whatever that line now says; restating the SHA ties the confirmation
    to the specific bytes being fetched, so an edited command with a
    stale confirmation fails instead of downloading something nobody
    asked for.

    ``volume_name`` is checked rather than used. The volume is bound when
    the app is built, so a mismatch means the operator believes they are
    writing somewhere other than where the write will land, and that is
    worth failing on.
    """
    from huggingface_hub import HfApi, snapshot_download

    wanted = require_model_revision(revision)
    confirmed = require_model_revision(confirm)
    if wanted != confirmed:
        raise ValueError(
            "--confirm must repeat --revision exactly; refusing to download"
        )
    if wanted != RECIPE.model_revision:
        raise ValueError(
            "requested revision does not match the revision this app was "
            f"configured with ({RECIPE.model_revision}); staging and serving "
            "must agree or the served weights are not the ones recorded"
        )
    repository = require_supported_repo(repo_id)
    if volume_name != PLAN.volume_name:
        raise ValueError(
            f"this app writes to volume {PLAN.volume_name!r}, not " f"{volume_name!r}"
        )

    existing = _read_manifest(wanted)
    if existing is not None and existing.matches(repo_id=repository, revision=wanted):
        print(
            f"Revision {wanted} already staged: {existing.file_count} files, "
            f"{existing.total_bytes / (1024 ** 3):.1f} GiB. Nothing to do."
        )
        return existing.to_dict()

    target = Path(MOUNT_PATH) / wanted
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repository} at {wanted} into {target}")
    started = time.monotonic()
    snapshot_download(
        repo_id=repository,
        revision=wanted,
        local_dir=str(target),
    )
    elapsed = time.monotonic() - started
    print(f"Download finished in {elapsed / 60:.1f} minutes")

    # Hashes come from the repository's own metadata where the upstream
    # published them (large files are LFS/Xet tracked and carry a
    # sha256). Files without published hashes are recorded with sizes
    # only rather than with a hash computed here: hashing hundreds of
    # GiB would double the container time this step is billed for, and a
    # locally computed digest proves the bytes are self-consistent, not
    # that they are the bytes upstream published.
    hashes: dict[str, str] = {}
    try:
        info = HfApi().model_info(
            repo_id=repository, revision=wanted, files_metadata=True
        )
        for sibling in info.siblings or ():
            lfs = getattr(sibling, "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs is not None else None
            if digest:
                hashes[sibling.rfilename] = str(digest)
    except Exception as exc:  # noqa: BLE001 - metadata is a bonus, not a gate
        print(f"File hashes unavailable from repository metadata: {type(exc).__name__}")

    files: list[StagedFile] = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        relative = path.relative_to(target).as_posix()
        files.append(
            StagedFile(
                path=relative,
                size_bytes=path.stat().st_size,
                sha256=hashes.get(relative),
            )
        )

    manifest = WeightStagingManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        completed_at=_utc_now_iso(),
        repo_id=repository,
        revision=wanted,
        mount_path=MOUNT_PATH,
        files=tuple(files),
    )
    _manifest_path(wanted).write_text(manifest.to_json(), encoding="utf-8")
    weights_volume.commit()

    print(
        f"Staged {manifest.file_count} files, "
        f"{manifest.total_bytes / (1024 ** 3):.1f} GiB, "
        f"{manifest.hashed_file_count} with published hashes."
    )
    return manifest.to_dict()


def _verification_path(revision: str) -> Path:
    return Path(MOUNT_PATH) / revision / VERIFICATION_FILENAME


def _read_verification(revision: str) -> WeightVerification | None:
    path = _verification_path(revision)
    if not path.is_file():
        return None
    try:
        return WeightVerification.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError):
        return None


@app.function(
    image=staging_image,
    volumes={MOUNT_PATH: weights_volume},
    timeout=VERIFY_TIMEOUT_SECONDS,
    cpu=VERIFY_CPU_CORES,
    memory=int(VERIFY_MEMORY_GIB * 1024),
    retries=0,
)
def verify_weights(
    volume_name: str, revision: str, check_hashes: bool = True
) -> dict[str, Any]:
    """Re-check the staged tree against its manifest. CPU and disk only.

    A manifest that names the right revision proves a download was
    started, not that it finished. An interruption near the end leaves
    the manifest and a short shard, and the serving container would
    discover that only after allocating four accelerators and loading
    most of a checkpoint. Doing it here costs CPU seconds instead.
    """
    if volume_name != PLAN.volume_name:
        raise ValueError(
            f"this app verifies volume {PLAN.volume_name!r}, not {volume_name!r}"
        )
    wanted = require_model_revision(revision)
    manifest = _read_manifest(wanted)
    if manifest is None:
        raise ValueError(
            f"no manifest for revision {wanted} on {PLAN.volume_name}; run "
            "the stage-weights step first"
        )

    verification = verify_staged_weights(
        manifest,
        Path(MOUNT_PATH) / wanted,
        now=_utc_now_iso,
        check_hashes=check_hashes,
    )
    _verification_path(wanted).write_text(verification.to_json(), encoding="utf-8")
    weights_volume.commit()

    print(verification.to_json())
    if not verification.ok:
        raise ValueError(
            f"{len(verification.issues)} staged file(s) do not match the "
            "manifest; re-run stage-weights before deploying"
        )
    print(
        f"Verified {verification.files_checked} files, "
        f"{verification.bytes_checked / (1024 ** 3):.1f} GiB, "
        f"{verification.hashes_checked} of {verification.hashes_available} "
        "published hashes."
    )
    return verification.to_dict()


@app.function(
    image=staging_image,
    volumes={MOUNT_PATH: weights_volume},
    timeout=300,
    retries=0,
)
def read_manifest(volume_name: str, revision: str) -> dict[str, Any]:
    """Print the staging manifest. CPU only, seconds of container time."""
    if volume_name != PLAN.volume_name:
        raise ValueError(
            f"this app reads volume {PLAN.volume_name!r}, not {volume_name!r}"
        )
    manifest = _read_manifest(require_model_revision(revision))
    if manifest is None:
        raise ValueError(
            f"no manifest for revision {revision} on {PLAN.volume_name}; "
            "run the stage-weights step first"
        )
    print(manifest.to_json())
    return manifest.to_dict()


def _deploy_expiry() -> datetime:
    """The instant after which no container may serve.

    Parsed rather than trusted: a malformed value here would otherwise
    become an exception in the middle of start up, on accelerators, when
    the safe reading of an unparseable expiry is simply "expired".
    """
    raw = (os.environ.get(DEPLOY_EXPIRES_AT_VAR) or DEPLOY_EXPIRES_AT).strip()
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _stop_after(process: subprocess.Popen[bytes], seconds: float) -> None:
    """End the container once its bounded lifetime is up.

    Terminating the server is not enough on its own, because Modal keeps
    the container while the web server port is bound. Exiting the process
    is what actually releases the accelerators, so the wait is bounded
    and then the exit is unconditional.
    """
    time.sleep(seconds)
    print(
        json.dumps({"event": "lifetime_reached", "seconds": round(seconds, 3)}),
        flush=True,
    )
    try:
        process.terminate()
        process.wait(timeout=30)
    except (OSError, subprocess.SubprocessError):
        # Deliberately swallowed. The server may already be gone, or may
        # refuse to die politely; either way the next statement exits the
        # process, which is what actually releases the accelerators.
        # Re-raising here would skip that and leave them allocated.
        pass
    finally:
        os._exit(0)


def _observed_gpus() -> tuple[str, ...]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


def _observed_cuda_version() -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query", "--display=COMPUTE"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        if "CUDA" in line and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _log_startup_when_ready(port: int, started: float, deadline_seconds: int) -> None:
    """Record how long the server took to accept connections.

    Startup time is one of the few things a serving container can
    honestly observe about itself, and on a model this size it is the
    number that dominates the cost of a short validation. It is recorded
    as an observation, with no comparison and no target.
    """
    deadline = started + deadline_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(2.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                print(
                    json.dumps(
                        {
                            "event": "server_ready",
                            "startup_seconds": round(time.monotonic() - started, 3),
                        }
                    )
                )
                return
        time.sleep(2.0)
    print(
        json.dumps(
            {
                "event": "server_not_ready",
                "waited_seconds": deadline_seconds,
            }
        )
    )


@app.function(
    image=serving_image,
    gpu=f"{RECIPE.gpu_type.upper()}:{RECIPE.gpu_count}",
    # Requested explicitly rather than left to the platform default,
    # because the envelope prices these exact figures.
    cpu=SERVING_CPU_CORES,
    memory=int(SERVING_MEMORY_GIB * 1024),
    volumes={MOUNT_PATH: weights_volume},
    secrets=[endpoint_secret],
    timeout=CONTROLS.timeout_seconds,
    scaledown_window=CONTROLS.scaledown_window_seconds,
    max_containers=CONTROLS.max_containers,
    min_containers=CONTROLS.min_containers,
)
@modal.concurrent(max_inputs=CONTROLS.max_concurrent_inputs)
@modal.web_server(
    port=RECIPE.port,
    startup_timeout=CONTROLS.startup_timeout_seconds,
    # The only control that stops an unauthenticated request allocating
    # accelerators. Everything this module checks runs after the platform
    # has already scheduled the container, so refusing in Python bounds
    # how long a container serves, not whether one is started. Modal
    # rejects a request without proxy credentials at its edge with a 401
    # (https://modal.com/docs/guide/webhook-proxy-auth, read 2026-08-30),
    # and its token pair can be sent as a single `Authorization: Bearer`
    # value, so the endpoint stays OpenAI-compatible for the collector.
    requires_proxy_auth=CONTROLS.require_proxy_auth,
)
def serve() -> None:
    """Start the OpenAI-compatible server against the staged weights.

    The staged manifest is verified before the server is launched. That
    check is cheap and it happens while the most expensive resource in
    this deployment is already allocated, so failing here costs seconds
    of accelerator time whereas letting the framework discover the
    missing weights costs its entire start up sequence first.
    """
    # Checked first, and before anything else does work, because it is
    # the gate that bounds total spend. Everything below this line costs
    # accelerator time.
    expiry = _deploy_expiry()
    now = datetime.now(timezone.utc)
    if now >= expiry:
        raise RuntimeError(
            f"this deployment expired at {expiry.isoformat(timespec='seconds')} "
            f"and it is now {now.isoformat(timespec='seconds')}. Refusing to "
            "start. The expiry is what makes the approved cost an upper "
            "bound rather than an estimate: without it a request arriving "
            "at any time in the future would allocate accelerators again. "
            "Run `modal app stop` and redeploy with a fresh window if you "
            "still need it."
        )

    manifest = _read_manifest(RECIPE.model_revision)
    if manifest is None or not manifest.matches(
        repo_id=RECIPE.model_repo_id, revision=RECIPE.model_revision
    ):
        raise RuntimeError(
            f"weights for {RECIPE.model_repo_id} at {RECIPE.model_revision} "
            f"are not staged on volume {PLAN.volume_name}. Run the "
            "stage-weights step before deploying; the server will not "
            "download weights on accelerators."
        )

    # A manifest says a download was attempted for this revision. It does
    # not say the bytes are all there. The verification step establishes
    # that on CPU, and requiring its result here is what keeps a
    # truncated volume from being discovered on four accelerators.
    verification = _read_verification(RECIPE.model_revision)
    if verification is None or not verification.covers(
        repo_id=RECIPE.model_repo_id, revision=RECIPE.model_revision
    ):
        raise RuntimeError(
            f"weights for {RECIPE.model_revision} have not passed "
            f"verification on volume {PLAN.volume_name}. Run the "
            "verify-weights step, which is CPU only, before deploying."
        )

    # Resolved here, with the other precondition, because an absent key is
    # a refusal rather than a degraded mode.
    #
    # Both frameworks install their authentication middleware only when a
    # non-empty key is supplied. Modal proxy auth prevents an unauthenticated
    # request from scheduling this container; framework auth is the second
    # layer once a request reaches it. The runbook uses the proxy token pair
    # as this key so one OpenAI-compatible bearer value satisfies both layers.
    #
    # Modal's `required_keys` does not cover this: it asserts that the
    # variable is present, not that it holds anything, and `modal secret
    # create NAME VAR="$VAR"` in a shell where the export was lost creates
    # exactly that empty secret.
    api_key = os.environ.get(PLAN.endpoint.api_key_env_var, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{PLAN.endpoint.api_key_env_var} is missing or empty in this "
            f"container, so Modal Secret {PLAN.endpoint.modal_secret_name!r} "
            "did not supply a value. Refusing to start: neither serving "
            "framework installs authentication without a key, and this "
            "endpoint has no other protection. Re-create the secret with a "
            "non-empty value and deploy again."
        )

    server_manifest = ServerManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        collected_at=_utc_now_iso(),
        app_name=PLAN.app_name,
        gpu_type=RECIPE.gpu_type,
        gpu_count=RECIPE.gpu_count,
        framework=RECIPE.framework,
        framework_version=RECIPE.framework_version,
        image_reference=RECIPE.image.reference,
        image_digest=RECIPE.image.digest,
        model_repo_id=RECIPE.model_repo_id,
        model_revision=RECIPE.model_revision,
        quantization=RECIPE.facts.quantization,
        quantization_format=RECIPE.facts.quantization_format,
        activation_scheme=RECIPE.facts.activation_scheme,
        tensor_parallel_size=RECIPE.tensor_parallel_size,
        context_length=RECIPE.context_length,
        observed_gpus=_observed_gpus(),
        observed_cuda_version=_observed_cuda_version(),
        credential_env_var_names_present=present_env_var_names(
            os.environ, [PLAN.endpoint.api_key_env_var]
        ),
    )
    # Safe operational metadata only: configuration, pins and observed
    # hardware. There is no field here that can carry a credential.
    print(server_manifest.to_json(indent=None))

    argv = list(RECIPE.launch_argv())
    print(json.dumps({"event": "server_argv", "argv": argv}))

    environment = dict(os.environ)
    if RECIPE.framework == SGLANG:
        # Appended after the argv above has been logged, so the argv this
        # module records stays credential free.
        #
        # That is the limit of what can be promised here. SGLang takes
        # the key only on its command line, and its engine logs its own
        # resolved configuration unredacted, so the value will appear in
        # the container's stdout and in /proc/<pid>/cmdline regardless of
        # what this module does. Reaching this branch at all requires the
        # operator to have passed --accept-argv-credential-exposure,
        # which the planner enforces; the default framework avoids the
        # situation.
        argv.extend(("--api-key", api_key))
    else:
        # vLLM reads the key from the environment, so it never reaches a
        # command line or the server's config repr.
        environment[VLLM_API_KEY_ENV_VAR] = api_key

    started = time.monotonic()
    threading.Thread(
        target=_log_startup_when_ready,
        args=(RECIPE.port, started, CONTROLS.startup_timeout_seconds),
        daemon=True,
    ).start()
    process = subprocess.Popen(argv, env=environment)

    # Modal's timeout does not stop this container: the function returns
    # here, having only launched the server. The watchdog is what makes
    # the per-container lifetime real, and it stops at whichever comes
    # first, this container's own timeout or the deployment expiry, so a
    # container started late cannot outlive the window that was priced.
    remaining_to_expiry = (expiry - now).total_seconds()
    lifetime = max(1.0, min(float(CONTROLS.timeout_seconds), remaining_to_expiry))
    threading.Thread(
        target=_stop_after,
        args=(process, lifetime),
        daemon=False,
    ).start()
