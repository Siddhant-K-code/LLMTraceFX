"""Offline Modal L4 delta for the Qwen3-8B vLLM crossover protocol.

This module defines a separate protocol identity for one future Modal L4
execution of the sealed crossover experiment. It preserves the frozen
scientific core of ``qwen3-8b-vllm-crossover-v2`` exactly (model pin,
runtime pins, two lanes, sealed 32-cell ABBA/BAAB schedule, eight
adjacent pairs per lane, 144 controlled fixed-token-count requests and 12
natural requests per cell, whole-pair statistics, no extrapolation) and
replaces only the provider-conditioned envelope: Modal Function/RPC
placement, per-second composite pricing, container lifecycle controls,
the memory admission gate, the reduced cache-control claim surface, the
authentication policy, and the teardown contract.

Nothing here imports the Modal SDK, opens a socket, reads a credential,
or performs a paid operation. The plan and refusal paths are pure
functions over frozen constants, and the execution-time gates are
verifiers over receipts an authorized caller supplies later.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from pathlib import Path
from typing import Any

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text
from . import vllm_compile
from .vllm_compile import (
    ANALYSIS_SEED,
    BASE_IMAGE_REFERENCE,
    BOOTSTRAP_RESAMPLES,
    CONTROLLED_REQUESTS_PER_CELL,
    CONTROLLED_SAMPLING,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    LANES,
    MODEL_DIRECTORY,
    MODEL_ID,
    MODEL_REVISION,
    NATURAL_REQUESTS_PER_CELL,
    NATURAL_SAMPLING,
    PAIRS_PER_LANE,
    RUNTIME_PINS,
    SAMPLING_SEED,
    SCHEDULE_SEED,
    SIGN_FLIP_ENUMERATIONS,
    VLLM_SOURCE_COMMIT,
    ScheduleCell,
    canonical_decimal,
    canonical_json,
    crossover_schedule,
)

BASE_PROTOCOL_ID = vllm_compile.PROTOCOL_ID
PROTOCOL_ID = "qwen3-8b-vllm-crossover-modal-l4-v1"
PLAN_SCHEMA_VERSION = "1"
LEDGER_SCHEMA_VERSION = "1"
PROVIDER = "Modal"

# ---------------------------------------------------------------------------
# Pricing. Every number below is the operator-supplied current published
# rate; the execution-time gate re-fetches the official page and refuses
# to run when the official rate is higher or a new charge component
# appears. Decimal only: a float cent is a wrong cent.
# ---------------------------------------------------------------------------
L4_GPU_USD_PER_SECOND = Decimal("0.000222")
CPU_USD_PER_CORE_SECOND = Decimal("0.0000131")
MEMORY_USD_PER_GIB_SECOND = Decimal("0.00000222")
VOLUME_USD_PER_GIB_MONTH = Decimal("0.09")
STORAGE_MONTH_DAYS = 30
RATE_COMPONENTS = ("l4_gpu_second", "cpu_core_second", "memory_gib_second")
STORAGE_RATE_COMPONENT = "volume_gib_month"
OFFICIAL_RATE_URL = "https://modal.com/pricing"
OFFICIAL_RATE_DOMAINS = ("modal.com",)

GPU_COUNT = 1
CPU_PHYSICAL_CORES = 4
MEMORY_GIB = 32
VOLUME_RESERVED_GIB = 32
VOLUME_ACTIVE_DAYS = 1
VOLUME_POST_DELETE_DAYS = 4

GPU_FUNCTION_USD_PER_SECOND = Decimal("0.00034544")
CPU_FUNCTION_USD_PER_SECOND = Decimal("0.00012344")
COMPUTE_PLANNED_SECONDS = 15_240
COMPUTE_PLANNED_USD = Decimal("4.5985056")
STORAGE_PLANNED_USD = Decimal("0.48")
TOTAL_PLANNED_USD = Decimal("5.0785056")
UNTOUCHED_MARGIN_USD = Decimal("0.9214944")
HARD_CAP_USD = Decimal("6")

CPU_STAGE_TIMEOUT_SECONDS = 1800
CPU_VERIFY_TIMEOUT_SECONDS = 300
EAGER_CANARY_TIMEOUT_SECONDS = 300
COMPILED_CANARY_TIMEOUT_SECONDS = 420
NATURAL_CELL_TIMEOUT_SECONDS = 240
CONTROLLED_CELL_TIMEOUT_SECONDS = 480
CPU_ANALYSIS_TIMEOUT_SECONDS = 900

EXPECTED_GPU_NAME = "NVIDIA L4"
MIN_TOTAL_VRAM_MIB = 22_000
VRAM_HEADROOM_MIB = 512
GPU_MEMORY_UTILIZATION = Decimal("0.94")
DECODE_STEPS = CONTROLLED_SAMPLING.max_tokens
MAX_MODEL_LEN_MARGIN_TOKENS = DECODE_STEPS

# ---------------------------------------------------------------------------
# Decode-bandwidth feasibility of one controlled cell.
#
# This is an arithmetic proof, computed offline from frozen constants, that
# runs before any authentication, rate fetch, SDK import, or provider call.
# It exists because a sealed timeout is a promise about physics, and the
# sealed controlled-cell timeout turned out to be one this hardware cannot
# keep.
#
# Assumptions, stated so a reviewer can attack them rather than guess them:
#
#   1. Batch-1 autoregressive BF16 decoding is memory-bandwidth bound. Every
#      generated token requires reading at least one full image of the model
#      weights from device memory. This is a *lower* bound: it ignores the KV
#      cache, activations, and every non-weight read.
#   2. The weight image is the exact sum of the five staged safetensors shards.
#      Tokenizer, config, index, README, and license bytes are excluded because
#      they are not streamed from HBM for every decode step.
#   3. The device streams at its advertised peak memory bandwidth. Real
#      achieved bandwidth is strictly lower, so the derived minimum is
#      optimistic in favour of feasibility.
#   4. Initialization, weight load, prefill, and (for the compiled mode)
#      torch.compile / CUDA-graph capture take non-negative time, and are
#      excluded entirely from the minimum.
#
# Sources for the two inputs are recorded, not fetched: the staged byte count
# is the protocol's own sealed staging inventory, and the L4 peak bandwidth is
# the vendor-advertised figure for the accelerator this delta pins.
# ---------------------------------------------------------------------------
STAGED_MODEL_BYTES = EXPECTED_MODEL_BYTES
MODEL_WEIGHT_BYTES = 16_381_516_776
CONTROLLED_CELL_OUTPUT_TOKENS = CONTROLLED_REQUESTS_PER_CELL * DECODE_STEPS
L4_ADVERTISED_PEAK_BANDWIDTH_BYTES_PER_SECOND = 300_000_000_000
DECODE_FEASIBILITY_KIND = "modal_l4_decode_bandwidth_feasibility"
DECODE_FEASIBILITY_SCHEMA_VERSION = "1"
# Reported rates that do not terminate as decimals are rounded down to this
# many places. Rounding down understates achievable throughput, so it can only
# make a *feasible* verdict look worse -- never make an infeasible one pass.
# The verdict itself is decided by exact integer cross-multiplication and never
# by a rounded value.
DECODE_RATE_DECIMAL_PLACES = 12
MODEL_WEIGHT_BYTES_PROVENANCE = (
    "exact sum of the five safetensors weight shards in the sealed staging "
    f"inventory ({MODEL_WEIGHT_BYTES} bytes); the full "
    f"{EXPECTED_MODEL_FILE_COUNT}-file staged inventory is "
    f"{EXPECTED_MODEL_BYTES} bytes"
)
L4_BANDWIDTH_PROVENANCE = (
    "NVIDIA L4 advertised peak memory bandwidth, recorded offline as an "
    "operator-supplied constant and never fetched"
)
DECODE_FEASIBILITY_ASSUMPTIONS = (
    "batch-1 BF16 autoregressive decoding is memory-bandwidth bound and must "
    "stream at least one full model-weight image per generated token",
    "the weight image is the exact sum of the five staged safetensors shards; "
    "non-weight staged files are excluded",
    "the device sustains its advertised peak memory bandwidth, which no real "
    "kernel exceeds",
    "KV-cache, activation, and every other non-weight read is ignored, so the "
    "derived time is a strict lower bound",
)
DECODE_FEASIBILITY_EXCLUSIONS = (
    "container start and image pull",
    "model weight load from the mounted volume",
    "engine initialization and KV-cache allocation",
    "prefill of the controlled prompts",
    "torch.compile and CUDA-graph capture for the compiled mode",
)
DECODE_FEASIBILITY_REMEDY_POLICY = (
    "an infeasible design is refused, not repaired: the sealed request count, "
    "token count, timeout, accelerator, and sample size are preregistered, so "
    "lowering n, tuning the runner, extending the timeout, or changing the GPU "
    "would be a different experiment rather than this one"
)

MAX_LEDGER_ARTIFACT_BYTES = 1_048_576
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_NONCE = re.compile(r"^[0-9a-f]{32,64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TIMESTAMP = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)

# Names that would move authentication, routing, or identity away from the
# operator's standard local Modal profile. Presence alone is refused; the
# value is never read, copied, hashed, or logged.
FORBIDDEN_AUTH_ENVIRONMENT = (
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_TOKEN",
    "MODAL_PROFILE",
    "MODAL_CONFIG_PATH",
    "MODAL_SERVER_URL",
    "MODAL_ENVIRONMENT",
    "MODAL_WORKSPACE",
    "MODAL_IMAGE_BUILDER_VERSION",
    "MODAL_FORCE_BUILD",
)
_CREDENTIAL_ENV = re.compile(
    r"(?:TOKEN|PASSWORD|SECRET|API_KEY|APIKEY|PRIVATE_KEY|CREDENTIAL|COOKIE)",
    re.IGNORECASE,
)

RESOURCE_SETTINGS = {
    "provider": PROVIDER,
    "surface": "modal_functions_rpc_only",
    "public_web_endpoint": False,
    "gpu": "L4",
    "gpu_count": GPU_COUNT,
    "cpu_physical_cores": CPU_PHYSICAL_CORES,
    "memory_gib": MEMORY_GIB,
    "max_containers": 1,
    "min_containers": 0,
    "buffer_containers": 0,
    "max_concurrent_inputs": 1,
    "target_concurrent_inputs": 1,
    "max_live_cells": 1,
    "retries": 0,
    "single_use_cell_containers": True,
    "explicit_timeout_per_stage": True,
}

LIFECYCLE_CONTROLS = {
    "fresh_container_per_cell": True,
    "adjacent_pair_cells": True,
    "hidden_generation_warmups": 0,
    "cell_unique_cache_directories": [
        "vllm",
        "torchinductor",
        "triton",
        "cuda",
        "home",
        "huggingface",
        "xdg",
    ],
    "compile_cache_disabled": True,
    "model_volume_mounted_read_only": True,
    "host_page_cache_reset": False,
    "dedicated_host_required": False,
    "prefix_caching": False,
    "speculative_decoding": False,
    "tensor_parallel_size": 1,
    "max_num_seqs": 1,
}

# The exact delta against the sealed CloudRift protocol. Written down so a
# reviewer sees what changed without diffing two 1,000-line modules, and
# so a test can assert the preserved half really is preserved.
PRESERVED_FROM_BASE_PROTOCOL = (
    "model identity, revision, file count, and byte count",
    "vLLM source commit and runtime version pins",
    "two lanes with eight adjacent eager/compiled pairs each",
    "sealed 32-cell counterbalanced ABBA/BAAB schedule",
    "144 controlled fixed-token-count requests per cell",
    "12 natural-output requests per cell",
    "frozen sampling contracts and sampling seed",
    "whole-pair bootstrap, sign-symmetry test, and censoring rules",
    "no replacement cells, no adaptive stopping, no extrapolation",
)
CHANGED_FROM_BASE_PROTOCOL = (
    "provider envelope is Modal Functions/RPC instead of local Docker",
    "accelerator is one NVIDIA L4 instead of one RTX 4090",
    "pricing is per-second composite instead of a single hourly rate",
    "host page-cache drops and dedicated-host control are removed",
    "two fail-closed GPU memory admission canaries gate the whole measured "
    "block; no cell is dispatched unless both pass",
    "authentication is a standard local Modal profile with no overrides",
    "teardown is a control-plane obligation with a storage allowance",
)

# ---------------------------------------------------------------------------
# Cache and placement claims. The CloudRift protocol could drop the host
# page cache and pin a dedicated host; on Modal neither is observable, so
# those requirements are removed here rather than asserted unverifiably,
# and the claim surface shrinks accordingly.
# ---------------------------------------------------------------------------
REMOVED_CLOUDRIFT_CACHE_REQUIREMENTS = (
    "host page-cache drop before the first cell",
    "host page-cache drop between every pair of adjacent cells",
    "dedicated single-tenant host reservation",
    "host quiescence and idle-GPU admission checks",
)
OBSERVABLE_CACHE_CONTROLS = (
    "fresh single-use container per measured cell",
    "unique writable cache directories per cell",
    "vLLM compile cache disabled by pinned environment",
    "read-only shared model volume",
    "zero hidden generation warmups",
)
UNCONTROLLED_CACHE_LIMITATIONS = (
    "provider container placement is chosen by Modal and is never controlled; "
    "the physical host is never identified, and only whether the two cells of "
    "a pair shared an anonymized placement group is observable",
    "physical host reuse and host page-cache state are not observable",
    "volume and image backend caching are not observable",
    "container scheduling order across a pair is not enforceable",
)
CLAIM_SURFACE = {
    "descriptive_provider_conditioned_paired_results": "supported_by_design",
    "pure_causal_compilation_effect": "unsupported_by_construction",
    "natural_end_to_end_causal_speedup": "unsupported_by_construction",
    "cache_state_controlled_comparison": "unsupported_by_construction",
    "provider_reported_spend": "unsupported_without_provider_receipt",
}

# ---------------------------------------------------------------------------
# Canonical claim identifiers.
#
# One registry, used verbatim by the preregistered (offline) claim matrix, the
# result claim matrix, the result contract, and the catalog verifier, so a
# reader can trace a claim from preregistration to result by its identifier
# alone. Three disjoint groups:
#
#   * offline-only claims are facts about the offline protocol itself and can
#     never be produced by a run;
#   * measured claims are preregistered as "not observed" and are adjudicated
#     against evidence after a run -- they appear in BOTH matrices with the
#     same identifier;
#   * claims unsupported by construction can never be supported on this
#     provider, and also appear in BOTH matrices with the same identifier and
#     the same blocking reason.
# ---------------------------------------------------------------------------
OFFLINE_ONLY_CLAIM_IDS = (
    "offline-protocol-defined",
    "zero-spend-offline-generation",
    "no-provider-authentication",
    "exposed-profile-credential-never-used-by-experiment",
    "exposed-profile-credential-revocation-confirmed",
    "fresh-local-profile-created-without-sharing",
)
MEASURED_CLAIM_IDS = (
    "application-ledger-within-hard-cap",
    "controlled-cell-decode-feasible-on-l4",
    "fixed-token-count-provider-conditioned-crossover",
    "memory-gate-passed",
    "natural-output-quality-preserved",
    "numerically-reproducible-generation",
    "output-identical-generation-crossover",
    "provider-reported-spend-within-hard-cap",
    "provider-teardown-complete",
)
UNSUPPORTED_BY_CONSTRUCTION_CLAIMS = {
    "cache-state-controlled-comparison": "host_page_cache_not_observable",
    "compile-cuda-graph-component-timing": "no_stable_offline_snapshot_hook",
    "hardware-matched-comparison": "container_placement_uncontrolled_across_cells",
    "natural-end-to-end-causal-speedup": "host_page_cache_and_placement_uncontrolled",
    "pure-causal-compilation-effect": "host_page_cache_and_placement_uncontrolled",
}
BLOCKED_CLAIM_IDS = tuple(sorted(UNSUPPORTED_BY_CONSTRUCTION_CLAIMS))
# The exact identifier set each matrix must publish. Both matrices carry the
# measured claims and the blocked claims; only the preregistration carries the
# offline-only claims.
PREREGISTERED_CLAIM_IDS = tuple(
    sorted({*OFFLINE_ONLY_CLAIM_IDS, *MEASURED_CLAIM_IDS, *BLOCKED_CLAIM_IDS})
)
RESULT_CLAIM_IDS = tuple(sorted({*MEASURED_CLAIM_IDS, *BLOCKED_CLAIM_IDS}))
UNSUPPORTED_BY_CONSTRUCTION_STATE = "unsupported_by_construction"

MEASUREMENT_DELTA = {
    "provider_lifecycle": {
        "provenance": "modal_control_plane_receipt_only",
        "missing": "null",
    },
    "container_identity": {
        "provenance": "modal_function_call_receipt",
        "missing": "null",
    },
    "host_page_cache_state": {
        "provenance": "not_observable_on_modal",
        "value": None,
    },
    "physical_host_identity": {
        "provenance": "not_observable_on_modal",
        "value": None,
    },
    "gpu_driver_version": {
        "provenance": "provider_managed_and_not_pinned",
        "missing": "null",
    },
    "application_cost": {
        "provenance": "decimal_seconds_times_committed_published_rate",
        "is_provider_proof": False,
    },
    "provider_reported_spend": {
        "provenance": "external_sanitized_receipt_only",
        "missing": "null",
    },
}

AUTHENTICATION_POLICY = {
    "allowed": "standard_local_modal_profile_at_execution_time",
    "forbidden_environment": list(FORBIDDEN_AUTH_ENVIRONMENT),
    "forbidden_credential_shaped_environment": True,
    "credential_values_never_read": True,
    "credential_values_never_logged_or_hashed": True,
    "authentication_during_offline_planning": False,
    "provider_sdk_imported_on_plan_or_verify_path": False,
}

# ---------------------------------------------------------------------------
# Local profile authentication verdict.
#
# Before any spend, one read-only profile probe confirms a standard
# authenticated local Modal profile. The probe is the running interpreter's own
# ``modal`` module (``sys.executable -m modal token info``), so the probed CLI
# version equals the loaded SDK version by construction rather than by hope, and
# only the exit status is observed. The recorded verdict is a closed boolean
# schema: no profile, account, workspace, or token identifier -- and neither
# HOME nor any path -- is ever retained. This schema is validated identically
# by the execution path that produces it and the result path that consumes it.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Signed operator headroom receipt.
#
# The provider exposes no pre-run spend authority, so headroom can only come
# from a receipt an operator signed out of band. A bare signed dollar figure is
# replayable: the same signature would authorize any plan, any source head, any
# run, forever. The receipt is therefore a closed schema bound to exactly this
# execution -- protocol, plan hash, source head, experiment nonce -- with a
# strict UTC validity window that must cover the whole signed authorization
# window, and its canonical hash is itself bound into the signed authorization.
# No account, workspace, profile, or contact identifier may appear: the key set
# is closed, so identity cannot be smuggled in even under a new name.
# ---------------------------------------------------------------------------
HEADROOM_RECEIPT_KIND = "llmtracefx.modal_l4_crossover.headroom_receipt"
HEADROOM_RECEIPT_SCHEMA_VERSION = "1"
HEADROOM_RECEIPT_FIELDS = (
    "schema_version",
    "kind",
    "protocol_id",
    "plan_sha256",
    "source_head",
    "experiment_nonce",
    "headroom_usd",
    "confirmed_at",
    "expires_at",
)
# The longest a signed headroom confirmation may claim to remain valid. A
# receipt is a statement about an account at a moment, so it is bounded to a
# day rather than left open-ended.
MAX_HEADROOM_RECEIPT_WINDOW_SECONDS = 24 * 3600
FORBIDDEN_HEADROOM_KEY_FRAGMENTS = (
    "account",
    "workspace",
    "profile",
    "email",
    "user",
    "token",
    "secret",
    "customer",
    "org",
)

PROFILE_AUTHENTICATION_GATE = "local_profile_authentication"
PROFILE_AUTHENTICATION_SCHEMA_VERSION = "1"
PROFILE_AUTHENTICATION_MECHANISM = "current_interpreter_python_m_modal_token_info"
PROFILE_AUTHENTICATION_FIELDS = (
    "schema_version",
    "gate",
    "authenticated",
    "mechanism",
    "cli_version",
    "sdk_version",
    "records_profile_identity",
    "checked_at",
)
# Key fragments that would turn the profile verdict into a place profile or
# credential identity could live. Extra keys are already rejected; this names
# why, so a well-meaning addition is refused loudly instead of stored.
FORBIDDEN_PROFILE_KEY_FRAGMENTS = (
    "token",
    "secret",
    "account",
    "workspace",
    "profile_id",
    "user",
    "email",
    "home",
    "path",
    "config",
)

# ---------------------------------------------------------------------------
# Credential exposure gate.
#
# A standard-profile credential was exposed outside this system. Nothing in
# this protocol ever read, copied, hashed, or logged it, and nothing here ever
# will: the gate is a set of booleans plus a short prose reason, and the
# attestation schema is a closed allowlist so a token value, a hash, a prefix,
# a fingerprint, or screenshot metadata cannot be recorded even by accident.
#
# Provider execution refuses until a coordinator confirms revocation of the
# exposed credential and confirms that a fresh standard-profile credential was
# created locally and never shared. Absence is refusal, not permission.
# ---------------------------------------------------------------------------
CREDENTIAL_EXPOSURE_ATTESTATION_KIND = (
    "llmtracefx.modal_l4_crossover.credential_exposure_attestation"
)
CREDENTIAL_EXPOSURE_ATTESTATION_SCHEMA_VERSION = "1"
CREDENTIAL_EXPOSURE_ATTESTATION_FIELDS = (
    "schema_version",
    "kind",
    "protocol_id",
    "exposed_profile_credential_never_used_by_experiment",
    "exposed_profile_credential_revocation_confirmed",
    "revocation_confirmed_by",
    "fresh_local_profile_created_without_sharing",
    "fresh_profile_shared_anywhere",
    "confirmed_at",
    "status",
    "reason",
)
CREDENTIAL_EXPOSURE_REQUIRED_TRUE = (
    "exposed_profile_credential_never_used_by_experiment",
    "exposed_profile_credential_revocation_confirmed",
    "fresh_local_profile_created_without_sharing",
)
CREDENTIAL_EXPOSURE_REQUIRED_FALSE = ("fresh_profile_shared_anywhere",)
CREDENTIAL_EXPOSURE_CONFIRMERS = ("coordinator",)
CREDENTIAL_EXPOSURE_STATUSES = ("cleared", "blocked")
MAX_CREDENTIAL_EXPOSURE_REASON_CHARS = 240
# Key fragments that would turn an attestation into a place a secret could
# live. Extra keys are already rejected; this names why, so a well-meaning
# addition is refused loudly instead of being stored.
FORBIDDEN_ATTESTATION_KEY_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "credential_value",
    "api_key",
    "apikey",
    "hash",
    "sha",
    "digest",
    "fingerprint",
    "prefix",
    "suffix",
    "screenshot",
    "image",
    "url",
    "account",
    "workspace",
    "email",
    "username",
    "user_id",
)
# Values that look like a secret, a digest, or an identifier derived from one.
_CREDENTIAL_SHAPED_VALUE = re.compile(
    r"(?i)(?:sha256:|token|secret|api[_-]?key|password|bearer\b|ak-|"
    r"[A-Za-z0-9+/=_-]{20,})"
)
CREDENTIAL_EXPOSURE_GATE = {
    "incident_class": "standard_profile_credential_exposed_outside_this_system",
    "attestation_kind": CREDENTIAL_EXPOSURE_ATTESTATION_KIND,
    "required_true": list(CREDENTIAL_EXPOSURE_REQUIRED_TRUE),
    "required_false": list(CREDENTIAL_EXPOSURE_REQUIRED_FALSE),
    "confirmed_by": list(CREDENTIAL_EXPOSURE_CONFIRMERS),
    "closed_field_allowlist": list(CREDENTIAL_EXPOSURE_ATTESTATION_FIELDS),
    "records_credential_values": False,
    "records_credential_hashes_or_prefixes": False,
    "records_screenshot_metadata": False,
    "records_credential_derived_identifiers": False,
    "absent_attestation": "refuse_provider_execution",
    "evaluated_before_provider_sdk_import": True,
    "blocks": [
        "provider authentication",
        "provider SDK import",
        "any provider call",
        "any spend",
    ],
}

SCALEDOWN_WINDOW_SECONDS = 2
# Bounded settling budget for the control-plane scale-to-zero observation at
# teardown. The provider's scaledown window for these functions is
# ``SCALEDOWN_WINDOW_SECONDS`` and its control plane is eventually consistent,
# so a single sample taken the instant the ephemeral app context exits observes
# timing rather than teardown. The budget is exact and finite -- twelve samples
# five seconds apart, fifty-five seconds worst case -- so there is never an
# unbounded wait, and a function that has not reported zero by the deadline is
# recorded as unverified rather than assumed torn down. Repeatedly *reading* an
# autoscaler counter is control-plane cleanup verification; it re-dispatches
# nothing, replaces no receipt, and is never a scientific retry.
SCALE_ZERO_POLL_INTERVAL_SECONDS = 5
SCALE_ZERO_POLL_ATTEMPTS = 12
SCALE_ZERO_POLL_TIMEOUT_SECONDS = SCALE_ZERO_POLL_INTERVAL_SECONDS * (
    SCALE_ZERO_POLL_ATTEMPTS - 1
)
SCALE_ZERO_SETTLING_MECHANISM = "bounded_control_plane_polling"

TEARDOWN_CONTRACT = {
    "run_scoped_names": True,
    "creates_credential_secret": False,
    "terminal_paths_cancel_outstanding_calls": True,
    # App shutdown is a local SDK action (exiting the ephemeral app.run()
    # context), not provider-verified deletion; functions are only observable
    # through autoscaler scale-to-zero, not a stop or delete.
    "terminal_paths_exit_app_context": True,
    "app_deletion_provider_verifiable": False,
    "function_teardown_observable_as": "control_plane_scale_to_zero_only",
    "container_inventory_observable": False,
    "verifies_scale_to_zero_through_control_plane": True,
    # Scale-to-zero is observed by a bounded settling poll, never by a single
    # immediate sample and never by an unbounded wait.
    "scale_to_zero_settling_mechanism": SCALE_ZERO_SETTLING_MECHANISM,
    "scale_to_zero_settling_timeout_seconds": SCALE_ZERO_POLL_TIMEOUT_SECONDS,
    "scale_to_zero_settling_is_scientific_retry": False,
    "named_resource_listing_scope": "volumes_only",
    "deletes_run_created_volume": True,
    "deletes_run_created_noncredential_secrets": True,
    "retains_sanitized_receipts": True,
    "ambiguous_teardown_fails_closed": True,
    "storage_allowance_days": VOLUME_POST_DELETE_DAYS,
    "provider_spend_nullable": True,
}

INVALIDATING_OBSERVATIONS = (
    "second_attempt",
    "crash",
    "preemption",
    "timeout",
    "missing_terminal_receipt",
)
ACCEPTED_RESIDUAL_RISK = (
    "Modal may reschedule a container after an infrastructure crash. The "
    "operator accepts that residual, and any observed second attempt, crash, "
    "preemption, timeout, or missing terminal receipt invalidates the run and "
    "triggers teardown."
)

MEMORY_GATE = {
    "immutable_runner_kwargs": {
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": canonical_decimal(GPU_MEMORY_UTILIZATION),
        "enable_prefix_caching": False,
        "speculative_config": None,
        "max_model_len_rule": (
            "exact longest frozen prompt token array length plus "
            f"{MAX_MODEL_LEN_MARGIN_TOKENS}"
        ),
    },
    "staging_verification": {
        "expected_file_count": EXPECTED_MODEL_FILE_COUNT,
        "expected_bytes": EXPECTED_MODEL_BYTES,
        "seals_prompt_token_arrays": True,
    },
    "canaries": ("eager", "compiled"),
    "canary_workload": (
        "the actual longest controlled prompt decoded for " f"{DECODE_STEPS} steps"
    ),
    "pass_conditions": (
        f"exactly one {EXPECTED_GPU_NAME}",
        "runtime pins observed exactly",
        "non-zero KV cache capacity for one max-length request",
        "no out-of-memory event",
        f"full terminal completion of {DECODE_STEPS} decode steps",
        f"peak device memory at most total VRAM minus {VRAM_HEADROOM_MIB} MiB",
    ),
    "tuning_allowed": False,
    "failure_action": "publish_refusal_only",
}


# ---------------------------------------------------------------------------
# Provider SDK surface. Pinned to a version whose API was inspected rather
# than assumed, and probed at execution time. A missing attribute or a
# missing decorator parameter is a refusal: this protocol never invents a
# provider method, and never silently drops a control it promised.
# ---------------------------------------------------------------------------
TESTED_MODAL_VERSION = "1.5.5"
MINIMUM_MODAL_VERSION = (1, 5, 0)
MAXIMUM_MODAL_VERSION_EXCLUSIVE = (2, 0, 0)
REQUIRED_SDK_ATTRIBUTES = (
    "App",
    "Image",
    "Volume",
    "Function",
    "FunctionCall",
    "Secret",
    "concurrent",
    "enable_output",
)
REQUIRED_SDK_MEMBERS = (
    ("App", "function"),
    ("App", "run"),
    ("App", "set_tags"),
    ("Image", "from_registry"),
    ("Volume", "from_name"),
    ("Volume", "with_mount_options"),
    ("Volume", "commit"),
    ("Function", "get_current_stats"),
    ("Function", "remote"),
    ("Function", "spawn"),
    ("FunctionCall", "cancel"),
)
REQUIRED_VOLUME_MANAGER_MEMBERS = ("delete", "list")
REQUIRED_FUNCTION_DECORATOR_PARAMETERS = (
    "image",
    "gpu",
    "cpu",
    "memory",
    "timeout",
    "retries",
    "volumes",
    "max_containers",
    "min_containers",
    "buffer_containers",
    "scaledown_window",
    "max_inputs",
    "single_use_containers",
    "block_network",
    "restrict_modal_access",
    "secrets",
)
# Controls this protocol wanted and the provider does not expose. Each one is
# published as a limitation instead of being claimed or emulated.
UNSUPPORTED_PROVIDER_CONTROLS = {
    "individual_container_deletion": (
        "the SDK exposes no per-container delete; teardown claims cancellation, "
        "ephemeral app termination, and scale-to-zero evidence only"
    ),
    "explicit_app_stop_method": (
        "modal.App has no stop(); the ephemeral app is terminated by exiting "
        "the app.run() context and verified through function autoscaler stats"
    ),
    "pre_run_account_budget_or_headroom": (
        "the SDK exposes no pre-run spend authority; headroom requires a "
        "separately signed operator receipt and is never inferred"
    ),
    "per_run_billed_spend_attribution": (
        "workspace billing reports are workspace-scoped and post-hoc; provider "
        "spend stays null unless a sanitized receipt isolates this run"
    ),
    "host_page_cache_control": "not exposed by the provider",
    "physical_host_pinning": "not exposed by the provider",
}
# The public results builder validates CloudRift authorization, ledger, host
# page-cache and Docker receipts that Modal cannot produce. That is a refusal
# path, not an invitation to synthesise those receipts.
# The exact provider-neutral primitives the Modal results path calls. This is
# the single source of truth: the preregistered plan publishes it, and the
# results module imports it rather than keeping a second list that could drift
# from the code (an earlier pair of lists disagreed, and one of them named a
# function the Modal path never calls directly).
REUSED_PROVIDER_NEUTRAL_PRIMITIVES = (
    "cloudrift_crossover_results._validate_request",
    "cloudrift_crossover_results._compute_pair_effects",
    "cloudrift_crossover_results._identity_summary",
    "cloudrift_crossover_results._analysis_document",
    "cloudrift_crossover_results._natural_evaluation",
    "cloudrift_crossover_results._quality_preservation",
    "vllm_compile.PairCurve",
    "vllm_compile.analyze_pair_curves",
)

STATISTICAL_PUBLICATION = {
    "delegated_builder": (
        "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_results.build_bundle"
    ),
    "delegated_verifier": (
        "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_results.verify_bundle"
    ),
    "accepts_modal_workspace": False,
    "reason": (
        "the existing CloudRift builder is bound to the CloudRift protocol "
        "identity, authorization schema, lifecycle ledger, and host receipts, "
        "so a Modal workspace is never passed to it"
    ),
    "provider_native_results_path": (
        "llmtracefx.optimizer.lab.qwen3_8b.modal_l4_crossover_results"
        ".analyze_modal_run"
    ),
    "consequence": (
        "a completed Modal run is validated and analyzed by the provider-native "
        "results path, which consumes the orchestration receipt and the 32 "
        "sealed inner cell receipts and reuses the CloudRift results core's "
        "provider-neutral statistical primitives without claiming CloudRift or "
        "host-cache proof"
    ),
    "reused_provider_neutral_primitives": list(REUSED_PROVIDER_NEUTRAL_PRIMITIVES),
    "receipt_fabrication_forbidden": True,
    "statistical_core_reimplementation_forbidden": True,
}

MODEL_MOUNT_PATH = "/model"
STATE_MOUNT_PATH = "/model/state"
CONTAINER_CACHE_ROOT = "/cache"
CONTAINER_OUTPUT_ROOT = "/run-output"
STAGING_IMAGE_PYTHON_VERSION = "3.12"
STAGING_IMAGE_HF_HUB_PIN = "huggingface_hub==1.29.0"
CONTAINER_MEMORY_MIB = MEMORY_GIB * 1024


class ModalL4ContractError(ValueError):
    """Raised when the Modal L4 delta contract is violated."""


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(canonical_json(value))


# ---------------------------------------------------------------------------
# Runtime/image identity. Modal builds the container image itself, so no
# provider image digest is available offline. Rather than pretend one exists,
# the exact, reproducible *inputs* to that build are committed to: the
# digest-pinned base image, the pinned Modal SDK version, the staging image
# inputs, the runtime version pins, and the vLLM source commit. Their canonical
# hash is a deterministic derived-image *specification* commitment, explicitly
# not a provider image digest. The signed authorization adds the source head to
# bind the commitment to one repository state, and the GPU canaries and cells
# attest the same commitment and runtime pins.
# ---------------------------------------------------------------------------
RUNTIME_IMAGE_SPEC = {
    "base_image_reference": BASE_IMAGE_REFERENCE,
    "provider_sdk_package": "modal",
    "provider_sdk_version": TESTED_MODAL_VERSION,
    "staging_image_python_version": STAGING_IMAGE_PYTHON_VERSION,
    "staging_image_hub_pin": STAGING_IMAGE_HF_HUB_PIN,
    "runtime_pins": dict(RUNTIME_PINS),
    "vllm_source_commit": VLLM_SOURCE_COMMIT,
}
RUNTIME_IMAGE_SPEC_COMMITMENT = _sha256_json(RUNTIME_IMAGE_SPEC)


def runtime_image_identity(*, source_head: str | None = None) -> dict[str, Any]:
    """Return the honest runtime/image identity block for plan and receipts.

    ``derived_provider_image_digest`` is always null with a stated reason: the
    provider builds the image, so a real image digest is not observable
    offline. ``derived_image_spec_commitment`` is the deterministic commitment
    to the build inputs, and ``source_head`` (when supplied) binds it to one
    repository state.
    """

    block: dict[str, Any] = {
        "base_image_reference": BASE_IMAGE_REFERENCE,
        "provider_sdk_package": "modal",
        "provider_sdk_version": TESTED_MODAL_VERSION,
        "runtime_pins": dict(RUNTIME_PINS),
        "image_build_inputs": dict(RUNTIME_IMAGE_SPEC),
        "derived_image_spec_commitment": RUNTIME_IMAGE_SPEC_COMMITMENT,
        "derived_provider_image_digest": None,
        "derived_provider_image_digest_null_reason": (
            "the provider builds the image; only the digest-pinned base image "
            "and a deterministic commitment to the build inputs are bound, "
            "never a fabricated provider image digest"
        ),
    }
    if source_head is not None:
        block["source_head"] = source_head
        block["runtime_image_run_commitment"] = _sha256_json(
            {
                "derived_image_spec_commitment": RUNTIME_IMAGE_SPEC_COMMITMENT,
                "source_head": source_head,
            }
        )
    return block


def _require_timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise ModalL4ContractError(f"{field} must be an ISO-8601 timestamp with offset")
    return value


def _require_decimal(value: Any, *, field: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ModalL4ContractError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except ArithmeticError as exc:
        raise ModalL4ContractError(
            f"{field} must be a canonical decimal string"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise ModalL4ContractError(f"{field} must be finite and >= 0")
    if canonical_decimal(parsed) != value:
        raise ModalL4ContractError(
            f"{field} must use canonical decimal spelling "
            f"{canonical_decimal(parsed)!r}"
        )
    return parsed


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModalL4ContractError(f"{field} must be a positive integer")
    return value


def gpu_function_rate_usd_per_second() -> Decimal:
    """Return the exact per-second price of one measured GPU container."""

    with localcontext() as context:
        context.prec = 28
        return (
            L4_GPU_USD_PER_SECOND * GPU_COUNT
            + CPU_USD_PER_CORE_SECOND * CPU_PHYSICAL_CORES
            + MEMORY_USD_PER_GIB_SECOND * MEMORY_GIB
        )


def cpu_function_rate_usd_per_second() -> Decimal:
    """Return the exact per-second price of one CPU-only container."""

    with localcontext() as context:
        context.prec = 28
        return (
            CPU_USD_PER_CORE_SECOND * CPU_PHYSICAL_CORES
            + MEMORY_USD_PER_GIB_SECOND * MEMORY_GIB
        )


def storage_reservation_usd() -> Decimal:
    """Return the exact volume reservation cost for the whole run."""

    with localcontext() as context:
        context.prec = 28
        days = Decimal(VOLUME_ACTIVE_DAYS + VOLUME_POST_DELETE_DAYS)
        return (
            Decimal(VOLUME_RESERVED_GIB)
            * VOLUME_USD_PER_GIB_MONTH
            * days
            / Decimal(STORAGE_MONTH_DAYS)
        )


def _cost(seconds: int, rate: Decimal) -> Decimal:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
        raise ModalL4ContractError("seconds must be a non-negative integer")
    with localcontext() as context:
        context.prec = 28
        return Decimal(seconds) * rate


@dataclass(frozen=True)
class ModalStage:
    """One priced stage of the run with its own explicit function timeout."""

    stage_id: str
    kind: str
    accelerated: bool
    occurrences: int
    timeout_seconds: int
    lane: str | None = None

    @property
    def rate_usd_per_second(self) -> Decimal:
        return (
            GPU_FUNCTION_USD_PER_SECOND
            if self.accelerated
            else CPU_FUNCTION_USD_PER_SECOND
        )

    @property
    def total_seconds(self) -> int:
        return self.occurrences * self.timeout_seconds

    @property
    def occurrence_ceiling_usd(self) -> Decimal:
        return _cost(self.timeout_seconds, self.rate_usd_per_second)

    @property
    def total_usd(self) -> Decimal:
        return _cost(self.total_seconds, self.rate_usd_per_second)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "kind": self.kind,
            "lane": self.lane,
            "accelerated": self.accelerated,
            "occurrences": self.occurrences,
            "timeout_seconds": self.timeout_seconds,
            "total_seconds": self.total_seconds,
            "rate_usd_per_second": canonical_decimal(self.rate_usd_per_second),
            "occurrence_ceiling_usd": canonical_decimal(self.occurrence_ceiling_usd),
            "total_usd": canonical_decimal(self.total_usd),
        }


def _controlled_cell_count() -> int:
    return sum(1 for cell in crossover_schedule() if cell.lane == "controlled")


def _natural_cell_count() -> int:
    return sum(1 for cell in crossover_schedule() if cell.lane == "natural")


STAGES = (
    ModalStage(
        stage_id="cpu-stage",
        kind="staging",
        accelerated=False,
        occurrences=1,
        timeout_seconds=CPU_STAGE_TIMEOUT_SECONDS,
    ),
    ModalStage(
        stage_id="cpu-verify",
        kind="staging_verification",
        accelerated=False,
        occurrences=1,
        timeout_seconds=CPU_VERIFY_TIMEOUT_SECONDS,
    ),
    ModalStage(
        stage_id="eager-canary",
        kind="memory_gate",
        accelerated=True,
        occurrences=1,
        timeout_seconds=EAGER_CANARY_TIMEOUT_SECONDS,
    ),
    ModalStage(
        stage_id="compiled-canary",
        kind="memory_gate",
        accelerated=True,
        occurrences=1,
        timeout_seconds=COMPILED_CANARY_TIMEOUT_SECONDS,
    ),
    ModalStage(
        stage_id="natural-cell",
        kind="cell",
        accelerated=True,
        occurrences=_natural_cell_count(),
        timeout_seconds=NATURAL_CELL_TIMEOUT_SECONDS,
        lane="natural",
    ),
    ModalStage(
        stage_id="controlled-cell",
        kind="cell",
        accelerated=True,
        occurrences=_controlled_cell_count(),
        timeout_seconds=CONTROLLED_CELL_TIMEOUT_SECONDS,
        lane="controlled",
    ),
    ModalStage(
        stage_id="cpu-analysis",
        kind="analysis",
        accelerated=False,
        occurrences=1,
        timeout_seconds=CPU_ANALYSIS_TIMEOUT_SECONDS,
    ),
)
STAGE_BY_ID = {stage.stage_id: stage for stage in STAGES}


@dataclass(frozen=True)
class ModalLifecycle:
    """One reservable container lifecycle bound to one stage occurrence."""

    lifecycle_id: str
    stage_id: str
    kind: str
    ordinal: int
    planned_seconds: int
    ceiling_usd: Decimal
    cell_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "stage_id": self.stage_id,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "planned_seconds": self.planned_seconds,
            "ceiling_usd": canonical_decimal(self.ceiling_usd),
            "cell_id": self.cell_id,
        }


def _build_lifecycles() -> tuple[ModalLifecycle, ...]:
    lifecycles: list[ModalLifecycle] = []

    def add(stage_id: str, *, suffix: str, cell_id: str | None = None) -> None:
        stage = STAGE_BY_ID[stage_id]
        lifecycles.append(
            ModalLifecycle(
                lifecycle_id=f"{stage_id}-{suffix}",
                stage_id=stage_id,
                kind=stage.kind,
                ordinal=len(lifecycles) + 1,
                planned_seconds=stage.timeout_seconds,
                ceiling_usd=stage.occurrence_ceiling_usd,
                cell_id=cell_id,
            )
        )

    add("cpu-stage", suffix="01")
    add("cpu-verify", suffix="01")
    add("eager-canary", suffix="01")
    add("compiled-canary", suffix="01")
    for cell in crossover_schedule():
        add(
            f"{cell.lane}-cell",
            suffix=f"{cell.pair_index:02d}-{cell.period_index:02d}",
            cell_id=cell.cell_id,
        )
    add("cpu-analysis", suffix="01")
    return tuple(lifecycles)


LIFECYCLES = _build_lifecycles()
LIFECYCLE_BY_ID = {lifecycle.lifecycle_id: lifecycle for lifecycle in LIFECYCLES}


def _validate_frozen_arithmetic() -> None:
    """Fail at import when a declared total drifts from its derivation."""

    if gpu_function_rate_usd_per_second() != GPU_FUNCTION_USD_PER_SECOND:
        raise ModalL4ContractError("GPU function rate does not derive from components")
    if cpu_function_rate_usd_per_second() != CPU_FUNCTION_USD_PER_SECOND:
        raise ModalL4ContractError("CPU function rate does not derive from components")
    if storage_reservation_usd() != STORAGE_PLANNED_USD:
        raise ModalL4ContractError("storage reservation does not derive from its rate")
    seconds = sum(stage.total_seconds for stage in STAGES)
    compute = sum((stage.total_usd for stage in STAGES), Decimal())
    if seconds != COMPUTE_PLANNED_SECONDS:
        raise ModalL4ContractError("planned compute seconds drifted")
    if compute != COMPUTE_PLANNED_USD:
        raise ModalL4ContractError("planned compute cost drifted")
    if compute + STORAGE_PLANNED_USD != TOTAL_PLANNED_USD:
        raise ModalL4ContractError("planned total cost drifted")
    if TOTAL_PLANNED_USD + UNTOUCHED_MARGIN_USD != HARD_CAP_USD:
        raise ModalL4ContractError("untouched contingency margin drifted")
    if len(LIFECYCLES) != len(crossover_schedule()) + 5:
        raise ModalL4ContractError("lifecycle count drifted from the sealed schedule")
    ledger_total = sum(
        (lifecycle.ceiling_usd for lifecycle in LIFECYCLES),
        Decimal(),
    )
    if ledger_total != COMPUTE_PLANNED_USD:
        raise ModalL4ContractError("lifecycle ceilings do not sum to planned compute")


_validate_frozen_arithmetic()


@dataclass(frozen=True)
class ModalFunctionSpec:
    """Exact provider configuration for one declared Modal Function."""

    function_key: str
    stage_id: str
    accelerated: bool
    image: str
    model_volume_mode: str
    block_network: bool
    restrict_modal_access: bool

    @property
    def timeout_seconds(self) -> int:
        return STAGE_BY_ID[self.stage_id].timeout_seconds

    def modal_kwargs(self) -> dict[str, Any]:
        """Return the exact keyword arguments for ``app.function``.

        The accelerator key is absent, not ``None``, on CPU-only stages:
        there must be no configuration under which a staging or analysis
        container can allocate an L4.
        """

        kwargs: dict[str, Any] = {
            "cpu": CPU_PHYSICAL_CORES,
            "memory": CONTAINER_MEMORY_MIB,
            "timeout": self.timeout_seconds,
            "retries": 0,
            "max_containers": 1,
            "min_containers": 0,
            "buffer_containers": 0,
            "scaledown_window": SCALEDOWN_WINDOW_SECONDS,
            "max_inputs": 1,
            "single_use_containers": True,
            "block_network": self.block_network,
            "restrict_modal_access": self.restrict_modal_access,
            "secrets": [],
        }
        if self.accelerated:
            kwargs["gpu"] = f"L4:{GPU_COUNT}"
        return kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_key": self.function_key,
            "stage_id": self.stage_id,
            "accelerated": self.accelerated,
            "image": self.image,
            "model_volume_mode": self.model_volume_mode,
            "timeout_seconds": self.timeout_seconds,
            "modal_kwargs": self.modal_kwargs(),
        }


FUNCTION_SPECS = (
    ModalFunctionSpec(
        function_key="stage",
        stage_id="cpu-stage",
        accelerated=False,
        image="staging",
        model_volume_mode="read_write",
        block_network=False,
        restrict_modal_access=False,
    ),
    ModalFunctionSpec(
        function_key="verify",
        stage_id="cpu-verify",
        accelerated=False,
        image="runtime",
        model_volume_mode="read_write",
        block_network=False,
        restrict_modal_access=False,
    ),
    ModalFunctionSpec(
        function_key="eager_canary",
        stage_id="eager-canary",
        accelerated=True,
        image="runtime",
        model_volume_mode="read_only",
        block_network=True,
        restrict_modal_access=True,
    ),
    ModalFunctionSpec(
        function_key="compiled_canary",
        stage_id="compiled-canary",
        accelerated=True,
        image="runtime",
        model_volume_mode="read_only",
        block_network=True,
        restrict_modal_access=True,
    ),
    ModalFunctionSpec(
        function_key="natural_cell",
        stage_id="natural-cell",
        accelerated=True,
        image="runtime",
        model_volume_mode="read_only",
        block_network=True,
        restrict_modal_access=True,
    ),
    ModalFunctionSpec(
        function_key="controlled_cell",
        stage_id="controlled-cell",
        accelerated=True,
        image="runtime",
        model_volume_mode="read_only",
        block_network=True,
        restrict_modal_access=True,
    ),
    ModalFunctionSpec(
        function_key="analysis",
        stage_id="cpu-analysis",
        accelerated=False,
        image="runtime",
        model_volume_mode="read_only",
        block_network=True,
        restrict_modal_access=True,
    ),
)
FUNCTION_SPEC_BY_KEY = {spec.function_key: spec for spec in FUNCTION_SPECS}
NO_WEB_ENDPOINT_DECORATORS = (
    "web_endpoint",
    "fastapi_endpoint",
    "asgi_app",
    "wsgi_app",
    "web_server",
)


def cell_function_key(cell: ScheduleCell) -> str:
    """Return the declared Function that may execute one sealed cell."""

    if cell.lane not in LANES:
        raise ModalL4ContractError(f"unknown lane {cell.lane!r}")
    return f"{cell.lane}_cell"


def cell_lifecycle_id(cell: ScheduleCell) -> str:
    """Return the ledger lifecycle identity bound to one sealed cell."""

    matches = [
        lifecycle.lifecycle_id
        for lifecycle in LIFECYCLES
        if lifecycle.cell_id == cell.cell_id
    ]
    if len(matches) != 1:
        raise ModalL4ContractError("cell lifecycle identity is not unique")
    return matches[0]


def call_sequence() -> tuple[dict[str, Any], ...]:
    """Return the exact, sequential provider call order for one run.

    Order is part of the contract, not an implementation detail: the
    canaries gate the measured cells, and the cells follow the sealed
    counterbalanced schedule with no reordering and no replacements.
    """

    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "function_key": "stage",
            "lifecycle_id": "cpu-stage-01",
            "cell_id": None,
            "gates_remaining_steps": True,
        },
        {
            "step": 2,
            "function_key": "verify",
            "lifecycle_id": "cpu-verify-01",
            "cell_id": None,
            "gates_remaining_steps": True,
        },
        {
            "step": 3,
            "function_key": "eager_canary",
            "lifecycle_id": "eager-canary-01",
            "cell_id": None,
            "gates_remaining_steps": True,
        },
        {
            "step": 4,
            "function_key": "compiled_canary",
            "lifecycle_id": "compiled-canary-01",
            "cell_id": None,
            "gates_remaining_steps": True,
        },
    ]
    for cell in crossover_schedule():
        steps.append(
            {
                "step": len(steps) + 1,
                "function_key": cell_function_key(cell),
                "lifecycle_id": cell_lifecycle_id(cell),
                "cell_id": cell.cell_id,
                "gates_remaining_steps": False,
            }
        )
    steps.append(
        {
            "step": len(steps) + 1,
            "function_key": "analysis",
            "lifecycle_id": "cpu-analysis-01",
            "cell_id": None,
            "gates_remaining_steps": False,
        }
    )
    return tuple(steps)


def _parse_version(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ModalL4ContractError("provider SDK version is missing")
    parts = value.split(".")[:3]
    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ModalL4ContractError(
            f"provider SDK version {value!r} is not a dotted release"
        ) from exc
    if len(numbers) != 3:
        raise ModalL4ContractError(
            f"provider SDK version {value!r} is not a dotted release"
        )
    return numbers[0], numbers[1], numbers[2]


def verify_sdk_capabilities(module: Any) -> dict[str, Any]:
    """Probe the installed provider SDK and refuse anything unproven.

    Called with the module object rather than importing it, so the offline
    tests can prove the probe rejects a stripped or drifted SDK without a
    provider package being installed anywhere near them.
    """

    version = getattr(module, "__version__", None)
    parsed = _parse_version(version)
    if not MINIMUM_MODAL_VERSION <= parsed < MAXIMUM_MODAL_VERSION_EXCLUSIVE:
        raise ModalL4ContractError(
            f"provider SDK version {version} is outside the supported range "
            f"{MINIMUM_MODAL_VERSION} <= version < {MAXIMUM_MODAL_VERSION_EXCLUSIVE}"
        )
    missing = [name for name in REQUIRED_SDK_ATTRIBUTES if not hasattr(module, name)]
    if missing:
        raise ModalL4ContractError(
            "provider SDK is missing required attributes: " + ", ".join(sorted(missing))
        )
    missing_members = [
        f"{owner}.{member}"
        for owner, member in REQUIRED_SDK_MEMBERS
        if not hasattr(getattr(module, owner), member)
    ]
    if missing_members:
        raise ModalL4ContractError(
            "provider SDK is missing required members: "
            + ", ".join(sorted(missing_members))
        )
    manager = getattr(module.Volume, "objects", None)
    manager_missing = [
        name
        for name in REQUIRED_VOLUME_MANAGER_MEMBERS
        if manager is None or not hasattr(manager, name)
    ]
    if manager_missing:
        raise ModalL4ContractError(
            "provider SDK volume manager is missing: "
            + ", ".join(sorted(manager_missing))
        )
    try:
        parameters = set(inspect.signature(module.App.function).parameters)
    except (TypeError, ValueError) as exc:
        raise ModalL4ContractError(
            "provider SDK function decorator cannot be inspected"
        ) from exc
    missing_parameters = [
        name
        for name in REQUIRED_FUNCTION_DECORATOR_PARAMETERS
        if name not in parameters
    ]
    if missing_parameters:
        raise ModalL4ContractError(
            "provider SDK function decorator is missing required controls: "
            + ", ".join(sorted(missing_parameters))
        )
    return {
        "verified": True,
        "version": version,
        "tested_version": TESTED_MODAL_VERSION,
        "unsupported_controls": dict(UNSUPPORTED_PROVIDER_CONTROLS),
    }


def max_model_len(longest_prompt_tokens: int) -> int:
    """Return the exact frozen context length for the pinned prompt set."""

    return (
        _require_positive_int(longest_prompt_tokens, field="longest_prompt_tokens")
        + MAX_MODEL_LEN_MARGIN_TOKENS
    )


def run_scoped_names(experiment_nonce: str) -> dict[str, str]:
    """Return unique, run-scoped names for every provider-side resource."""

    if (
        not isinstance(experiment_nonce, str)
        or _NONCE.fullmatch(experiment_nonce) is None
    ):
        raise ModalL4ContractError("experiment nonce must be 32-64 hex characters")
    prefix = f"llmtracefx-qwen3-8b-modal-l4-{experiment_nonce}"
    return {
        "app_name": prefix,
        "volume_name": f"{prefix}-model",
        "stage_function": f"{prefix}-stage",
        "verify_function": f"{prefix}-verify",
        "canary_function": f"{prefix}-canary",
        "cell_function": f"{prefix}-cell",
        "analysis_function": f"{prefix}-analysis",
    }


def require_local_profile_authentication(environ: Mapping[str, str]) -> None:
    """Refuse any credential or routing override before execution begins.

    Only variable names and whether they are non-empty are examined. No
    value is read into a message, hashed, copied, or logged, because a
    guard that quotes the secret it rejected is a leak with good manners.
    """

    if not isinstance(environ, Mapping):
        raise ModalL4ContractError("environment must be a mapping")
    overrides = sorted(
        name
        for name in FORBIDDEN_AUTH_ENVIRONMENT
        if isinstance(environ.get(name), str) and environ.get(name, "").strip()
    )
    if overrides:
        raise ModalL4ContractError(
            "Modal profile, credential, or routing overrides are forbidden: "
            + ", ".join(overrides)
        )
    credential_shaped = sorted(
        name
        for name, value in environ.items()
        if isinstance(value, str)
        and value.strip()
        and _CREDENTIAL_ENV.search(name)
        and name not in FORBIDDEN_AUTH_ENVIRONMENT
    )
    if credential_shaped:
        raise ModalL4ContractError(
            "credential-shaped environment variables are forbidden: "
            + ", ".join(credential_shaped)
        )


def assert_provider_sdk_absent(modules: Mapping[str, Any] | None = None) -> None:
    """Refuse to continue when the Modal SDK is loaded on an offline path."""

    loaded = sys.modules if modules is None else modules
    names = sorted(
        name for name in loaded if name == "modal" or name.startswith("modal.")
    )
    if names:
        raise ModalL4ContractError(
            "the Modal SDK must not be imported on an offline path: " + ", ".join(names)
        )


def _rate_component(receipt: Mapping[str, Any], component: str) -> Decimal:
    rates = receipt.get("rates")
    if not isinstance(rates, dict):
        raise ModalL4ContractError("rate receipt rates must be an object")
    if component not in rates:
        raise ModalL4ContractError(f"rate receipt is missing {component}")
    return _require_decimal(rates[component], field=f"official {component}")


def verify_official_rate_receipt(receipt: Any) -> dict[str, Any]:
    """Gate the run on a freshly fetched, hashed official rate document.

    The run may proceed only when every official rate is exactly the
    committed rate or lower and no additional charge component appeared.
    A higher official rate means the sealed budget understates the run,
    which is a refusal, not a re-plan.
    """

    if not isinstance(receipt, Mapping):
        raise ModalL4ContractError("rate receipt must be an object")
    url = receipt.get("source_url")
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ModalL4ContractError("rate receipt source_url must be an https URL")
    host = url.split("/")[2].split("@")[-1].split(":")[0].lower()
    if host not in OFFICIAL_RATE_DOMAINS and not any(
        host.endswith(f".{domain}") for domain in OFFICIAL_RATE_DOMAINS
    ):
        raise ModalL4ContractError("rate receipt is not from an official domain")
    digest = receipt.get("document_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ModalL4ContractError("rate receipt document hash is invalid")
    fetched_at = _require_timestamp(receipt.get("fetched_at"), field="fetched_at")

    committed = {
        "l4_gpu_second": L4_GPU_USD_PER_SECOND,
        "cpu_core_second": CPU_USD_PER_CORE_SECOND,
        "memory_gib_second": MEMORY_USD_PER_GIB_SECOND,
        STORAGE_RATE_COMPONENT: VOLUME_USD_PER_GIB_MONTH,
    }
    rates = receipt.get("rates")
    if not isinstance(rates, dict):
        raise ModalL4ContractError("rate receipt rates must be an object")
    unexpected = sorted(set(rates) - set(committed))
    if unexpected:
        raise ModalL4ContractError(
            "rate receipt introduces uncommitted charge components: "
            + ", ".join(unexpected)
        )
    missing = sorted(set(committed) - set(rates))
    if missing:
        raise ModalL4ContractError(
            "rate receipt is missing committed charge components: " + ", ".join(missing)
        )
    increases = sorted(
        component
        for component, value in committed.items()
        if _rate_component(receipt, component) > value
    )
    if increases:
        raise ModalL4ContractError(
            "official rates exceed the committed rates: " + ", ".join(increases)
        )
    new_charges = receipt.get("additional_charges")
    if new_charges not in (None, [], ()):
        raise ModalL4ContractError("rate receipt reports an additional charge")
    return {
        "verified": True,
        "source_url": url,
        "document_sha256": digest,
        "fetched_at": fetched_at,
        "official_rates_at_or_below_committed": True,
        "additional_charges": [],
    }


def evaluate_decode_bandwidth_feasibility(
    *,
    model_bytes: int = MODEL_WEIGHT_BYTES,
    output_tokens: int = CONTROLLED_CELL_OUTPUT_TOKENS,
    peak_bandwidth_bytes_per_second: int = (
        L4_ADVERTISED_PEAK_BANDWIDTH_BYTES_PER_SECOND
    ),
    timeout_seconds: int = CONTROLLED_CELL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Prove offline whether one controlled cell can decode inside its timeout.

    The verdict is decided by exact integer arithmetic:

        model_bytes * output_tokens  <=  peak_bandwidth * timeout_seconds

    Both sides are exact integers, so the decision never depends on a rounded
    or floating-point value. The derived seconds and token rates are reported
    alongside so the arithmetic can be re-done by hand from the receipt.

    Nothing here reads the network, the provider, or a credential; the defaults
    are the sealed protocol constants, and the inputs are parameterised only so
    the policy can be exercised against a hypothetical device offline.
    """

    values = {
        "model_bytes": model_bytes,
        "output_tokens": output_tokens,
        "peak_bandwidth_bytes_per_second": peak_bandwidth_bytes_per_second,
        "timeout_seconds": timeout_seconds,
    }
    for field, value in values.items():
        _require_positive_int(value, field=field)
    required_bytes = model_bytes * output_tokens
    streamable_bytes = peak_bandwidth_bytes_per_second * timeout_seconds
    feasible = required_bytes <= streamable_bytes
    with localcontext() as context:
        context.prec = 60
        minimum_decode_seconds = Decimal(required_bytes) / Decimal(
            peak_bandwidth_bytes_per_second
        )
        required_tokens_per_second = Decimal(output_tokens) / Decimal(timeout_seconds)
        peak_tokens_per_second = Decimal(peak_bandwidth_bytes_per_second) / Decimal(
            model_bytes
        )
        shortfall_ratio = minimum_decode_seconds / Decimal(timeout_seconds)
    quantum = Decimal(1).scaleb(-DECODE_RATE_DECIMAL_PLACES)
    return {
        "schema_version": DECODE_FEASIBILITY_SCHEMA_VERSION,
        "kind": DECODE_FEASIBILITY_KIND,
        "protocol_id": PROTOCOL_ID,
        "feasible": feasible,
        "computed_offline": True,
        "uses_sealed_constants": values
        == {
            "model_bytes": MODEL_WEIGHT_BYTES,
            "output_tokens": CONTROLLED_CELL_OUTPUT_TOKENS,
            "peak_bandwidth_bytes_per_second": (
                L4_ADVERTISED_PEAK_BANDWIDTH_BYTES_PER_SECOND
            ),
            "timeout_seconds": CONTROLLED_CELL_TIMEOUT_SECONDS,
        },
        "inputs": {
            **values,
            "controlled_requests_per_cell": CONTROLLED_REQUESTS_PER_CELL,
            "output_tokens_per_request": DECODE_STEPS,
            "accelerator": EXPECTED_GPU_NAME,
            "model_bytes_provenance": MODEL_WEIGHT_BYTES_PROVENANCE,
            "peak_bandwidth_provenance": L4_BANDWIDTH_PROVENANCE,
            "decode_execution_contract": {
                "dtype": "bfloat16",
                "max_num_seqs": 1,
                "enable_prefix_caching": False,
                "speculative_config": None,
                "request_execution": "sequential",
            },
        },
        "assumptions": list(DECODE_FEASIBILITY_ASSUMPTIONS),
        "derivation": {
            "weight_bytes_streamed_per_cell": required_bytes,
            "bytes_streamable_within_timeout": streamable_bytes,
            "minimum_decode_only_seconds": canonical_decimal(minimum_decode_seconds),
            "required_tokens_per_second": canonical_decimal(required_tokens_per_second),
            "theoretical_peak_tokens_per_second": canonical_decimal(
                peak_tokens_per_second.quantize(quantum, rounding=ROUND_DOWN)
            ),
            "minimum_over_timeout_ratio": canonical_decimal(
                shortfall_ratio.quantize(quantum, rounding=ROUND_UP)
            ),
            "rounding": (
                f"rates are reported to {DECODE_RATE_DECIMAL_PLACES} decimal "
                "places; the verdict itself is exact integer arithmetic"
            ),
            "excluded_from_the_minimum": list(DECODE_FEASIBILITY_EXCLUSIONS),
        },
        "verdict": (
            "one controlled cell can stream its weight images within the sealed "
            "timeout"
            if feasible
            else (
                "decode-only weight streaming alone exceeds the sealed "
                f"{timeout_seconds}s controlled-cell timeout, before "
                "initialization, weight load, prefill, or compilation"
            )
        ),
        "remedy_policy": DECODE_FEASIBILITY_REMEDY_POLICY,
    }


def require_controlled_cell_decode_feasible() -> dict[str, Any]:
    """Refuse an infeasible design before anything else can happen.

    The production gate always recomputes the sealed constants. Hypothetical
    devices may be evaluated with ``evaluate_decode_bandwidth_feasibility`` for
    offline planning, but their verdicts cannot be supplied to this gate. An
    infeasible verdict is terminal: the design is refused rather than resized,
    retimed, or re-targeted.
    """

    result = evaluate_decode_bandwidth_feasibility()
    if result.get("feasible") is not True:
        derivation = result.get("derivation")
        detail = ""
        if isinstance(derivation, Mapping):
            detail = (
                " decode-only minimum is "
                f"{derivation.get('minimum_decode_only_seconds')}s against a "
                f"{result.get('inputs', {}).get('timeout_seconds')}s ceiling, "
                f"requiring {derivation.get('required_tokens_per_second')} "
                "tokens/s against a theoretical peak of "
                f"{derivation.get('theoretical_peak_tokens_per_second')} "
                "tokens/s."
            )
        raise ModalL4ContractError(
            "the approved design is infeasible on the pinned accelerator; "
            "refusing before any authentication, rate fetch, provider SDK "
            "import, or provider call." + detail
        )
    return result


def evaluate_memory_gate(observation: Any) -> dict[str, Any]:
    """Adjudicate one canary observation against the frozen memory gate."""

    if not isinstance(observation, Mapping):
        raise ModalL4ContractError("memory gate observation must be an object")
    mode = observation.get("mode")
    if mode not in ("eager", "compiled"):
        raise ModalL4ContractError("memory gate mode must be eager or compiled")
    failures: list[str] = []

    if observation.get("gpu_name") != EXPECTED_GPU_NAME:
        failures.append("gpu_name")
    if observation.get("gpu_count") != GPU_COUNT:
        failures.append("gpu_count")
    pins = observation.get("runtime_pins")
    if not isinstance(pins, Mapping) or dict(pins) != dict(RUNTIME_PINS):
        failures.append("runtime_pins")

    total_vram = observation.get("total_vram_mib")
    peak_vram = observation.get("peak_vram_mib")
    if (
        isinstance(total_vram, bool)
        or not isinstance(total_vram, int)
        or total_vram < MIN_TOTAL_VRAM_MIB
    ):
        failures.append("total_vram_mib")
    elif (
        isinstance(peak_vram, bool)
        or not isinstance(peak_vram, int)
        or peak_vram < 0
        or peak_vram > total_vram - VRAM_HEADROOM_MIB
    ):
        failures.append("peak_vram_mib")

    kv_blocks = observation.get("kv_cache_blocks")
    if isinstance(kv_blocks, bool) or not isinstance(kv_blocks, int) or kv_blocks <= 0:
        failures.append("kv_cache_blocks")
    kv_tokens = observation.get("kv_cache_tokens")
    requested_len = observation.get("max_model_len")
    if (
        isinstance(kv_tokens, bool)
        or not isinstance(kv_tokens, int)
        or isinstance(requested_len, bool)
        or not isinstance(requested_len, int)
        or requested_len <= MAX_MODEL_LEN_MARGIN_TOKENS
        or kv_tokens < requested_len
    ):
        failures.append("kv_cache_tokens")

    if observation.get("out_of_memory") is not False:
        failures.append("out_of_memory")
    if observation.get("generated_tokens") != DECODE_STEPS:
        failures.append("generated_tokens")
    if observation.get("terminal") is not True:
        failures.append("terminal")
    if observation.get("used_longest_controlled_prompt") is not True:
        failures.append("used_longest_controlled_prompt")

    kwargs = observation.get("runner_kwargs")
    expected_kwargs = {
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": canonical_decimal(GPU_MEMORY_UTILIZATION),
        "enable_prefix_caching": False,
        "speculative_config": None,
        "enforce_eager": mode == "eager",
        "max_model_len": requested_len,
    }
    if not isinstance(kwargs, Mapping) or dict(kwargs) != expected_kwargs:
        failures.append("runner_kwargs")

    passed = not failures
    return {
        "mode": mode,
        "passed": passed,
        "failures": sorted(failures),
        "tuning_allowed": False,
        "action": "proceed" if passed else "publish_refusal_only",
    }


def evaluate_attempt_receipts(receipts: Any) -> dict[str, Any]:
    """Adjudicate provider attempt receipts against the invalidation rules."""

    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise ModalL4ContractError("attempt receipts must be a sequence")
    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(receipts, start=1):
        if not isinstance(raw, Mapping):
            raise ModalL4ContractError("attempt receipt must be an object")
        lifecycle_id = raw.get("lifecycle_id")
        if lifecycle_id not in LIFECYCLE_BY_ID:
            raise ModalL4ContractError(
                f"attempt receipt {index} names an unplanned lifecycle"
            )
        attempt = raw.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ModalL4ContractError(f"attempt receipt {index} attempt is invalid")
        if attempt > 1 or lifecycle_id in seen:
            findings.append(
                {"lifecycle_id": lifecycle_id, "observation": "second_attempt"}
            )
        seen.add(lifecycle_id)
        if raw.get("crashed") is True:
            findings.append({"lifecycle_id": lifecycle_id, "observation": "crash"})
        if raw.get("preempted") is True:
            findings.append({"lifecycle_id": lifecycle_id, "observation": "preemption"})
        if raw.get("timed_out") is True:
            findings.append({"lifecycle_id": lifecycle_id, "observation": "timeout"})
        if raw.get("terminal_receipt") is not True:
            findings.append(
                {
                    "lifecycle_id": lifecycle_id,
                    "observation": "missing_terminal_receipt",
                }
            )
    missing = sorted(set(LIFECYCLE_BY_ID) - seen)
    for lifecycle_id in missing:
        findings.append(
            {"lifecycle_id": lifecycle_id, "observation": "missing_terminal_receipt"}
        )
    valid = not findings
    return {
        "valid": valid,
        "findings": sorted(
            findings, key=lambda item: (item["lifecycle_id"], item["observation"])
        ),
        "teardown_required": True,
        "action": "publish_results" if valid else "invalidate_and_tear_down",
    }


def evaluate_credential_exposure_attestation(attestation: Any) -> dict[str, Any]:
    """Adjudicate the coordinator's credential-exposure attestation.

    The attestation carries booleans, one confirmer name, one timestamp, one
    status, and a short reason. Nothing else is accepted, so this function
    cannot become a place where a token, a hash of one, a prefix of one, or
    any identifier derived from one is recorded. A missing or malformed
    attestation is a refusal, never an assumption of clearance.
    """

    if attestation is None:
        return {
            "gate": "credential_exposure",
            "cleared": False,
            "exposed_profile_credential_never_used_by_experiment": True,
            "exposed_profile_credential_revocation_confirmed": False,
            "fresh_local_profile_created_without_sharing": False,
            "fresh_profile_shared_anywhere": False,
            "confirmed_by": None,
            "confirmed_at": None,
            "reason": "no coordinator credential-exposure attestation is present",
            "records_credential_values": False,
            "action": "refuse_provider_execution",
        }
    if not isinstance(attestation, Mapping):
        raise ModalL4ContractError("credential exposure attestation must be an object")
    extra = sorted(set(attestation) - set(CREDENTIAL_EXPOSURE_ATTESTATION_FIELDS))
    unsafe = sorted(
        name
        for name in extra
        if any(
            fragment in name.lower() for fragment in FORBIDDEN_ATTESTATION_KEY_FRAGMENTS
        )
    )
    if unsafe:
        raise ModalL4ContractError(
            "credential exposure attestation must never carry credential or "
            "screenshot derived fields: " + ", ".join(unsafe)
        )
    if extra:
        raise ModalL4ContractError(
            "credential exposure attestation has fields outside the closed "
            "allowlist: " + ", ".join(extra)
        )
    missing = sorted(set(CREDENTIAL_EXPOSURE_ATTESTATION_FIELDS) - set(attestation))
    if missing:
        raise ModalL4ContractError(
            "credential exposure attestation is incomplete: " + ", ".join(missing)
        )
    if attestation["schema_version"] != CREDENTIAL_EXPOSURE_ATTESTATION_SCHEMA_VERSION:
        raise ModalL4ContractError(
            "credential exposure attestation schema version differs"
        )
    if attestation["kind"] != CREDENTIAL_EXPOSURE_ATTESTATION_KIND:
        raise ModalL4ContractError("credential exposure attestation kind differs")
    if attestation["protocol_id"] != PROTOCOL_ID:
        raise ModalL4ContractError(
            "credential exposure attestation is bound to another protocol"
        )
    for field in (
        *CREDENTIAL_EXPOSURE_REQUIRED_TRUE,
        *CREDENTIAL_EXPOSURE_REQUIRED_FALSE,
    ):
        if not isinstance(attestation[field], bool):
            raise ModalL4ContractError(
                f"credential exposure attestation {field} must be a boolean"
            )
    if attestation["status"] not in CREDENTIAL_EXPOSURE_STATUSES:
        raise ModalL4ContractError("credential exposure attestation status is invalid")
    confirmer = attestation["revocation_confirmed_by"]
    if confirmer is not None and confirmer not in CREDENTIAL_EXPOSURE_CONFIRMERS:
        raise ModalL4ContractError(
            "credential exposure revocation must be confirmed by the coordinator"
        )
    reason = attestation["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ModalL4ContractError(
            "credential exposure attestation reason must be non-empty prose"
        )
    if len(reason) > MAX_CREDENTIAL_EXPOSURE_REASON_CHARS:
        raise ModalL4ContractError(
            "credential exposure attestation reason exceeds its bound"
        )
    if _CREDENTIAL_SHAPED_VALUE.search(reason):
        raise ModalL4ContractError(
            "credential exposure attestation reason looks credential shaped and "
            "was refused without being stored"
        )
    confirmed_at = attestation["confirmed_at"]
    if confirmed_at is not None:
        _require_timestamp(confirmed_at, field="credential exposure confirmed_at")

    cleared = (
        all(attestation[field] is True for field in CREDENTIAL_EXPOSURE_REQUIRED_TRUE)
        and all(
            attestation[field] is False for field in CREDENTIAL_EXPOSURE_REQUIRED_FALSE
        )
        and attestation["status"] == "cleared"
        and confirmer in CREDENTIAL_EXPOSURE_CONFIRMERS
        and confirmed_at is not None
    )
    return {
        "gate": "credential_exposure",
        "cleared": cleared,
        "exposed_profile_credential_never_used_by_experiment": attestation[
            "exposed_profile_credential_never_used_by_experiment"
        ],
        "exposed_profile_credential_revocation_confirmed": attestation[
            "exposed_profile_credential_revocation_confirmed"
        ],
        "fresh_local_profile_created_without_sharing": attestation[
            "fresh_local_profile_created_without_sharing"
        ],
        "fresh_profile_shared_anywhere": attestation["fresh_profile_shared_anywhere"],
        "confirmed_by": confirmer if cleared else None,
        "confirmed_at": confirmed_at if cleared else None,
        "reason": reason,
        "records_credential_values": False,
        "action": "proceed" if cleared else "refuse_provider_execution",
    }


def require_credential_exposure_cleared(attestation: Any) -> dict[str, Any]:
    """Refuse every provider path until the exposure gate is cleared."""

    verdict = evaluate_credential_exposure_attestation(attestation)
    if not verdict["cleared"]:
        raise ModalL4ContractError(
            "provider execution is blocked until the coordinator confirms "
            "revocation of the exposed profile credential and creation of a "
            "fresh local profile that was never shared"
        )
    return verdict


def build_credential_exposure_attestation(
    *,
    confirmed_at: str,
    reason: str = (
        "coordinator confirmed revocation of the exposed standard-profile "
        "credential and creation of a fresh local profile that was never shared"
    ),
) -> dict[str, Any]:
    """Return a boolean-only cleared credential-exposure attestation template.

    This records only the now-confirmed status as booleans plus a coordinator
    confirmation and time: the exposed credential was never used by the
    experiment, revocation is confirmed, a fresh local profile was created and
    never shared, and no credential value is recorded. It deliberately carries
    no user, session, account, workspace, or provider identifier, no token,
    hash, prefix, or screenshot metadata.

    It is *not* a signed execution authorization and this function neither
    inspects a profile nor reads any credential; it emits an attestation
    template a coordinator can adopt. The result is validated through the same
    closed-allowlist gate that adjudicates it, so a template that would not
    clear the gate is refused here.
    """

    _require_timestamp(confirmed_at, field="credential exposure confirmed_at")
    attestation = {
        "schema_version": CREDENTIAL_EXPOSURE_ATTESTATION_SCHEMA_VERSION,
        "kind": CREDENTIAL_EXPOSURE_ATTESTATION_KIND,
        "protocol_id": PROTOCOL_ID,
        "exposed_profile_credential_never_used_by_experiment": True,
        "exposed_profile_credential_revocation_confirmed": True,
        "revocation_confirmed_by": "coordinator",
        "fresh_local_profile_created_without_sharing": True,
        "fresh_profile_shared_anywhere": False,
        "confirmed_at": confirmed_at,
        "status": "cleared",
        "reason": reason,
    }
    # Fail closed: the template must itself clear the gate and stay boolean-only.
    require_credential_exposure_cleared(attestation)
    return attestation


def verify_profile_authentication(profile: Any) -> dict[str, Any]:
    """Validate the closed-schema local-profile authentication verdict.

    The verdict must be exactly the pinned schema: the schema version, the gate
    identity, an authenticated boolean, the same-interpreter module mechanism, a
    probed CLI version that equals the loaded SDK version (the real check that
    the probe was the loaded SDK and not some other install), a promise that no
    profile identity was retained, and a timestamp. Any extra key, a
    profile-identity-shaped key, a divergent mechanism, or a cli/sdk version
    mismatch is refused so nothing profile-derived can be bound into evidence.
    """

    if not isinstance(profile, Mapping):
        raise ModalL4ContractError("profile authentication must be an object")
    extra = sorted(set(profile) - set(PROFILE_AUTHENTICATION_FIELDS))
    unsafe = sorted(
        name
        for name in extra
        if any(fragment in name.lower() for fragment in FORBIDDEN_PROFILE_KEY_FRAGMENTS)
    )
    if unsafe:
        raise ModalL4ContractError(
            "profile authentication must never carry profile or credential "
            "derived fields: " + ", ".join(unsafe)
        )
    if extra:
        raise ModalL4ContractError(
            "profile authentication has fields outside its closed allowlist: "
            + ", ".join(extra)
        )
    missing = sorted(set(PROFILE_AUTHENTICATION_FIELDS) - set(profile))
    if missing:
        raise ModalL4ContractError(
            "profile authentication is incomplete: " + ", ".join(missing)
        )
    if profile["schema_version"] != PROFILE_AUTHENTICATION_SCHEMA_VERSION:
        raise ModalL4ContractError("profile authentication schema version differs")
    if profile["gate"] != PROFILE_AUTHENTICATION_GATE:
        raise ModalL4ContractError("profile authentication gate identity differs")
    if profile["authenticated"] is not True:
        raise ModalL4ContractError("local Modal profile is not authenticated")
    if profile["mechanism"] != PROFILE_AUTHENTICATION_MECHANISM:
        raise ModalL4ContractError(
            "profile authentication mechanism is not the same-interpreter module "
            "probe"
        )
    if profile["records_profile_identity"] is not False:
        raise ModalL4ContractError(
            "profile authentication must never retain profile identity"
        )
    cli_version = profile["cli_version"]
    sdk_version = profile["sdk_version"]
    if not isinstance(sdk_version, str) or not sdk_version:
        raise ModalL4ContractError("profile authentication must record the SDK version")
    if not isinstance(cli_version, str) or not cli_version:
        raise ModalL4ContractError("profile authentication must record the CLI version")
    if cli_version != sdk_version:
        raise ModalL4ContractError(
            "probed CLI version does not equal the loaded SDK version"
        )
    _require_timestamp(profile["checked_at"], field="profile authentication checked_at")
    return dict(profile)


def evaluate_teardown_receipt(receipt: Any) -> dict[str, Any]:
    """Adjudicate one terminal teardown receipt; absence or ambiguity fails.

    Only the run-scoped volume can be enumerated by name, so the receipt must
    scope its "no live resource" claim to volumes and must not present app or
    container teardown as provider-verified. App context exit is a local SDK
    action; function teardown is observable only as control-plane scale-to-
    zero. Any ambiguity (a listing that could not be performed, recorded as a
    teardown failure) fails closed rather than reading as complete.
    """

    if not isinstance(receipt, Mapping):
        raise ModalL4ContractError("teardown receipt must be an object")
    required_true = (
        "outstanding_calls_cancelled",
        "app_context_exited",
        "scale_zero_verified_via_control_plane",
        "volume_deleted",
        "run_created_noncredential_secrets_deleted",
        "sanitized_receipts_retained",
    )
    failures = [field for field in required_true if receipt.get(field) is not True]
    if receipt.get("credential_secret_created") is not False:
        failures.append("credential_secret_created")
    # App and container teardown are not provider-verifiable; the receipt must
    # represent that honestly and must never claim otherwise.
    if receipt.get("app_deletion_provider_verified") is not None:
        failures.append("app_deletion_overclaimed")
    if receipt.get("container_inventory_observable") is not False:
        failures.append("container_inventory_overclaimed")
    # Only volumes are enumerable by name; the empty-listing claim must be
    # scoped to volumes so it cannot be read as covering every resource.
    if receipt.get("named_resource_listing_scope") != "volumes_only":
        failures.append("named_resource_listing_scope")
    # Scale-to-zero must have been observed by a bounded settling poll. A
    # single immediate sample after the app context exits would record the
    # provider's scaledown timing rather than its teardown, and an unbounded
    # wait is not a receipt at all, so the mechanism, its finite budget, and
    # its explicit non-retry status are all required.
    settling = receipt.get("scale_zero_settling")
    if (
        not isinstance(settling, Mapping)
        or settling.get("mechanism") != SCALE_ZERO_SETTLING_MECHANISM
        or settling.get("is_scientific_retry") is not False
        or settling.get("poll_timeout_seconds") != SCALE_ZERO_POLL_TIMEOUT_SECONDS
        or not isinstance(settling.get("samples_taken"), int)
        or isinstance(settling.get("samples_taken"), bool)
        or not 1 <= int(settling["samples_taken"]) <= SCALE_ZERO_POLL_ATTEMPTS
    ):
        failures.append("scale_zero_settling")
    live = receipt.get("live_named_volumes")
    if not isinstance(live, Sequence) or isinstance(live, (str, bytes)) or list(live):
        failures.append("live_named_volumes")
    # Fail closed on ambiguity: any recorded teardown failure (including a
    # volume listing that could not be performed) is terminal, never complete.
    reported = receipt.get("teardown_failures")
    if (
        not isinstance(reported, Sequence)
        or isinstance(reported, (str, bytes))
        or list(reported)
    ):
        failures.append("teardown_failures")
    spend = receipt.get("provider_reported_spend_usd")
    if spend is not None:
        _require_decimal(spend, field="provider_reported_spend_usd")
    return {
        "complete": not failures,
        "failures": sorted(failures),
        "provider_reported_spend_usd": spend,
        "provider_reported_spend_null_reason": (
            None if spend is not None else "provider spend receipt is unavailable"
        ),
        "storage_allowance_days": VOLUME_POST_DELETE_DAYS,
    }


def _schedule_document() -> list[dict[str, Any]]:
    return [cell.to_dict() for cell in crossover_schedule()]


def _preserved_core_document() -> dict[str, Any]:
    schedule = _schedule_document()
    return {
        "base_protocol_id": BASE_PROTOCOL_ID,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "expected_file_count": EXPECTED_MODEL_FILE_COUNT,
            "expected_bytes": EXPECTED_MODEL_BYTES,
            "directory": MODEL_DIRECTORY,
        },
        "runtime": {
            "vllm_source_commit": VLLM_SOURCE_COMMIT,
            "base_image_reference": BASE_IMAGE_REFERENCE,
            "runtime_pins": dict(RUNTIME_PINS),
        },
        "lanes": list(LANES),
        "pairs_per_lane": PAIRS_PER_LANE,
        "controlled_requests_per_cell": CONTROLLED_REQUESTS_PER_CELL,
        "natural_requests_per_cell": NATURAL_REQUESTS_PER_CELL,
        "schedule": schedule,
        "schedule_sha256": _sha256_json(schedule),
        "schedule_seed": SCHEDULE_SEED,
        "analysis_seed": ANALYSIS_SEED,
        "sampling_seed": SAMPLING_SEED,
        "sampling": {
            "controlled": CONTROLLED_SAMPLING.to_dict(),
            "natural": NATURAL_SAMPLING.to_dict(),
        },
        "statistics": {
            "independent_unit": "adjacent eager-compiled lifecycle pair",
            "bootstrap_unit": "whole_pair",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "sign_flip_enumerations": SIGN_FLIP_ENUMERATIONS,
            "request_level_resampling": False,
            "headline_extrapolation": False,
            "replacement_cells": False,
            "adaptive_stopping": False,
            "implementation": (
                "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_results"
            ),
        },
    }


@dataclass(frozen=True)
class ModalL4Plan:
    """Strict immutable plan for the Modal L4 crossover delta."""

    stages: tuple[ModalStage, ...]
    lifecycles: tuple[ModalLifecycle, ...]

    @classmethod
    def create(cls) -> ModalL4Plan:
        return cls(stages=STAGES, lifecycles=LIFECYCLES)

    def _content_dict(self) -> dict[str, Any]:
        # Normalised through canonical JSON so a plan read back from disk
        # compares equal to a plan built in memory: a tuple constant and
        # the list it serialises to must not be two different plans.
        return dict(json.loads(canonical_json(self._raw_content_dict())))

    def _raw_content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "provider": PROVIDER,
            "preserved_core": _preserved_core_document(),
            "preserved_from_base_protocol": list(PRESERVED_FROM_BASE_PROTOCOL),
            "changed_from_base_protocol": list(CHANGED_FROM_BASE_PROTOCOL),
            "resource_settings": dict(RESOURCE_SETTINGS),
            "provider_sdk": {
                "package": "modal",
                "tested_version": TESTED_MODAL_VERSION,
                "minimum_version": ".".join(str(p) for p in MINIMUM_MODAL_VERSION),
                "maximum_version_exclusive": ".".join(
                    str(p) for p in MAXIMUM_MODAL_VERSION_EXCLUSIVE
                ),
                "required_attributes": list(REQUIRED_SDK_ATTRIBUTES),
                "required_members": [
                    f"{owner}.{member}" for owner, member in REQUIRED_SDK_MEMBERS
                ],
                "required_function_parameters": list(
                    REQUIRED_FUNCTION_DECORATOR_PARAMETERS
                ),
                "forbidden_web_decorators": list(NO_WEB_ENDPOINT_DECORATORS),
                "capability_probe": "fail_closed_before_any_provider_call",
                "unsupported_controls": dict(UNSUPPORTED_PROVIDER_CONTROLS),
            },
            "functions": [spec.to_dict() for spec in FUNCTION_SPECS],
            "call_sequence": list(call_sequence()),
            "mounts": {
                "model_mount_path": MODEL_MOUNT_PATH,
                "state_mount_path": STATE_MOUNT_PATH,
                "container_cache_root": CONTAINER_CACHE_ROOT,
                "container_output_root": CONTAINER_OUTPUT_ROOT,
                "staging_image_python_version": STAGING_IMAGE_PYTHON_VERSION,
                "staging_image_hub_pin": STAGING_IMAGE_HF_HUB_PIN,
            },
            "statistical_publication": dict(STATISTICAL_PUBLICATION),
            "lifecycle_controls": dict(LIFECYCLE_CONTROLS),
            "hardware": {
                "expected_gpu_name": EXPECTED_GPU_NAME,
                "minimum_total_vram_mib": MIN_TOTAL_VRAM_MIB,
                "vram_headroom_mib": VRAM_HEADROOM_MIB,
                "expected_driver": None,
                "expected_driver_null_reason": (
                    "the provider manages and may change the driver; it is "
                    "recorded, never pinned"
                ),
            },
            "runtime_image": runtime_image_identity(),
            "memory_gate": dict(MEMORY_GATE),
            # The production preflight always recomputes this same sealed
            # verdict. Hypothetical-device verdicts are planning-only and
            # cannot be supplied to the execution or result paths.
            "decode_feasibility": evaluate_decode_bandwidth_feasibility(),
            "claim_ids": {
                "offline_only": list(OFFLINE_ONLY_CLAIM_IDS),
                "measured": list(MEASURED_CLAIM_IDS),
                "unsupported_by_construction": dict(UNSUPPORTED_BY_CONSTRUCTION_CLAIMS),
                "preregistered": list(PREREGISTERED_CLAIM_IDS),
                "result": list(RESULT_CLAIM_IDS),
            },
            "cache_claims": {
                "removed_cloudrift_requirements": list(
                    REMOVED_CLOUDRIFT_CACHE_REQUIREMENTS
                ),
                "observable_controls": list(OBSERVABLE_CACHE_CONTROLS),
                "uncontrolled_limitations": list(UNCONTROLLED_CACHE_LIMITATIONS),
                "claim_surface": dict(CLAIM_SURFACE),
                "blocked_claim_ids": list(BLOCKED_CLAIM_IDS),
            },
            "measurement_delta": dict(MEASUREMENT_DELTA),
            "authentication_policy": dict(AUTHENTICATION_POLICY),
            "credential_exposure_gate": dict(CREDENTIAL_EXPOSURE_GATE),
            "teardown_contract": dict(TEARDOWN_CONTRACT),
            "invalidating_observations": list(INVALIDATING_OBSERVATIONS),
            "accepted_residual_risk": ACCEPTED_RESIDUAL_RISK,
            "pricing": {
                "source_url": OFFICIAL_RATE_URL,
                "l4_gpu_usd_per_second": canonical_decimal(L4_GPU_USD_PER_SECOND),
                "cpu_usd_per_core_second": canonical_decimal(CPU_USD_PER_CORE_SECOND),
                "memory_usd_per_gib_second": canonical_decimal(
                    MEMORY_USD_PER_GIB_SECOND
                ),
                "gpu_function_usd_per_second": canonical_decimal(
                    GPU_FUNCTION_USD_PER_SECOND
                ),
                "cpu_function_usd_per_second": canonical_decimal(
                    CPU_FUNCTION_USD_PER_SECOND
                ),
                "volume_usd_per_gib_month": canonical_decimal(VOLUME_USD_PER_GIB_MONTH),
                "storage_month_days": STORAGE_MONTH_DAYS,
                "rate_verification": (
                    "re-fetch and hash the official page before execution; "
                    "refuse when any official rate is higher or a new charge "
                    "component appears"
                ),
            },
            "budget": {
                "stages": [stage.to_dict() for stage in self.stages],
                "lifecycles": [lifecycle.to_dict() for lifecycle in self.lifecycles],
                "compute_planned_seconds": COMPUTE_PLANNED_SECONDS,
                "compute_planned_usd": canonical_decimal(COMPUTE_PLANNED_USD),
                "storage": {
                    "reserved_gib": VOLUME_RESERVED_GIB,
                    "active_days": VOLUME_ACTIVE_DAYS,
                    "post_delete_days": VOLUME_POST_DELETE_DAYS,
                    "planned_usd": canonical_decimal(STORAGE_PLANNED_USD),
                },
                "total_planned_usd": canonical_decimal(TOTAL_PLANNED_USD),
                "untouched_margin_usd": canonical_decimal(UNTOUCHED_MARGIN_USD),
                "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
                "contingency_is_never_spent_on_science": True,
                "application_ledger_required": True,
                "application_ledger_is_provider_proof": False,
            },
            "provenance_policy": {
                "offline_only": True,
                "imports_provider_sdk": False,
                "path_bound": True,
                "source_head_bound": True,
                "canonical_json_only": True,
            },
            "null_policy": {
                "allow_null_unobservable_components": True,
                "null_requires_explicit_reason": True,
                "missing_observation_is_not_zero": True,
            },
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        data = self._content_dict()
        data["plan_sha256"] = self.content_sha256
        return data

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> ModalL4Plan:
        if not isinstance(data, Mapping):
            raise ModalL4ContractError("plan must be an object")
        expected = cls.create().to_dict()
        missing = sorted(set(expected) - set(data))
        extra = sorted(set(data) - set(expected))
        if missing or extra:
            raise ModalL4ContractError(
                f"plan keys must match exactly; missing={missing!r} extra={extra!r}"
            )
        if dict(data) != expected:
            raise ModalL4ContractError(
                "plan does not exactly match the frozen Modal L4 crossover delta"
            )
        return cls.create()

    @classmethod
    def from_json(cls, payload: str) -> ModalL4Plan:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise ModalL4ContractError(f"invalid plan JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> ModalL4Plan:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError) as exc:
            raise ModalL4ContractError(f"failed to read plan: {exc}") from exc
        return cls.from_json(payload)


def build_default_plan() -> ModalL4Plan:
    return ModalL4Plan.create()


def offline_plan_document() -> dict[str, Any]:
    """Return the deterministic no-spend Modal L4 plan/refusal document."""

    assert_provider_sdk_absent()
    plan = build_default_plan()
    feasibility = evaluate_decode_bandwidth_feasibility()
    blockers = [
        (
            "No coordinator attestation of exposed-credential revocation and "
            "fresh unshared local profile creation is present."
        ),
        "No explicit execution authorization receipt is present.",
        "No re-fetched and hashed official Modal rate receipt is present.",
        "No memory-gate canary observation is present.",
        "No provider attempt or terminal teardown receipt is present.",
        "Provider-reported spend is unavailable and is never inferred.",
    ]
    if not feasibility["feasible"]:
        # The refusal is terminal and comes first: it is decided offline from
        # frozen constants, so no later receipt can unblock it.
        blockers.insert(
            0,
            (
                "The approved design is infeasible on the pinned accelerator: "
                "decode-only weight streaming for one controlled cell needs at "
                f"least {feasibility['derivation']['minimum_decode_only_seconds']}"
                "s against the sealed "
                f"{feasibility['inputs']['timeout_seconds']}s timeout."
            ),
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "llmtracefx.modal_l4_crossover.offline_plan",
        "protocol_id": PROTOCOL_ID,
        "base_protocol_id": BASE_PROTOCOL_ID,
        "plan": plan.to_dict(),
        "execution_authorized": False,
        "offline_only": True,
        "network_request_performed": False,
        "provider_authentication_used": False,
        "provider_sdk_imported": False,
        "credential_exposure_gate": evaluate_credential_exposure_attestation(None),
        "exposed_profile_credential_never_used_by_experiment": True,
        "container_created": False,
        "model_downloaded": False,
        "gpu_used": False,
        "spend_usd": "0",
        "decode_feasibility": feasibility,
        "execution_refused_offline": not feasibility["feasible"],
        "blockers": blockers,
        "unsupported_claims": [
            "compilation crossover",
            "performance improvement",
            "output identity",
            "correctness preservation",
            "runtime component timing",
            "cache-state controlled comparison",
            "causal serving speedup",
            "provider-reported spend",
            "provider teardown",
        ],
    }


def _event_hash(event: Mapping[str, Any]) -> str:
    material = {key: value for key, value in event.items() if key != "event_sha256"}
    return _sha256_json(material)


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    material = {key: value for key, value in payload.items() if key != "ledger_sha256"}
    material["ledger_sha256"] = _sha256_json(material)
    return material


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _planned_ledger_entry(lifecycle: ModalLifecycle) -> dict[str, Any]:
    return {
        **lifecycle.to_dict(),
        "status": "planned",
        "call_id": None,
        "reserved_at": None,
        "completed_at": None,
        "aborted_at": None,
        "actual_seconds": None,
        "actual_cost_usd": None,
        "observed_duration_provenance": None,
        "abort_reason": None,
    }


def _initial_ledger_payload(
    *,
    plan: ModalL4Plan,
    source_head: str,
    experiment_nonce: str,
    ledger_path_sha256: Any,
) -> dict[str, Any]:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "provider": PROVIDER,
        "is_provider_proof": False,
        "provider_reported_spend_usd": None,
        "provider_reported_spend_null_reason": (
            "provider spend is external, sanitized, and never inferred"
        ),
        "plan_sha256": plan.content_sha256,
        "source_head": source_head,
        "experiment_nonce": experiment_nonce,
        "ledger_path_sha256": ledger_path_sha256,
        "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
        "storage_reserved_usd": canonical_decimal(STORAGE_PLANNED_USD),
        "revision": 0,
        "reserved_usd": canonical_decimal(STORAGE_PLANNED_USD),
        "remaining_usd": canonical_decimal(HARD_CAP_USD - STORAGE_PLANNED_USD),
        "entries": [_planned_ledger_entry(lifecycle) for lifecycle in plan.lifecycles],
        "events": [],
    }


def _append_ledger_event(
    payload: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    event["previous_event_sha256"] = (
        payload["events"][-1]["event_sha256"] if payload["events"] else None
    )
    event["index"] = len(payload["events"]) + 1
    event["event_sha256"] = _event_hash(event)
    payload["events"].append(event)
    payload["revision"] += 1
    return event


def _replay_ledger_events(
    events: Sequence[Any], *, lifecycles: Sequence[ModalLifecycle]
) -> dict[str, Any]:
    """Replay an append-only ledger event log into its derived entry states.

    Pure and file-free so both the live file-backed ledger and any later
    adjudicator of an embedded ledger snapshot verify the seal, the event
    hash-chain, the per-lifecycle state machine, and the reserved total with
    exactly the same code, never a re-implementation that could drift.
    """

    entries = {
        lifecycle.lifecycle_id: _planned_ledger_entry(lifecycle)
        for lifecycle in lifecycles
    }
    total = Decimal()
    previous: str | None = None
    calls: dict[str, str] = {}
    for index, raw in enumerate(events, start=1):
        if not isinstance(raw, dict):
            raise ModalL4ContractError("ledger event must be an object")
        if raw.get("index") != index:
            raise ModalL4ContractError("ledger event index is invalid")
        if raw.get("previous_event_sha256") != previous:
            raise ModalL4ContractError("ledger event chain is broken")
        if raw.get("event_sha256") != _event_hash(raw):
            raise ModalL4ContractError("ledger event hash does not verify")
        call_id = raw.get("call_id")
        if not isinstance(call_id, str) or _SAFE_ID.fullmatch(call_id) is None:
            raise ModalL4ContractError("ledger event call identity is invalid")
        event_type = raw.get("event_type")
        if event_type == "reserve":
            lifecycle_id = raw.get("lifecycle_id")
            if lifecycle_id not in entries:
                raise ModalL4ContractError(
                    "ledger event lifecycle_id is not in the plan"
                )
            entry = entries[lifecycle_id]
            if entry["status"] != "planned":
                raise ModalL4ContractError("ledger reservation was replayed")
            if call_id in calls:
                raise ModalL4ContractError("ledger call identity is duplicated")
            reserved = _require_decimal(
                raw.get("reserved_usd"), field="ledger event reserved_usd"
            )
            if reserved != _require_decimal(
                entry["ceiling_usd"], field="planned lifecycle ceiling_usd"
            ):
                raise ModalL4ContractError(
                    "ledger reservation differs from its planned ceiling"
                )
            _require_timestamp(raw.get("reserved_at"), field="reserved_at")
            entry["status"] = "reserved"
            entry["call_id"] = call_id
            entry["reserved_at"] = raw["reserved_at"]
            calls[call_id] = lifecycle_id
            total += reserved
        elif event_type == "complete":
            if call_id not in calls:
                raise ModalL4ContractError("ledger completion is unknown")
            entry = entries[calls[call_id]]
            if entry["status"] != "reserved":
                raise ModalL4ContractError("ledger completion was replayed")
            _require_timestamp(raw.get("completed_at"), field="completed_at")
            actual_seconds = raw.get("actual_seconds")
            if (
                isinstance(actual_seconds, bool)
                or not isinstance(actual_seconds, int)
                or actual_seconds < 0
                or actual_seconds > entry["planned_seconds"]
            ):
                raise ModalL4ContractError(
                    "ledger completion actual_seconds is invalid"
                )
            stage = STAGE_BY_ID[entry["stage_id"]]
            actual_cost = _require_decimal(
                raw.get("actual_cost_usd"), field="ledger event actual_cost_usd"
            )
            if actual_cost != _cost(actual_seconds, stage.rate_usd_per_second):
                raise ModalL4ContractError(
                    "ledger completion cost does not match its stage rate"
                )
            entry["status"] = "completed"
            entry["completed_at"] = raw["completed_at"]
            entry["actual_seconds"] = actual_seconds
            entry["actual_cost_usd"] = raw["actual_cost_usd"]
            entry["observed_duration_provenance"] = raw.get("duration_provenance")
        elif event_type == "abort":
            if call_id not in calls:
                raise ModalL4ContractError("ledger abort is unknown")
            entry = entries[calls[call_id]]
            if entry["status"] != "reserved":
                raise ModalL4ContractError("ledger abort was replayed")
            _require_timestamp(raw.get("aborted_at"), field="aborted_at")
            reason = raw.get("abort_reason")
            if not isinstance(reason, str) or not reason:
                raise ModalL4ContractError("ledger abort reason must be non-empty")
            entry["status"] = "aborted"
            entry["aborted_at"] = raw["aborted_at"]
            entry["abort_reason"] = reason
        else:
            raise ModalL4ContractError("ledger event type is invalid")
        previous = raw["event_sha256"]
    return {"entries": entries, "reserved_usd": total}


def verify_ledger_document(
    payload: Any,
    *,
    plan: ModalL4Plan,
    source_head: str,
    experiment_nonce: str,
    path_sha256: str | None = None,
) -> dict[str, Any]:
    """Comprehensively adjudicate an application-ledger document.

    Verifies the seal, the immutable header bindings (protocol, plan hash,
    source head, nonce), the append-only event hash-chain and per-lifecycle
    state machine, that the persisted entries recompute exactly from the log,
    and that the reserved and remaining totals reconcile within the hard cap.
    This is the single trusted validator: the live ledger's ``_read`` and any
    later adjudicator of an embedded snapshot both call it, so no shallow
    ``reserved_usd`` check can stand in for it.
    """

    if not isinstance(payload, Mapping):
        raise ModalL4ContractError("application ledger must be an object")
    payload = dict(payload)
    seal = payload.get("ledger_sha256")
    if not isinstance(seal, str) or _seal(payload)["ledger_sha256"] != seal:
        raise ModalL4ContractError("application ledger seal does not verify")
    ledger_path_sha256 = payload.get("ledger_path_sha256")
    if (
        not isinstance(ledger_path_sha256, str)
        or _SHA256.fullmatch(ledger_path_sha256) is None
    ):
        raise ModalL4ContractError("application ledger path commitment is invalid")
    if path_sha256 is not None and ledger_path_sha256 != path_sha256:
        raise ModalL4ContractError("application ledger path binding does not match")
    expected = _initial_ledger_payload(
        plan=plan,
        source_head=source_head,
        experiment_nonce=experiment_nonce,
        ledger_path_sha256=ledger_path_sha256,
    )
    for field in (
        "schema_version",
        "protocol_id",
        "provider",
        "is_provider_proof",
        "plan_sha256",
        "source_head",
        "experiment_nonce",
        "hard_cap_usd",
        "storage_reserved_usd",
    ):
        if payload.get(field) != expected[field]:
            raise ModalL4ContractError(
                f"application ledger {field} binding does not match"
            )
    revision = payload.get("revision")
    events = payload.get("events")
    entries = payload.get("entries")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 0
        or not isinstance(events, list)
        or not isinstance(entries, list)
        or revision != len(events)
    ):
        raise ModalL4ContractError("application ledger revision/event count is invalid")
    replayed = _replay_ledger_events(events, lifecycles=plan.lifecycles)
    expected_entries = [
        replayed["entries"][lifecycle.lifecycle_id] for lifecycle in plan.lifecycles
    ]
    if entries != expected_entries:
        raise ModalL4ContractError(
            "application ledger entries do not match the append-only event log"
        )
    total = replayed["reserved_usd"] + STORAGE_PLANNED_USD
    if total > HARD_CAP_USD:
        raise ModalL4ContractError("application reservations exceed the hard cap")
    if payload.get("reserved_usd") != canonical_decimal(total):
        raise ModalL4ContractError("application reserved total does not verify")
    if payload.get("remaining_usd") != canonical_decimal(HARD_CAP_USD - total):
        raise ModalL4ContractError("application remaining total does not verify")
    return {
        "entries": expected_entries,
        "reserved_usd": canonical_decimal(total),
        "revision": revision,
    }


def build_completed_ledger_document(
    *,
    plan: ModalL4Plan,
    source_head: str,
    experiment_nonce: str,
    ledger_path_sha256: str,
    reserved_at: str,
    completed_at: str,
    actual_seconds: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build the exact sealed ledger a clean, fully completed run produces.

    File-free and deterministic. It runs the real reserve/complete event and
    seal logic for every planned lifecycle so the returned snapshot verifies
    under :func:`verify_ledger_document`. Used to model production output in
    offline fixtures without any provider or filesystem interaction.
    """

    payload = _initial_ledger_payload(
        plan=plan,
        source_head=source_head,
        experiment_nonce=experiment_nonce,
        ledger_path_sha256=ledger_path_sha256,
    )
    entries_by_id = {entry["lifecycle_id"]: entry for entry in payload["entries"]}
    for index, lifecycle in enumerate(plan.lifecycles, start=1):
        call_id = f"call-{index:03d}"
        entry = entries_by_id[lifecycle.lifecycle_id]
        already = _require_decimal(payload["reserved_usd"], field="ledger reserved_usd")
        _append_ledger_event(
            payload,
            {
                "event_type": "reserve",
                "call_id": call_id,
                "lifecycle_id": lifecycle.lifecycle_id,
                "stage_id": lifecycle.stage_id,
                "reserved_usd": canonical_decimal(lifecycle.ceiling_usd),
                "reserved_at": reserved_at,
            },
        )
        entry["status"] = "reserved"
        entry["call_id"] = call_id
        entry["reserved_at"] = reserved_at
        payload["reserved_usd"] = canonical_decimal(already + lifecycle.ceiling_usd)
        payload["remaining_usd"] = canonical_decimal(
            HARD_CAP_USD - already - lifecycle.ceiling_usd
        )
        seconds = lifecycle.planned_seconds
        if actual_seconds is not None and lifecycle.lifecycle_id in actual_seconds:
            seconds = int(actual_seconds[lifecycle.lifecycle_id])
        stage = STAGE_BY_ID[lifecycle.stage_id]
        actual_cost = canonical_decimal(_cost(seconds, stage.rate_usd_per_second))
        _append_ledger_event(
            payload,
            {
                "event_type": "complete",
                "call_id": call_id,
                "completed_at": completed_at,
                "actual_seconds": seconds,
                "duration_provenance": "client_observed_monotonic_ceiling_seconds",
                "actual_cost_usd": actual_cost,
            },
        )
        entry["status"] = "completed"
        entry["completed_at"] = completed_at
        entry["actual_seconds"] = seconds
        entry["actual_cost_usd"] = actual_cost
        entry["observed_duration_provenance"] = (
            "client_observed_monotonic_ceiling_seconds"
        )
    return _seal(payload)


class ModalApplicationLedger:
    """Append-only application-side reservation ledger for Modal lifecycles.

    This ledger is mandatory and is explicitly *not* provider proof. It
    records what this code reserved and observed at published list rates;
    it cannot attest what Modal billed, and every document it produces
    says so.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        plan: ModalL4Plan,
        git_head: str,
        experiment_nonce: str,
    ) -> None:
        if not isinstance(git_head, str) or _GIT_HEAD.fullmatch(git_head) is None:
            raise ModalL4ContractError("git head must be an exact 40-hex commit")
        if (
            not isinstance(experiment_nonce, str)
            or _NONCE.fullmatch(experiment_nonce) is None
        ):
            raise ModalL4ContractError("experiment nonce must be 32-64 hex characters")
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.plan = plan
        self.git_head = git_head
        self.experiment_nonce = experiment_nonce
        self._high_water_revision = 0
        self._high_water_reserved = Decimal()
        self._lifecycle_by_id = {
            lifecycle.lifecycle_id: lifecycle for lifecycle in plan.lifecycles
        }

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        plan: ModalL4Plan,
        git_head: str,
        experiment_nonce: str,
    ) -> ModalApplicationLedger:
        ledger = cls(
            path, plan=plan, git_head=git_head, experiment_nonce=experiment_nonce
        )
        with _locked(ledger.lock_path):
            if ledger.path.exists():
                raise ModalL4ContractError(
                    "application ledger already exists and cannot be reset"
                )
            ledger._write(ledger._initial_payload())
        return ledger

    def _initial_payload(self) -> dict[str, Any]:
        return _initial_ledger_payload(
            plan=self.plan,
            source_head=self.git_head,
            experiment_nonce=self.experiment_nonce,
            ledger_path_sha256=_sha256_text(str(self.path)),
        )

    def _write(self, payload: Mapping[str, Any]) -> None:
        sealed = _seal(payload)
        atomic_write_text(
            self.path,
            json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        revision = sealed.get("revision")
        reserved = sealed.get("reserved_usd")
        if isinstance(revision, int) and isinstance(reserved, str):
            self._high_water_revision = max(self._high_water_revision, revision)
            self._high_water_reserved = max(
                self._high_water_reserved,
                _require_decimal(reserved, field="ledger reserved_usd"),
            )

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(
                read_bounded_regular_text(self.path, MAX_LEDGER_ARTIFACT_BYTES),
                parse_constant=reject_non_finite_json_constant,
            )
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise ModalL4ContractError(
                f"failed to read application ledger: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ModalL4ContractError("application ledger must be an object")
        summary = verify_ledger_document(
            payload,
            plan=self.plan,
            source_head=self.git_head,
            experiment_nonce=self.experiment_nonce,
            path_sha256=_sha256_text(str(self.path)),
        )
        revision = summary["revision"]
        total = _require_decimal(summary["reserved_usd"], field="ledger reserved_usd")
        if revision < self._high_water_revision or total < self._high_water_reserved:
            raise ModalL4ContractError("application ledger rollback detected")
        self._high_water_revision = revision
        self._high_water_reserved = total
        return payload

    def _replay(self, events: Sequence[Any]) -> dict[str, Any]:
        return _replay_ledger_events(events, lifecycles=self.plan.lifecycles)

    def snapshot(self) -> dict[str, Any]:
        with _locked(self.lock_path):
            return self._read()

    def _append(self, payload: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        return _append_ledger_event(payload, event)

    def reserve(
        self,
        call_id: str,
        *,
        lifecycle_id: str,
        reserved_at: str,
    ) -> dict[str, Any]:
        if not isinstance(call_id, str) or _SAFE_ID.fullmatch(call_id) is None:
            raise ModalL4ContractError("call_id is not a safe stable identity")
        if lifecycle_id not in self._lifecycle_by_id:
            raise ModalL4ContractError("reservation lifecycle_id is not in the plan")
        _require_timestamp(reserved_at, field="reserved_at")
        lifecycle = self._lifecycle_by_id[lifecycle_id]
        with _locked(self.lock_path):
            payload = self._read()
            entry = next(
                item
                for item in payload["entries"]
                if item["lifecycle_id"] == lifecycle_id
            )
            if entry["status"] != "planned":
                raise ModalL4ContractError(
                    "planned lifecycle is already reserved or completed"
                )
            if any(event["call_id"] == call_id for event in payload["events"]):
                raise ModalL4ContractError(f"call {call_id!r} is already reserved")
            already = _require_decimal(
                payload["reserved_usd"], field="ledger reserved_usd"
            )
            if already + lifecycle.ceiling_usd > HARD_CAP_USD:
                raise ModalL4ContractError(
                    "pre-call reservation refused: the hard cap would be exceeded"
                )
            event = self._append(
                payload,
                {
                    "event_type": "reserve",
                    "call_id": call_id,
                    "lifecycle_id": lifecycle_id,
                    "stage_id": lifecycle.stage_id,
                    "reserved_usd": canonical_decimal(lifecycle.ceiling_usd),
                    "reserved_at": reserved_at,
                },
            )
            entry["status"] = "reserved"
            entry["call_id"] = call_id
            entry["reserved_at"] = reserved_at
            payload["reserved_usd"] = canonical_decimal(already + lifecycle.ceiling_usd)
            payload["remaining_usd"] = canonical_decimal(
                HARD_CAP_USD - already - lifecycle.ceiling_usd
            )
            self._write(payload)
            return dict(event)

    def complete(
        self,
        call_id: str,
        *,
        completed_at: str,
        actual_seconds: int,
        duration_provenance: str = "client_observed_monotonic_ceiling_seconds",
    ) -> dict[str, Any]:
        if not isinstance(call_id, str) or _SAFE_ID.fullmatch(call_id) is None:
            raise ModalL4ContractError("call_id is not a safe stable identity")
        _require_timestamp(completed_at, field="completed_at")
        if (
            isinstance(actual_seconds, bool)
            or not isinstance(actual_seconds, int)
            or actual_seconds < 0
        ):
            raise ModalL4ContractError("actual_seconds must be a non-negative integer")
        if not isinstance(duration_provenance, str) or not duration_provenance:
            raise ModalL4ContractError("duration_provenance must be a non-empty string")
        with _locked(self.lock_path):
            payload = self._read()
            matches = [
                item for item in payload["entries"] if item["call_id"] == call_id
            ]
            if len(matches) != 1:
                raise ModalL4ContractError("completion call_id is unknown")
            entry = matches[0]
            if entry["status"] != "reserved":
                raise ModalL4ContractError("completion call_id is not reservable")
            if actual_seconds > entry["planned_seconds"]:
                raise ModalL4ContractError(
                    "actual_seconds exceeds the planned lifecycle ceiling"
                )
            stage = STAGE_BY_ID[entry["stage_id"]]
            event = self._append(
                payload,
                {
                    "event_type": "complete",
                    "call_id": call_id,
                    "completed_at": completed_at,
                    "actual_seconds": actual_seconds,
                    "duration_provenance": duration_provenance,
                    "actual_cost_usd": canonical_decimal(
                        _cost(actual_seconds, stage.rate_usd_per_second)
                    ),
                },
            )
            entry["status"] = "completed"
            entry["completed_at"] = completed_at
            entry["actual_seconds"] = actual_seconds
            entry["actual_cost_usd"] = event["actual_cost_usd"]
            entry["observed_duration_provenance"] = duration_provenance
            self._write(payload)
            return dict(event)

    def abort(self, call_id: str, *, aborted_at: str, reason: str) -> dict[str, Any]:
        if not isinstance(call_id, str) or _SAFE_ID.fullmatch(call_id) is None:
            raise ModalL4ContractError("call_id is not a safe stable identity")
        _require_timestamp(aborted_at, field="aborted_at")
        if not isinstance(reason, str) or not reason:
            raise ModalL4ContractError("abort reason must be a non-empty string")
        with _locked(self.lock_path):
            payload = self._read()
            matches = [
                item for item in payload["entries"] if item["call_id"] == call_id
            ]
            if len(matches) != 1:
                raise ModalL4ContractError("abort call_id is unknown")
            entry = matches[0]
            if entry["status"] != "reserved":
                raise ModalL4ContractError("abort call_id is not reservable")
            event = self._append(
                payload,
                {
                    "event_type": "abort",
                    "call_id": call_id,
                    "aborted_at": aborted_at,
                    "abort_reason": reason,
                },
            )
            entry["status"] = "aborted"
            entry["aborted_at"] = aborted_at
            entry["abort_reason"] = reason
            self._write(payload)
            return dict(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-modal-l4-crossover",
        description=(
            "Render or verify the offline Modal L4 crossover protocol delta. "
            "Every action is offline: no Modal SDK import, authentication, "
            "container, GPU, or spend."
        ),
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="action")
    plan = subparsers.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--output", type=Path)
    verify = subparsers.add_parser("verify-plan", allow_abbrev=False)
    verify.add_argument("--plan", required=True, type=Path)
    attestation = subparsers.add_parser(
        "attestation-template",
        allow_abbrev=False,
        help=(
            "Print a boolean-only cleared credential-exposure attestation "
            "template. This is not a signed execution authorization and reads "
            "no credential or profile."
        ),
    )
    attestation.add_argument("--confirmed-at", required=True)
    attestation.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = args.action or "plan"
    try:
        assert_provider_sdk_absent()
        if action == "plan":
            rendered = (
                json.dumps(
                    offline_plan_document(),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            if getattr(args, "output", None) is not None:
                atomic_write_text(args.output, rendered)
            print(rendered, end="")
            return 0
        if action == "attestation-template":
            rendered = (
                json.dumps(
                    build_credential_exposure_attestation(
                        confirmed_at=args.confirmed_at
                    ),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            if getattr(args, "output", None) is not None:
                atomic_write_text(args.output, rendered)
            print(rendered, end="")
            return 0
        print(ModalL4Plan.read_json(args.plan).content_sha256)
        return 0
    except (OSError, ValueError, ModalL4ContractError) as exc:
        print(f"llmtracefx-modal-l4-crossover: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
