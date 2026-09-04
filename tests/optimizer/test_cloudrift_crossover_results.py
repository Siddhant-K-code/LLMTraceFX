"""Completed-run analysis and public evidence tests."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import sys
from pathlib import Path

import pytest

from llmtracefx.evidence import core as evidence_core
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover_results as results
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as core

ROOT = Path(__file__).resolve().parents[2]


def _typed(
    value: float | None,
    *,
    unit: str = "seconds",
    clock_domain: str = "same_process_perf_counter",
    provenance: str = "measured_perf_counter_ns",
    state: str | None = None,
    reason: str = "not_exposed",
) -> dict:
    observed = value is not None
    return {
        "value": value,
        "unit": unit,
        "clock_domain": clock_domain,
        "provenance": provenance,
        "observability_state": state or ("observed" if observed else "unobservable"),
        "null_reason": None if observed else reason,
    }


def _authorization(plan: core.VLLMCompilePlan) -> dict:
    authorization = {
        "schema_version": "1",
        "protocol_id": core.PROTOCOL_ID,
        "provider": "CloudRift",
        "approved": True,
        "plan_sha256": plan.content_sha256,
        "source_head": "a" * 40,
        "runtime_image_id": core.DERIVED_IMAGE_ID,
        "experiment_nonce": "c" * 32,
        "workspace_sha256": "sha256:" + "3" * 64,
        "authorized_at": "2026-09-04T09:00:00+00:00",
        "billing_started_at": "2026-09-04T09:00:00+00:00",
        "scheduled_shutdown_at": "2026-09-04T14:28:00+00:00",
        "rate_usd_per_hour": "0.39",
        "hard_cap_usd": "3",
        "automatic_retries": 0,
        "provider_access_managed_externally": True,
    }
    authorization["authorization_sha256"] = results._sha256_json(authorization)
    return authorization


def _write_orchestration(path: Path, value: dict) -> None:
    value.pop("orchestration_sha256", None)
    value["orchestration_sha256"] = results._sha256_json(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_ledger(workspace: Path, plan: core.VLLMCompilePlan) -> dict:
    ledger = core.LifecycleBudgetLedger.initialize(
        workspace / "budget-ledger.json",
        plan=plan,
        git_head="a" * 40,
        workspace_path=workspace,
    )
    for index, lifecycle in enumerate(plan.budget_lifecycles, start=1):
        command = f"command-{index:03d}"
        ledger.reserve(
            command,
            line_id=lifecycle.line_id,
            lifecycle_id=lifecycle.lifecycle_id,
            ceiling_usd=lifecycle.ceiling_usd,
            argv=("synthetic-test-command", lifecycle.lifecycle_id),
            reserved_at="2026-09-04T09:00:00+00:00",
        )
        ledger.complete(
            command,
            completed_at="2026-09-04T09:00:01+00:00",
            actual_seconds=1,
        )
    return ledger.snapshot()


def _optional(value: float | None, mode: str) -> dict:
    if mode == "eager":
        return _typed(
            None,
            clock_domain="vllm_internal_runtime",
            provenance="version_pinned_vllm_0_28_internal",
            state="not_applicable",
            reason="not_applicable_eager_mode",
        )
    return _typed(
        value,
        clock_domain="vllm_internal_runtime",
        provenance="version_pinned_vllm_0_28_internal",
        reason="not_exposed",
    )


def _request(
    cell: core.ScheduleCell,
    descriptor: core.WorkloadDescriptor,
    index: int,
    *,
    cumulative_ns: int,
    output_ids: list[int],
    decoded: str | None,
) -> dict:
    latency_ns = 100_000_000
    latency = latency_ns / 1_000_000_000
    cycle = (index - 1) // 12 + 1
    base = (index - 1) % 12 + 1
    request = {
        **descriptor.to_dict(),
        "cycle_index": cycle,
        "base_ordinal": base,
        "request_sequence_index": index,
        "cell_id": cell.cell_id,
        "pair_id": cell.pair_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "input_token_count": 20 + base,
        "input_token_ids_sha256": "sha256:" + f"{base:064x}",
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": core.token_ids_sha256(output_ids),
        "finish_reason": "length" if cell.lane == "controlled" else "stop",
        "timing": {
            "latency_seconds": _typed(latency),
            "cumulative_from_initialization_seconds": _typed(
                cumulative_ns / 1_000_000_000
            ),
            "latency_perf_counter_ns": latency_ns,
            "cumulative_from_initialization_perf_counter_ns": cumulative_ns,
            "output_token_rate_tokens_per_second": _typed(
                len(output_ids) / latency,
                unit="tokens_per_second",
                provenance="derived_exact_token_count_over_perf_counter_latency",
            ),
        },
        "metrics": {
            "ttft_seconds": _typed(
                None,
                clock_domain="request_output_metrics",
                provenance="version_pinned_vllm_0_28_request_state_stats",
                reason="request_state_stats_first_token_latency_unavailable",
            ),
            **{
                name: _typed(
                    None,
                    clock_domain="request_output_metrics",
                    provenance="version_pinned_vllm_0_28_request_state_stats",
                    reason=reason,
                )
                for name, reason in {
                    "queue_seconds": (
                        "request_state_stats_has_no_queue_duration_field"
                    ),
                    "prefill_seconds": (
                        "request_state_stats_has_no_prefill_duration_field"
                    ),
                    "inference_seconds": (
                        "request_state_stats_has_no_inference_duration_field"
                    ),
                    "decode_seconds": (
                        "request_state_stats_has_no_decode_duration_field"
                    ),
                    "mean_time_per_output_token_seconds": (
                        "request_state_stats_has_no_mean_output_token_duration_field"
                    ),
                    "e2e_seconds": ("request_state_stats_has_no_e2e_duration_field"),
                }.items()
            },
        },
        "terminal": True,
    }
    if decoded is not None:
        request["decoded_output"] = decoded
    return request


def _cell_receipt(
    cell: core.ScheduleCell,
    plan: core.VLLMCompilePlan,
    *,
    censored: bool,
) -> dict:
    descriptors = core.lane_request_descriptors(cell.lane)
    requests = []
    for index, descriptor in enumerate(descriptors, start=1):
        if cell.lane == "controlled":
            output_ids = [1000 + descriptor.ordinal] * 96
            decoded = None
        else:
            output_ids = [2000 + descriptor.ordinal, 3]
            decoded = (
                '{"name":"Priya Nakamura","age":34,"is_active":true}'
                if descriptor.workload_id == "structured-json-profile-extraction"
                else "3 hours because the combined closing speed is 70 mph."
            )
        eager_ns = 1_000_000_000 + index * 200_000_000
        if cell.mode == "eager":
            cumulative_ns = eager_ns
        elif censored:
            cumulative_ns = eager_ns + 500_000_000
        else:
            cumulative_ns = 2_000_000_000 + index * 100_000_000
        requests.append(
            _request(
                cell,
                descriptor,
                index,
                cumulative_ns=cumulative_ns,
                output_ids=output_ids,
                decoded=decoded,
            )
        )
    resolved = (
        {
            "enforce_eager": True,
            "compilation_mode": "NONE",
            "cuda_graph_mode": "NONE",
        }
        if cell.mode == "eager"
        else {
            "enforce_eager": False,
            "compilation_mode": "VLLM_COMPILE",
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
        }
    )
    payload = {
        "schema_version": "2",
        "protocol_id": core.PROTOCOL_ID,
        "cell": cell.to_dict(),
        "plan_sha256": plan.content_sha256,
        "analysis_seed": core.ANALYSIS_SEED,
        "model": {
            "id": core.MODEL_ID,
            "revision": core.MODEL_REVISION,
            "expected_file_count": core.EXPECTED_MODEL_FILE_COUNT,
            "expected_bytes": core.EXPECTED_MODEL_BYTES,
            "state_receipt": "staging-receipt.json",
            "prompt_receipt": "prompt-token-ids.json",
        },
        "budget": {"hard_cap_usd": "3.00"},
        "runtime": {
            "pins": dict(core.RUNTIME_PINS),
            "expected_pins": dict(core.RUNTIME_PINS),
            "runtime_image": {
                "base_reference": core.BASE_IMAGE_REFERENCE,
                "derived_image_id": core.DERIVED_IMAGE_ID,
            },
            "resolved_execution_config": resolved,
            "optional_version_pinned_fields": {
                "compiled_mode_expected": cell.mode == "compiled",
                "compilation_config_fields": {
                    "value": None,
                    "unit": "json",
                    "clock_domain": "resolved_runtime_config",
                    "provenance": "version_pinned_vllm_0_28_internal",
                    "observability_state": "unobservable",
                    "null_reason": "optional_compilation_fields_not_exposed",
                },
                "encoder_compilation_config": {
                    "field_name": "encoder_compilation_config",
                    "value": None,
                    "unit": "json",
                    "clock_domain": "resolved_runtime_config",
                    "provenance": "version_pinned_vllm_0_28_internal",
                    "observability_state": "unobservable",
                    "null_reason": "encoder_compilation_config_not_exposed",
                },
                "compilation_time_seconds": _optional(0.25, cell.mode),
                "encoder_compilation_time_seconds": _optional(0.1, cell.mode),
                "cuda_graph_capture_duration_seconds": _typed(
                    None,
                    clock_domain="vllm_internal_runtime",
                    provenance="version_pinned_vllm_0_28_internal",
                    state=(
                        "unobservable" if cell.mode == "compiled" else "not_applicable"
                    ),
                    reason=(
                        "cuda_graph_capture_duration_not_exposed_by_vllm"
                        if cell.mode == "compiled"
                        else "not_applicable_eager_mode"
                    ),
                ),
                "cuda_graph_dispatch_counter": _typed(
                    None,
                    unit="requests",
                    clock_domain="vllm_metrics_registry",
                    provenance="documented_vllm_0_28_metric",
                    state=(
                        "unobservable" if cell.mode == "compiled" else "not_applicable"
                    ),
                    reason=(
                        "offline_llm_has_no_stable_cuda_graph_dispatch_metric_snapshot_hook"
                        if cell.mode == "compiled"
                        else "not_applicable_eager_mode"
                    ),
                ),
            },
        },
        "deterministic_environment": {
            "variables": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "HF_HUB_OFFLINE": "1",
                "PYTHONHASHSEED": str(core.SAMPLING_SEED),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "VLLM_BATCH_INVARIANT": "0",
                "VLLM_NO_USAGE_STATS": "1",
            },
            "cache_root_role": {
                "relative_identity": cell.cell_id,
                "path_sha256": "sha256:" + "d" * 64,
            },
            "cache_roles": {
                role: {
                    "env_var": env_var,
                    "relative_path": role,
                    "path_sha256": "sha256:" + character * 64,
                }
                for role, env_var, character in (
                    ("vllm", "VLLM_CACHE_ROOT", "2"),
                    ("torchinductor", "TORCHINDUCTOR_CACHE_DIR", "3"),
                    ("triton", "TRITON_CACHE_DIR", "4"),
                    ("cuda", "CUDA_CACHE_PATH", "5"),
                    ("home", "HOME", "6"),
                    ("huggingface", "HF_HOME", "7"),
                    ("xdg", "XDG_CACHE_HOME", "8"),
                )
            },
        },
        "hardware_commitment": {
            "gpu_name": core.EXPECTED_GPU_NAME,
            "gpu_count": 1,
            "driver_version": core.EXPECTED_DRIVER,
            "memory_total_mib": core.EXPECTED_MEMORY_MIB,
            "memory_used_mib": 1024,
            "public_experiment_nonce": "c" * 32,
            "gpu_identity_commitment": "sha256:" + "e" * 64,
        },
        "process_tree": {
            "nodes": [
                {
                    "node_index": 0,
                    "parent_node_index": None,
                    "process_name": "python3",
                }
            ],
            "clock_domain": "runner_process_snapshot",
            "provenance": "linux_procfs_stat",
            "observability_state": "observed",
            "null_reason": None,
        },
        "measurements": {
            "initialization_seconds": _typed(1.0),
            "initialization_perf_counter_ns": 1_000_000_000,
            "peak_gpu_memory_mib": _typed(
                2048.0,
                unit="MiB",
                clock_domain="sampled_nvidia_smi",
                provenance="sampled_nvidia_smi",
            ),
            "gpu_memory_series": {
                "value": [
                    {"offset_ns": 100_000_000, "memory_used_mib": 1024},
                    {"offset_ns": 300_000_000, "memory_used_mib": 2048},
                ],
                "unit": "MiB",
                "clock_domain": "same_process_perf_counter_offset_ns",
                "provenance": "sampled_nvidia_smi",
                "observability_state": "observed",
                "null_reason": None,
                "target_interval_ms": 200,
                "sampling_error_count": 0,
                "sampling_error_types": [],
            },
        },
        "request_count_expected": cell.requests_per_cell,
        "request_count_observed": cell.requests_per_cell,
        "prompt_ids_sha256": "sha256:" + "f" * 64,
        "staging_receipt_sha256": "sha256:" + "1" * 64,
        "requests": requests,
        "terminal": True,
    }
    payload["cell_sha256"] = results._sha256_json(payload)
    return payload


def _write_progress(raw: Path, cell: core.ScheduleCell, requests: list[dict]) -> None:
    progress = {
        "schema_version": "2",
        "protocol_id": core.PROTOCOL_ID,
        "cell_id": cell.cell_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "request_count_expected": cell.requests_per_cell,
        "request_count_completed": cell.requests_per_cell,
        "last_request_sequence_index": cell.requests_per_cell,
        "requests": requests,
        "terminal": False,
    }
    progress["progress_sha256"] = results._sha256_json(progress)
    (raw / f".{cell.cell_id}-progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )


def _sync_progress(cell_path: Path, receipt: dict) -> None:
    cell_id = receipt["cell"]["cell_id"]
    cell = next(
        item for item in core.build_default_plan().schedule if item.cell_id == cell_id
    )
    _write_progress(cell_path.parent, cell, receipt["requests"])


def _workspace(tmp_path: Path, *, censored: bool = False) -> Path:
    workspace = tmp_path / "workspace"
    raw = workspace / "raw"
    raw.mkdir(parents=True)
    plan = core.build_default_plan()
    authorization = _authorization(plan)
    (workspace / "authorization.json").write_text(
        json.dumps(authorization), encoding="utf-8"
    )
    ledger = _complete_ledger(workspace, plan)
    ledger_bytes = (workspace / "budget-ledger.json").read_bytes()
    operation_receipts = []
    for index, entry in enumerate(ledger["entries"], start=1):
        started_ns = index * 2_000_000_000
        operation_receipts.append(
            {
                "command_id": entry["command_id"],
                "lifecycle_id": entry["lifecycle_id"],
                "line_id": entry["line_id"],
                "clock_domain": "host_perf_counter",
                "started_ns": started_ns,
                "ended_ns": started_ns + 1_000_000_000,
                "duration_ns": 1_000_000_000,
                "status": "completed",
            }
        )
    observation_ids = ["preflight-after-reset"]
    for cell in plan.schedule:
        observation_ids.extend(
            (
                f"{cell.cell_id}-before-container",
                f"{cell.cell_id}-after-container",
            )
        )
    hardware_observations = [
        {
            "observation_id": observation_id,
            "clock_domain": "host_perf_counter",
            "host_perf_counter_ns": index * 1_000_000,
            "gpu_identity_commitment": "sha256:" + "e" * 64,
            "gpu_name": core.EXPECTED_GPU_NAME,
            "driver_version": core.EXPECTED_DRIVER,
            "memory_total_mib": core.EXPECTED_MEMORY_MIB,
            "memory_used_mib": 256,
            "temperature_c": 40,
            "utilization_percent": 0,
            "power_limit_watts": 450.0,
            "sm_clock_mhz": 210,
            "compute_capability": "8.9",
        }
        for index, observation_id in enumerate(observation_ids, start=1)
    ]
    orchestration = {
        "schema_version": "1",
        "protocol_id": core.PROTOCOL_ID,
        "plan_sha256": plan.content_sha256,
        "source_head": "a" * 40,
        "runtime_image_id": core.DERIVED_IMAGE_ID,
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_authentication": {
            "mechanism": "openssh_detached_signature",
            "namespace": "llmtracefx-vllm-crossover-authorization-v1",
            "signer_identity": "vllm-crossover-coordinator",
            "signature_sha256": "sha256:" + "4" * 64,
            "authorized_signers_sha256": "sha256:" + "5" * 64,
            "verified": True,
        },
        "scheduled_shutdown_at": authorization["scheduled_shutdown_at"],
        "repository_path_sha256": "sha256:" + "2" * 64,
        "workspace_path_sha256": "sha256:" + "3" * 64,
        "completed_cell_ids": [cell.cell_id for cell in plan.schedule],
        "operation_receipts": operation_receipts,
        "hardware_observations": hardware_observations,
        "ledger_abort_failures": [],
        "status": "complete",
        "failure": None,
        "teardown_status": "local_cleanup_complete",
        "host_shutdown_observed_at": None,
        "host_shutdown_observed_null_reason": (
            "the local process cannot observe its own later host shutdown"
        ),
        "external_provider_console_confirmation": None,
        "external_provider_console_confirmation_null_reason": (
            "external operator confirmation was not supplied to the local runner"
        ),
        "independently_verified_provider_termination": None,
        "independently_verified_provider_termination_null_reason": (
            "no provider API receipt is available"
        ),
        "provider_teardown": None,
        "provider_teardown_null_reason": (
            "provider teardown remains externally user-confirmed"
        ),
        "ledger_sha256": results._sha256_uri(ledger_bytes),
    }
    _write_orchestration(workspace / "orchestration-receipt.json", orchestration)
    for cell in plan.schedule:
        receipt = _cell_receipt(cell, plan, censored=censored)
        (raw / f"{cell.cell_id}.json").write_text(json.dumps(receipt), encoding="utf-8")
        _write_progress(raw, cell, receipt["requests"])
    return workspace


def _mutate_request(
    workspace: Path,
    *,
    lane: str,
    mode: str,
    field: str,
    value: object,
) -> None:
    plan = core.build_default_plan()
    cell = next(
        item for item in plan.schedule if item.lane == lane and item.mode == mode
    )
    path = workspace / "raw" / f"{cell.cell_id}.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["requests"][0][field] = value
    if field == "output_token_ids":
        receipt["requests"][0]["output_token_count"] = len(value)  # type: ignore[arg-type]
        receipt["requests"][0]["output_token_ids_sha256"] = core.token_ids_sha256(
            value  # type: ignore[arg-type]
        )
        latency = receipt["requests"][0]["timing"]["latency_seconds"]["value"]
        output_rate = len(value) / latency  # type: ignore[arg-type]
        receipt["requests"][0]["timing"]["output_token_rate_tokens_per_second"][
            "value"
        ] = output_rate
    receipt.pop("cell_sha256")
    receipt["cell_sha256"] = results._sha256_json(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    _sync_progress(path, receipt)


@pytest.fixture(autouse=True)
def _fast_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(results, "BOOTSTRAP_RESAMPLES", 40)


def test_completed_catalog_adapter_invokes_trusted_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, arguments = evidence_core.SCRIPT_ADAPTERS["vllm_crossover_results_v1"]
    observed: list[Path] = []
    monkeypatch.setattr(results, "verify_bundle", observed.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(ROOT / script),
            *(str(tmp_path) if value == "{bundle}" else value for value in arguments),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / script), run_name="__main__")

    assert exit_info.value.code == 0
    assert observed == [tmp_path]


def test_successful_bundle_is_deterministic_and_verifies(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    results.build_bundle(workspace, first)
    results.build_bundle(workspace, second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    results.verify_bundle(first)
    runpy.run_path(str(first / "evidence_bundle.py"))["verify"](first)
    assert "BOOTSTRAP_RESAMPLES = 40" in (first / "evidence_bundle.py").read_text(
        encoding="utf-8"
    )
    protocol = json.loads((first / "protocol.json").read_text(encoding="utf-8"))
    plan_document = core.build_default_plan().to_dict()
    assert protocol["analysis"]["bootstrap_resamples"] == 40
    assert protocol["analysis"]["request_level_resampling"] is False
    assert (
        protocol["sample_stopping_rule"]
        == core.build_default_plan().to_dict()["sample_stopping_rule"]
    )
    assert protocol["quality_preservation"]["resamples"] == 20_000
    assert protocol["quality_preservation"]["executed_resamples"] == 40
    assert protocol["hardware_observations"]["observation_count"] == 65
    assert protocol["bindings_verified"]["operation_receipts"] is True
    assert protocol["authorization_authentication"] == {
        "mechanism": "openssh_detached_signature",
        "namespace": "llmtracefx-vllm-crossover-authorization-v1",
        "signer_identity": "vllm-crossover-coordinator",
        "signature_sha256": "sha256:" + "4" * 64,
        "authorized_signers_sha256": "sha256:" + "5" * 64,
        "verified": True,
    }
    for field in (
        "execution_modes",
        "lifecycle_controls",
        "measurement_contract",
        "reproducibility",
    ):
        assert protocol[field] == plan_document[field]
    pairs = json.loads((first / "lifecycle-pairs.json").read_text(encoding="utf-8"))
    first_effects = pairs["pairs"][0]["pair_effects"]
    assert set(first_effects) == {
        "initialization",
        "host_lifecycle",
        "request_phase",
        "cumulative_init_to_terminal",
        "mean_ttft",
        "mean_prefill",
        "mean_decode",
        "mean_output_rate",
        "peak_gpu_memory",
    }
    assert first_effects["host_lifecycle"]["value"] == 0.0
    assert first_effects["host_lifecycle"]["unit"] == "seconds"
    assert first_effects["mean_ttft"]["value"] is None
    assert (
        first_effects["mean_ttft"]["null_reason"]
        == "not_all_requests_observed_in_both_cells"
    )
    assert first_effects["mean_prefill"]["value"] is None
    assert first_effects["mean_decode"]["value"] is None
    assert first_effects["mean_output_rate"]["null_reason"] is None
    request_records = [
        json.loads(line)
        for line in (first / "request-records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    eager_id = pairs["pairs"][0]["eager"]["cell_id"]
    compiled_id = pairs["pairs"][0]["compiled"]["cell_id"]
    eager_requests = [
        request for request in request_records if request["cell_id"] == eager_id
    ]
    compiled_requests = [
        request for request in request_records if request["cell_id"] == compiled_id
    ]
    eager_requests[0]["timing"]["output_token_rate_tokens_per_second"].update(
        {
            "value": None,
            "observability_state": "unobservable",
            "null_reason": "test_unobserved_rate",
        }
    )
    all_or_nothing_effects = results._compute_pair_effects(
        pairs["pairs"][0]["eager"],
        pairs["pairs"][0]["compiled"],
        eager_requests,
        compiled_requests,
    )
    assert all_or_nothing_effects["mean_output_rate"]["value"] is None
    assert (
        all_or_nothing_effects["mean_output_rate"]["null_reason"]
        == "not_all_requests_observed_in_both_cells"
    )
    memory_series = pairs["pairs"][0]["compiled"]["measurements"]["gpu_memory_series"]
    assert memory_series["clock_domain"] == "same_process_perf_counter_offset_ns"
    assert memory_series["value"][-1]["memory_used_mib"] == 2048
    assert pairs["pairs"][0]["compiled"]["process_tree"] == {
        "nodes": [
            {
                "node_index": 0,
                "parent_node_index": None,
                "process_name": "python3",
            }
        ],
        "clock_domain": "runner_process_snapshot",
        "provenance": "linux_procfs_stat",
        "observability_state": "observed",
        "null_reason": None,
    }
    report = (first / "report.html").read_text(encoding="utf-8")
    assert "Supported sustained crossing:" in report
    assert "Terminal sign-symmetry p-value:" in report
    assert "Natural timing mean compiled-minus-eager terminal effect:" in report
    assert "95% whole-pair percentile interval" in report
    assert "Causal eligibility:" in report
    assert "8 pairs per lane" in report
    assert "intervals may under-cover" in report
    provenance = json.loads(
        (first / "provenance-null-matrix.json").read_text(encoding="utf-8")
    )
    provenance_by_field = {row["field"]: row for row in provenance["fields"]}
    assert provenance_by_field["sampled_gpu_memory_series"] == {
        "field": "sampled_gpu_memory_series",
        "provenance": "sampled_nvidia_smi",
        "value_state": "observed",
        "null_reason": None,
    }
    assert provenance_by_field["process_tree"] == {
        "field": "process_tree",
        "provenance": "linux_procfs_stat",
        "value_state": "observed",
        "null_reason": None,
    }
    assert provenance_by_field["cuda_graph_dispatch_counter"] == {
        "field": "cuda_graph_dispatch_counter",
        "provenance": "documented_vllm_0_28_metric",
        "value_state": "null",
        "null_reason": (
            "offline_llm_has_no_stable_cuda_graph_dispatch_metric_snapshot_hook"
        ),
    }
    assert (
        provenance_by_field["active_operation_list_rate_equivalent"]["value_state"]
        == "observed"
    )
    for field in (
        "provider_billed_seconds",
        "provider_reported_spend",
        "provider_list_rate_cost",
        "actual_cost",
    ):
        assert provenance_by_field[field]["value_state"] == "null"
        assert (
            provenance_by_field[field]["null_reason"]
            == "external_provider_end_receipt_absent"
        )
    dispatch_counter = pairs["pairs"][0]["compiled"]["compile_component_measurements"][
        "cuda_graph_dispatch_counter"
    ]
    assert dispatch_counter == {
        "value": None,
        "unit": "requests",
        "clock_domain": "vllm_metrics_registry",
        "provenance": "documented_vllm_0_28_metric",
        "observability_state": "unobservable",
        "null_reason": (
            "offline_llm_has_no_stable_cuda_graph_dispatch_metric_snapshot_hook"
        ),
    }
    claims = json.loads((first / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["fixed-token-count-crossover"]["state"] == "supported"
    assert by_id["output-identical-generation-crossover"]["state"] == "supported"
    assert by_id["budget-reservations-within-hard-cap"]["state"] == "supported"
    assert (
        by_id["active-operation-list-rate-equivalent-within-hard-cap"]["state"]
        == "supported"
    )
    assert by_id["provider-billed-cost-within-hard-cap"] == {
        "claim_id": "provider-billed-cost-within-hard-cap",
        "state": "unsupported",
        "blockers": ["external_provider_end_receipt_absent"],
    }
    budget = json.loads((first / "budget-teardown.json").read_text(encoding="utf-8"))
    assert "within_hard_cap" not in budget
    assert budget["reservations_within_hard_cap"] is True
    assert budget["active_operation_equivalent_within_hard_cap"] is True
    assert budget["active_operation_list_rate_equivalent_usd"] is not None
    for value_field, reason_field in (
        ("provider_billed_seconds", "provider_billed_seconds_null_reason"),
        ("provider_reported_spend_usd", "provider_reported_spend_null_reason"),
        ("provider_list_rate_cost_usd", "provider_list_rate_cost_null_reason"),
        ("actual_cost_usd", "actual_cost_null_reason"),
    ):
        assert budget[value_field] is None
        assert reason_field in budget
        assert budget[reason_field] == "external_provider_end_receipt_absent"
    teardown_domains = {
        "host_shutdown_observed_at": (
            "host_shutdown_observed_null_reason",
            "the local process cannot observe its own later host shutdown",
        ),
        "external_provider_console_confirmation": (
            "external_provider_console_confirmation_null_reason",
            "external operator confirmation was not supplied to the local runner",
        ),
        "independently_verified_provider_termination": (
            "independently_verified_provider_termination_null_reason",
            "no provider API receipt is available",
        ),
    }
    for value_field, (reason_field, reason) in teardown_domains.items():
        assert budget[value_field] is None
        assert budget[reason_field] == reason
        assert provenance_by_field[value_field]["value_state"] == "null"
        assert provenance_by_field[value_field]["null_reason"] == reason
    analysis = json.loads((first / "analysis.json").read_text(encoding="utf-8"))
    distributions = analysis["pair_effect_distributions"]
    assert distributions["request_level_resampling"] is False
    for lane in ("controlled", "natural"):
        for metric, summary in distributions["lanes"][lane].items():
            assert summary["pair_count"] == 8
            assert len(summary["effects"]) == 8
            if metric in {"mean_ttft", "mean_prefill", "mean_decode"}:
                assert summary["observed_effect_count"] == 0
                assert summary["mean"] is None
                assert summary["summary_null_reason"] == "no_observed_pair_effects"


def test_bundle_tamper_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    (bundle / "report.html").write_text("tampered", encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="SHA256SUMS"):
        results.verify_bundle(bundle)


def _reseal_bundle(bundle: Path) -> None:
    sums = "\n".join(
        f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}"
        for name in results.HASHED_FILES
    )
    (bundle / "SHA256SUMS").write_text(sums + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    "artifact",
    [
        "report.html",
        "crossover.svg",
        "provenance-null-matrix.json",
        "evidence_bundle.py",
    ],
)
def test_resealed_derived_artifact_tamper_is_rejected(
    tmp_path: Path, artifact: str
) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    path = bundle / artifact
    if artifact == "provenance-null-matrix.json":
        document = json.loads(path.read_text(encoding="utf-8"))
        document["fields"][0]["provenance"] = "tampered"
        path.write_text(results._json_text(document), encoding="utf-8")
    else:
        path.write_text(
            path.read_text(encoding="utf-8") + "\n<!-- tampered -->\n",
            encoding="utf-8",
        )
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match=re.escape(artifact)):
        results.verify_bundle(bundle)
    if artifact != "evidence_bundle.py":
        standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
        with pytest.raises(ValueError, match=re.escape(artifact)):
            standalone_verify(bundle)


@pytest.mark.parametrize("tamper", ["mode_slot_swap", "cross_pair_cell_swap"])
def test_resealed_pair_role_swap_is_rejected(tmp_path: Path, tamper: str) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    pairs_path = bundle / "lifecycle-pairs.json"
    document = json.loads(pairs_path.read_text(encoding="utf-8"))
    pair_records = document["pairs"]
    if tamper == "mode_slot_swap":
        pair_records[0]["eager"], pair_records[0]["compiled"] = (
            pair_records[0]["compiled"],
            pair_records[0]["eager"],
        )
    else:
        pair_records[0]["eager"], pair_records[1]["eager"] = (
            pair_records[1]["eager"],
            pair_records[0]["eager"],
        )
    pairs_path.write_text(results._json_text(document), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="canonical binding"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="canonical binding"):
        standalone_verify(bundle)


@pytest.mark.parametrize("tamper", ["decoded_output", "success"])
def test_semantic_correctness_tamper_is_rejected_after_reseal(
    tmp_path: Path, tamper: str
) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    requests_path = bundle / "request-records.jsonl"
    requests = [
        json.loads(line)
        for line in requests_path.read_text(encoding="utf-8").splitlines()
    ]
    natural = next(item for item in requests if item["lane"] == "natural")
    if tamper == "decoded_output":
        natural["decoded_output"] = "semantically invalid output"
    else:
        natural["correctness"]["success"] = not natural["correctness"]["success"]
        correctness_path = bundle / "correctness.json"
        correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
        correctness["evaluations"][0]["success"] = not correctness["evaluations"][0][
            "success"
        ]
        correctness_path.write_text(results._json_text(correctness), encoding="utf-8")
    requests_path.write_text(results._jsonl_text(requests), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="correctness"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="correctness"):
        standalone_verify(bundle)


@pytest.mark.parametrize("tamper", ["direct_scalar", "null_metric"])
def test_pair_effect_tamper_is_rejected_after_reseal(
    tmp_path: Path, tamper: str
) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    pairs_path = bundle / "lifecycle-pairs.json"
    pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    effects = pairs["pairs"][0]["pair_effects"]
    if tamper == "direct_scalar":
        effects["initialization"]["value"] += 1.0
    else:
        effects["mean_prefill"]["value"] = 0.0
        effects["mean_prefill"]["null_reason"] = None
    pairs_path.write_text(results._json_text(pairs), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="pair effect"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="pair effect"):
        standalone_verify(bundle)


def test_resealed_bootstrap_count_downgrade_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    protocol_path = bundle / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["analysis"]["bootstrap_resamples"] = 1
    protocol["quality_preservation"]["executed_resamples"] = 1
    protocol_path.write_text(results._json_text(protocol), encoding="utf-8")
    analysis_path = bundle / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["controlled"]["resample_count"] = 1
    analysis_path.write_text(results._json_text(analysis), encoding="utf-8")
    correctness_path = bundle / "correctness.json"
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    correctness["quality_preservation"]["resample_count"] = 1
    correctness_path.write_text(results._json_text(correctness), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="protocol"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="bootstrap execution count"):
        standalone_verify(bundle)


@pytest.mark.parametrize("forgery", ["controlled_derived", "natural_difference"])
def test_resealed_pair_curve_forgery_is_rejected(tmp_path: Path, forgery: str) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path), bundle)
    pairs_path = bundle / "lifecycle-pairs.json"
    pair_document = json.loads(pairs_path.read_text(encoding="utf-8"))
    pair_records = pair_document["pairs"]
    if forgery == "controlled_derived":
        pair = next(item for item in pair_records if item["lane"] == "controlled")
        pair["compiled"]["cumulative_seconds"] = [
            value + 0.25 for value in pair["compiled"]["cumulative_seconds"]
        ]
        pair["compiled_minus_eager_seconds"] = [
            compiled - eager
            for eager, compiled in zip(
                pair["eager"]["cumulative_seconds"],
                pair["compiled"]["cumulative_seconds"],
                strict=True,
            )
        ]
        requests = [
            json.loads(line)
            for line in (bundle / "request-records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        requests_by_cell: dict[str, list[dict]] = {}
        for request in requests:
            requests_by_cell.setdefault(request["cell_id"], []).append(request)
        pair["pair_effects"] = results._compute_pair_effects(
            pair["eager"],
            pair["compiled"],
            requests_by_cell[pair["eager"]["cell_id"]],
            requests_by_cell[pair["compiled"]["cell_id"]],
        )
        plan = core.build_default_plan()
        controlled_identity = results._identity_summary(
            plan.schedule, requests_by_cell, "controlled"
        )
        natural_identity = results._identity_summary(
            plan.schedule, requests_by_cell, "natural"
        )
        curves = [
            core.PairCurve(
                pair_id=item["pair_id"],
                order=item["order"],
                eager_cumulative=tuple(item["eager"]["cumulative_seconds"]),
                compiled_cumulative=tuple(item["compiled"]["cumulative_seconds"]),
            )
            for item in pair_records
            if item["lane"] == "controlled"
        ]
        analysis = results._analysis_document(
            curves,
            natural_identity=natural_identity["all_corresponding_outputs_identical"],
            natural_terminal_effects=[
                item["compiled_minus_eager_seconds"][-1]
                for item in pair_records
                if item["lane"] == "natural"
            ],
            pair_records=pair_records,
        )
        (bundle / "analysis.json").write_text(
            results._json_text(analysis), encoding="utf-8"
        )
        correctness = json.loads(
            (bundle / "correctness.json").read_text(encoding="utf-8")
        )
        claims = results._claim_matrix_document(
            analysis=analysis,
            controlled_identity=controlled_identity,
            natural_identity=natural_identity,
            natural_all_correct=correctness["natural_all_correct"],
            quality_preservation=correctness["quality_preservation"],
            component_observability=results._component_observability(pair_records),
        )
        (bundle / "claim-matrix.json").write_text(
            results._json_text(claims), encoding="utf-8"
        )
    else:
        pair = next(item for item in pair_records if item["lane"] == "natural")
        pair["compiled_minus_eager_seconds"][-1] += 0.25
    pairs_path.write_text(results._json_text(pair_document), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="curve"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="curve"):
        standalone_verify(bundle)


def test_missing_terminal_cell_is_rejected_before_publication(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    next((workspace / "raw").iterdir()).unlink()
    output = tmp_path / "bundle"
    with pytest.raises(results.CrossoverResultsError, match="inventory"):
        results.build_bundle(workspace, output)
    assert not output.exists()


def test_progress_receipt_seal_and_terminal_equality_are_required(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    cell = core.build_default_plan().schedule[0]
    progress_path = workspace / "raw" / f".{cell.cell_id}-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["request_count_completed"] -= 1
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="progress_sha256"):
        results.build_bundle(workspace, tmp_path / "bad-progress-seal")

    second = _workspace(tmp_path / "second")
    progress_path = second / "raw" / f".{cell.cell_id}-progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["requests"][0]["output_token_ids"][0] += 1
    progress.pop("progress_sha256")
    progress["progress_sha256"] = results._sha256_json(progress)
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="terminal cell"):
        results.build_bundle(second, tmp_path / "progress-terminal-mismatch")


def test_authorization_shutdown_seal_and_orchestration_binding(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    authorization_path = workspace / "authorization.json"
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["scheduled_shutdown_at"] = "2026-09-04T14:29:00+00:00"
    authorization["authorization_sha256"] = results._sha256_json(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
    )
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="scheduled shutdown"):
        results.build_bundle(workspace, tmp_path / "bad-shutdown")

    second = _workspace(tmp_path / "second")
    orchestration_path = second / "orchestration-receipt.json"
    orchestration = json.loads(orchestration_path.read_text(encoding="utf-8"))
    orchestration["authorization_sha256"] = "sha256:" + "9" * 64
    _write_orchestration(orchestration_path, orchestration)
    with pytest.raises(results.CrossoverResultsError, match="orchestration"):
        results.build_bundle(second, tmp_path / "bad-orchestration")


def test_orchestration_seal_is_verified_before_receipt_contents(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    path = workspace / "orchestration-receipt.json"
    orchestration = json.loads(path.read_text(encoding="utf-8"))
    orchestration["status"] = "incomplete"
    path.write_text(json.dumps(orchestration), encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="orchestration_sha256"):
        results.build_bundle(workspace, tmp_path / "bundle")


def test_operation_receipt_and_abort_failures_are_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    path = workspace / "orchestration-receipt.json"
    orchestration = json.loads(path.read_text(encoding="utf-8"))
    orchestration["operation_receipts"][0]["duration_ns"] += 1_000_000_000
    _write_orchestration(path, orchestration)
    with pytest.raises(results.CrossoverResultsError, match="operation duration"):
        results.build_bundle(workspace, tmp_path / "bad-operation")

    second = _workspace(tmp_path / "second")
    second_path = second / "orchestration-receipt.json"
    orchestration = json.loads(second_path.read_text(encoding="utf-8"))
    orchestration["ledger_abort_failures"] = ["LedgerWriteError"]
    _write_orchestration(second_path, orchestration)
    with pytest.raises(results.CrossoverResultsError, match="orchestration"):
        results.build_bundle(second, tmp_path / "bad-abort")


def test_hardware_observation_continuity_and_raw_uuid_privacy(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    path = workspace / "orchestration-receipt.json"
    orchestration = json.loads(path.read_text(encoding="utf-8"))
    orchestration["hardware_observations"][1]["gpu_identity_commitment"] = (
        "sha256:" + "8" * 64
    )
    _write_orchestration(path, orchestration)
    with pytest.raises(results.CrossoverResultsError, match="identity"):
        results.build_bundle(workspace, tmp_path / "bad-hardware")

    second = _workspace(tmp_path / "second")
    plan = core.build_default_plan()
    cell_path = second / "raw" / f"{plan.schedule[0].cell_id}.json"
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    cell["hardware_commitment"]["gpu_identity_commitment"] = "sha256:" + "7" * 64
    cell.pop("cell_sha256")
    cell["cell_sha256"] = results._sha256_json(cell)
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    with pytest.raises(results.CrossoverResultsError, match="host and cell"):
        results.build_bundle(second, tmp_path / "cell-hardware-mismatch")

    third = _workspace(tmp_path / "third")
    third_path = third / "orchestration-receipt.json"
    orchestration = json.loads(third_path.read_text(encoding="utf-8"))
    orchestration["provider_teardown_null_reason"] = "GPU-aaaaaaaa-bbbb-cccc-dddd"
    _write_orchestration(third_path, orchestration)
    with pytest.raises(results.CrossoverResultsError, match="raw GPU UUID"):
        results.build_bundle(third, tmp_path / "gpu-leak")


def test_unequal_natural_output_disables_causal_timing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _mutate_request(
        workspace,
        lane="natural",
        mode="compiled",
        field="output_token_ids",
        value=[999, 3],
    )
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    analysis = json.loads((bundle / "analysis.json").read_text(encoding="utf-8"))
    natural_timing = analysis["controlled"]["natural_timing"]
    assert natural_timing["causal_claim_eligible"] is False
    assert natural_timing[
        "mean_terminal_compiled_minus_eager_seconds"
    ] == pytest.approx(-0.2)
    assert natural_timing["lower_confidence_endpoint_seconds"] == pytest.approx(-0.2)
    assert natural_timing["upper_confidence_endpoint_seconds"] == pytest.approx(-0.2)
    assert natural_timing["speedup_supported"] is False
    assert (
        natural_timing["causal_claim_blocker"]
        == "natural_outputs_differ_across_modes_or_lifecycles"
    )
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["natural-end-to-end-causal-speedup"]["state"] == "unsupported"
    assert (
        "natural_output_identity"
        in by_id["natural-end-to-end-causal-speedup"]["blockers"]
    )


def test_compiled_slower_natural_lane_disables_speedup_claim(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    for cell in core.build_default_plan().schedule:
        if cell.lane != "natural" or cell.mode != "compiled":
            continue
        path = workspace / "raw" / f"{cell.cell_id}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        for request in receipt["requests"]:
            timing = request["timing"]
            timing["cumulative_from_initialization_seconds"]["value"] += 0.5
            timing["cumulative_from_initialization_perf_counter_ns"] += 500_000_000
        receipt.pop("cell_sha256")
        receipt["cell_sha256"] = results._sha256_json(receipt)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        _sync_progress(path, receipt)
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)

    analysis = json.loads((bundle / "analysis.json").read_text(encoding="utf-8"))
    natural_timing = analysis["controlled"]["natural_timing"]
    assert natural_timing["causal_claim_eligible"] is True
    assert natural_timing[
        "mean_terminal_compiled_minus_eager_seconds"
    ] == pytest.approx(0.3)
    assert natural_timing["upper_confidence_endpoint_seconds"] == pytest.approx(0.3)
    assert natural_timing["speedup_supported"] is False
    assert natural_timing["causal_claim_blocker"] is None
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["natural-end-to-end-causal-speedup"] == {
        "claim_id": "natural-end-to-end-causal-speedup",
        "state": "unsupported",
        "blockers": ["natural_supported_speedup"],
    }


def test_controlled_divergence_downgrades_output_identity_claim(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _mutate_request(
        workspace,
        lane="controlled",
        mode="compiled",
        field="output_token_ids",
        value=[777] * 96,
    )
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["output-identical-generation-crossover"]["state"] == "unsupported"
    assert by_id["fixed-token-count-crossover"]["state"] == "supported"


def test_output_identity_claim_also_requires_within_mode_reproducibility(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    plan = core.build_default_plan()
    for cell in plan.schedule:
        if cell.lane != "controlled" or cell.pair_index != 1:
            continue
        path = workspace / "raw" / f"{cell.cell_id}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        request = receipt["requests"][0]
        output_ids = [555] * 96
        request["output_token_ids"] = output_ids
        request["output_token_ids_sha256"] = core.token_ids_sha256(output_ids)
        latency = request["timing"]["latency_seconds"]["value"]
        request["timing"]["output_token_rate_tokens_per_second"]["value"] = (
            len(output_ids) / latency
        )
        receipt.pop("cell_sha256")
        receipt["cell_sha256"] = results._sha256_json(receipt)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        _sync_progress(path, receipt)
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    correctness = json.loads((bundle / "correctness.json").read_text(encoding="utf-8"))
    identity = correctness["controlled_output_identity"]
    assert identity["cross_mode_pair_outputs_identical"] is True
    assert identity["within_mode_lifecycles_identical"] is False
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    decision = by_id["output-identical-generation-crossover"]
    assert decision["state"] == "unsupported"
    assert decision["blockers"] == ["controlled_numeric_reproducibility"]


def test_natural_correctness_is_recomputed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _mutate_request(
        workspace,
        lane="natural",
        mode="compiled",
        field="decoded_output",
        value="not valid evaluator output",
    )
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    correctness = json.loads((bundle / "correctness.json").read_text(encoding="utf-8"))
    assert correctness["natural_all_correct"] is False
    assert any(not item["success"] for item in correctness["evaluations"])
    quality = correctness["quality_preservation"]
    assert len(quality["pair_effects"]) == 8
    assert quality["lower_confidence_endpoint"] < 0
    assert quality["noninferiority_supported"] is False
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["natural-output-quality-preserved"]["state"] == "unsupported"


def test_natural_claim_requires_correctness_and_quality_preservation(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    plan = core.build_default_plan()
    for cell in plan.schedule:
        if cell.lane != "natural":
            continue
        path = workspace / "raw" / f"{cell.cell_id}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["requests"][0]["decoded_output"] = "invalid in both modes"
        receipt.pop("cell_sha256")
        receipt["cell_sha256"] = results._sha256_json(receipt)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        _sync_progress(path, receipt)
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    correctness = json.loads((bundle / "correctness.json").read_text(encoding="utf-8"))
    quality = correctness["quality_preservation"]
    assert correctness["natural_all_correct"] is False
    assert [
        effect["compiled_minus_eager_request_success_rate"]
        for effect in quality["pair_effects"]
    ] == [0.0] * 8
    assert quality["confidence_method"] == "not_applicable_deterministic_pair_effects"
    assert quality["confidence_level"] is None
    assert quality["lower_confidence_endpoint"] is None
    assert quality["upper_confidence_endpoint"] is None
    assert quality["inference_state"] == "deterministic_complete_agreement"
    assert quality["noninferiority_supported"] is True
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["natural-output-quality-preserved"] == {
        "claim_id": "natural-output-quality-preserved",
        "state": "unsupported",
        "blockers": ["natural_absolute_correctness"],
    }
    assert by_id["natural-end-to-end-causal-speedup"] == {
        "claim_id": "natural-end-to-end-causal-speedup",
        "state": "unsupported",
        "blockers": ["natural_absolute_correctness"],
    }
    results.verify_bundle(bundle)
    runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"](bundle)


def test_deterministic_identical_quality_degradation_has_no_interval(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    for cell in core.build_default_plan().schedule:
        if cell.lane != "natural" or cell.mode != "compiled":
            continue
        path = workspace / "raw" / f"{cell.cell_id}.json"
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["requests"][0]["decoded_output"] = "invalid compiled output"
        receipt.pop("cell_sha256")
        receipt["cell_sha256"] = results._sha256_json(receipt)
        path.write_text(json.dumps(receipt), encoding="utf-8")
        _sync_progress(path, receipt)
    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)

    correctness = json.loads((bundle / "correctness.json").read_text(encoding="utf-8"))
    quality = correctness["quality_preservation"]
    assert (
        len(
            {
                effect["compiled_minus_eager_request_success_rate"]
                for effect in quality["pair_effects"]
            }
        )
        == 1
    )
    assert quality["confidence_method"] == "not_applicable_deterministic_pair_effects"
    assert quality["confidence_level"] is None
    assert quality["lower_confidence_endpoint"] is None
    assert quality["upper_confidence_endpoint"] is None
    assert quality["inference_state"] == "deterministic_noninferiority_failed"
    assert quality["noninferiority_supported"] is False


def test_censored_crossing_has_open_endpoints(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path, censored=True), bundle)
    analysis = json.loads((bundle / "analysis.json").read_text(encoding="utf-8"))
    controlled = analysis["controlled"]
    assert controlled["aggregate_first_crossing"]["state"] == "open"
    assert controlled["aggregate_sustained_crossing"]["state"] == "right_censored"
    assert controlled["simultaneous_band_first_crossing"]["state"] == "open"
    assert controlled["simultaneous_band_sustained_crossing"]["state"] == (
        "right_censored"
    )
    assert controlled["supported_crossing_gate_satisfied"] is False
    assert controlled["supported_sustained_crossing"] == {
        "state": "unsupported",
        "request_count": None,
        "lower_bound": None,
        "reason": "combined_band_crossing_and_sign_symmetry_gate_not_satisfied",
    }
    assert controlled["supported_crossing_basis"] == (
        "simultaneous_upper_band_sustained_crossing_observed_and_"
        "terminal_effect_sign_symmetry_p_value_at_most_0.05"
    )
    assert controlled["bootstrap_sustained_crossing_interval"]["state"] == "open"
    assert controlled["bootstrap_sustained_crossing_interval"]["lower"]["state"] == (
        "open"
    )
    assert controlled["bootstrap_sustained_crossing_interval"]["upper"]["state"] == (
        "open"
    )
    assert (
        controlled["bootstrap_sustained_crossing_interval"][
            "censor_sentinel_request_count"
        ]
        == 145
    )
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    decision = by_id["fixed-token-count-crossover"]
    assert decision["state"] == "unsupported"
    assert decision["blockers"] == ["controlled_supported_crossing"]


def test_controlled_crossing_also_requires_sign_symmetry_support(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    plan = core.build_default_plan()
    compiled = next(
        cell
        for cell in plan.schedule
        if cell.lane == "controlled"
        and cell.mode == "compiled"
        and cell.pair_index == 1
    )
    eager = next(
        cell
        for cell in plan.schedule
        if cell.lane == "controlled" and cell.mode == "eager" and cell.pair_index == 1
    )
    compiled_path = workspace / "raw" / f"{compiled.cell_id}.json"
    eager_path = workspace / "raw" / f"{eager.cell_id}.json"
    compiled_receipt = json.loads(compiled_path.read_text(encoding="utf-8"))
    eager_receipt = json.loads(eager_path.read_text(encoding="utf-8"))
    for compiled_request, eager_request in zip(
        compiled_receipt["requests"], eager_receipt["requests"], strict=True
    ):
        compiled_timing = compiled_request["timing"]
        cumulative_key = "cumulative_from_initialization_perf_counter_ns"
        eager_ns = eager_request["timing"][cumulative_key]
        compiled_ns = compiled_timing[cumulative_key]
        new_ns = 2 * eager_ns - compiled_ns + 900_000_000
        compiled_timing[cumulative_key] = new_ns
        compiled_timing["cumulative_from_initialization_seconds"]["value"] = (
            new_ns / 1_000_000_000
        )
    compiled_receipt.pop("cell_sha256")
    compiled_receipt["cell_sha256"] = results._sha256_json(compiled_receipt)
    compiled_path.write_text(json.dumps(compiled_receipt), encoding="utf-8")
    _sync_progress(compiled_path, compiled_receipt)

    bundle = tmp_path / "bundle"
    results.build_bundle(workspace, bundle)
    results.verify_bundle(bundle)
    runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"](bundle)

    analysis = json.loads((bundle / "analysis.json").read_text(encoding="utf-8"))
    controlled = analysis["controlled"]
    assert controlled["simultaneous_band_sustained_crossing_request_count"] is not None
    assert (
        controlled["terminal_effect_sign_flip_p_value"]
        > core.CONTROLLED_SIGN_SYMMETRY_ALPHA
    )
    assert controlled["supported_crossing_gate_satisfied"] is False
    assert controlled["supported_sustained_crossing"]["state"] == "unsupported"
    claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
    by_id = {claim["claim_id"]: claim for claim in claims["claims"]}
    assert by_id["fixed-token-count-crossover"] == {
        "claim_id": "fixed-token-count-crossover",
        "state": "unsupported",
        "blockers": ["controlled_supported_crossing"],
    }


def test_resealed_no_crossing_supported_analysis_and_claim_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    results.build_bundle(_workspace(tmp_path, censored=True), bundle)
    analysis_path = bundle / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    controlled = analysis["controlled"]
    controlled["simultaneous_band_sustained_crossing_request_count"] = 1
    controlled["simultaneous_band_sustained_crossing"] = {
        "state": "observed",
        "request_count": 1,
        "lower_bound": None,
    }
    controlled["supported_sustained_crossing"] = dict(
        controlled["simultaneous_band_sustained_crossing"]
    )
    analysis_path.write_text(results._json_text(analysis), encoding="utf-8")
    claims_path = bundle / "claim-matrix.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    decision = next(
        claim
        for claim in claims["claims"]
        if claim["claim_id"] == "fixed-token-count-crossover"
    )
    decision["state"] = "supported"
    decision["blockers"] = []
    claims_path.write_text(results._json_text(claims), encoding="utf-8")
    _reseal_bundle(bundle)

    with pytest.raises(results.CrossoverResultsError, match="analysis"):
        results.verify_bundle(bundle)
    standalone_verify = runpy.run_path(str(bundle / "evidence_bundle.py"))["verify"]
    with pytest.raises(ValueError, match="analysis"):
        standalone_verify(bundle)


def test_private_output_is_refused(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _mutate_request(
        workspace,
        lane="natural",
        mode="compiled",
        field="decoded_output",
        value="/Users/private/account",
    )
    with pytest.raises(results.CrossoverResultsError, match="private path"):
        results.build_bundle(workspace, tmp_path / "bundle")
