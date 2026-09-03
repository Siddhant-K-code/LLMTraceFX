"""Synthetic offline tests for the vLLM compilation evidence bundle."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    APPROVED_PLAN_SHA256,
    CELLS,
    CURRENT_RATES,
    EXPECTED_GPU_NAMES,
    EXPECTED_MODEL_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_VLLM_IMAGE_DIGEST,
    build_plan,
    canonical_decimal,
    workload_descriptors,
)

HEAD = "a" * 40
PAGE_HASH = "sha256:" + "b" * 64
VOLUME_HASH = "sha256:" + "c" * 64


@pytest.fixture
def artifact_root(request: pytest.FixtureRequest) -> Iterator[Path]:
    root = Path(".cache/llmtracefx-tests/vllm-evidence") / request.node.name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = evidence._sha256_json(result)
    return result


def iso(second: int | float) -> str:
    return (
        datetime(2026, 9, 3, tzinfo=timezone.utc) + timedelta(seconds=second)
    ).isoformat()


def synthetic() -> dict[str, Any]:
    plan = build_plan(
        prices={name: str(value) for name, value in CURRENT_RATES.items()},
        effective_date="2026-09-03",
        price_source="modal-cli://billing/rates/1.5.4",
        price_source_sha256="sha256:" + "d" * 64,
        image_digest=OFFICIAL_VLLM_IMAGE_DIGEST,
        runtime_pins=evidence.RUNTIME_PINS,
        as_of_date=date(2026, 9, 3).isoformat(),
    )
    workload_hash, output_hash = evidence._harness_hashes()
    unique_descriptors = [
        item for item in workload_descriptors() if item.repetition == 1
    ]
    prompts: list[dict[str, Any]] = []
    arrays: dict[str, list[int]] = {}
    for index, descriptor in enumerate(unique_descriptors):
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        token_ids = [100 + index, 200 + index, 300 + index]
        arrays[key] = token_ids
        prompts.append(
            {
                "key": key,
                "prompt_sha256": descriptor.prompt_sha256,
                "decoded_prompt_sha256": "sha256:" + f"{index + 1:064x}",
                "prompt_token_ids": token_ids,
                "prompt_token_ids_sha256": evidence._sha256_json(token_ids),
                "input_token_count": len(token_ids),
            }
        )
    receipt = seal(
        {
            "schema_version": "1",
            "plan_sha256": plan.content_sha256,
            "workload_sha256": workload_hash,
            "output_contract_sha256": output_hash,
            "runtime_sha256": evidence._sha256_json(evidence.RUNTIME_PINS),
            "image_sha256": evidence._sha256_json(
                {"reference": evidence.IMAGE_REFERENCE}
            ),
            "image_digest": OFFICIAL_VLLM_IMAGE_DIGEST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_file_count": 15,
            "model_bytes": EXPECTED_MODEL_BYTES,
            "inventory": evidence._conversion_inventory(),
            "prompts": prompts,
            "prompt_ids_sha256": evidence._sha256_json(
                {
                    "schema_version": "1",
                    "workload_sha256": workload_hash,
                    "prompts": arrays,
                }
            ),
            "staged_at": iso(0),
        },
        "receipt_sha256",
    )
    prompt_map = {item["key"]: item for item in prompts}
    cell_records: list[dict[str, Any]] = []
    lifecycle_records: list[dict[str, Any]] = []
    latencies = (Decimal("2"), Decimal("1"), Decimal("2"), Decimal("1"))
    init_durations = (1, 3, 1, 3)
    for cell_index, cell in enumerate(CELLS):
        requests: list[dict[str, Any]] = []
        for descriptor in workload_descriptors():
            key = f"{descriptor.context_tier}/{descriptor.workload_id}"
            prompt = prompt_map[key]
            latency = latencies[cell_index]
            output_ids = [1, 2]
            request_start = 30 + descriptor.ordinal * 3
            requests.append(
                {
                    **descriptor.to_dict(),
                    "terminal": True,
                    "started_at": iso(request_start),
                    "ended_at": iso(request_start + int(latency)),
                    "wall_clock_seconds": float(latency),
                    "input_token_count": prompt["input_token_count"],
                    "input_token_ids_sha256": prompt["prompt_token_ids_sha256"],
                    "output_token_count": len(output_ids),
                    "output_tokens_per_second": float(
                        Decimal(len(output_ids)) / latency
                    ),
                    "output_rate_basis": ("output_tokens_per_complete_response_second"),
                    "output_token_ids": output_ids,
                    "decoded_output": "not a valid expected answer",
                    "finish_reason": "length",
                    "ttft_seconds": None,
                    "evaluator_input": {
                        "workload_id": descriptor.workload_id,
                        "context_tier": descriptor.context_tier,
                        "decoded_output": "not a valid expected answer",
                        "output_token_ids": output_ids,
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
            )
        cell_records.append(
            seal(
                {
                    "schema_version": "1",
                    "cell": cell.to_dict(),
                    "plan_sha256": plan.content_sha256,
                    "staging_receipt_sha256": receipt["receipt_sha256"],
                    "workload_sha256": workload_hash,
                    "output_contract_sha256": output_hash,
                    "runtime_sha256": evidence._sha256_json(evidence.RUNTIME_PINS),
                    "image_sha256": evidence._sha256_json(
                        {"reference": evidence.IMAGE_REFERENCE}
                    ),
                    "hardware": {
                        "gpu_name": EXPECTED_GPU_NAMES[cell.accelerator],
                        "gpu_count": 1,
                        "driver_version": "580.1",
                        "memory_total_mib": 48_000.0,
                        "memory_used_mib": 100.0,
                    },
                    "runtime": evidence.RUNTIME_PINS,
                    "initialization_started_at": iso(20),
                    "initialization_ready_at": iso(20 + init_durations[cell_index]),
                    "compilation_seconds": (None if not cell.compile_enabled else 2.0),
                    "cuda_graph_seconds": None,
                    "peak_gpu_memory_mib": 12_000.0,
                    "requests": requests,
                    "correctness_evaluated_remotely": False,
                    "terminal": True,
                },
                "cell_sha256",
            )
        )
        function = (
            "l40s_eager",
            "l40s_compiled",
            "h100_eager",
            "h100_compiled",
        )[cell_index]
        lifecycle_records.append(
            seal(
                {
                    "function": function,
                    "started_at": iso(100 + cell_index * 40),
                    "events": [
                        {
                            "received_at": iso(101 + cell_index * 40),
                            "event": {
                                "event": "container_started",
                                "provenance": "modal_provider",
                            },
                        },
                        {
                            "received_at": iso(129 + cell_index * 40),
                            "event": {
                                "event": "cell_terminal",
                                "provenance": "derived",
                            },
                        },
                    ],
                    "ended_at": iso(130 + cell_index * 40),
                },
                "artifact_sha256",
            )
        )
    unavailable = {
        "facts": None,
        "unavailable_reason": "unsupported",
        "unsupported_fields": [
            "credits_usd",
            "budget_usd",
            "spend_limit_usd",
        ],
    }
    inventories = {
        name: {"count": 0, "status_counts": {}}
        for name in ("apps", "volumes", "containers", "secrets")
    }
    teardown = seal(
        {
            "schema_version": "1",
            "experiment_id": "run-01",
            "complete": True,
            "steps": [
                {"operation": "stop_app", "status": "already_stopped"},
                {"operation": "delete_volume", "status": "complete"},
            ],
            "billing_after": None,
            "billing_after_unavailable_reason": "unsupported",
            "billing_unsupported_fields": unavailable["unsupported_fields"],
            "secrets_created": 0,
            "credentials_to_revoke": [],
            "post_delete_storage_billing_days_accounted": 4,
            "provider_inventory_after": inventories,
            "inventory_status": "complete",
        },
        "artifact_sha256",
    )
    return {
        "execution_contract": {
            "schema_version": "1",
            "experiment_id": "run-01",
            "git_head": HEAD,
            "approved_plan_sha256": APPROVED_PLAN_SHA256,
            "safe_command_argv": [
                "uv",
                "run",
                "python",
                "-m",
                "llmtracefx.optimizer.lab.qwen3_8b.modal_orchestrator",
                "--approval",
                "<approved-plan-path>",
                "--approval-sha256",
                APPROVED_PLAN_SHA256,
                "--git-head",
                HEAD,
                "--workspace",
                "<repository-root>",
                "--output-dir",
                "<private-output-directory>",
                "--ledger",
                "<private-ledger-path>",
                "--experiment-id",
                "run-01",
            ],
            "plan": plan.to_dict(),
        },
        "pricing_snapshot": {
            "schema_version": "1",
            "retrieved_date": "2026-09-03",
            "pricing_page": {"status": 200, "sha256": PAGE_HASH},
            "volumes_page": {"status": 200, "sha256": VOLUME_HASH},
            "rates_response_sha256": plan.prices.source_sha256,
        },
        "staging_receipt": receipt,
        "cell_records": cell_records,
        "lifecycle_records": lifecycle_records,
        "ledger_snapshot": {
            "schema_version": "1",
            "plan_sha256": plan.content_sha256,
            "revision": 7,
            "reserved_usd": canonical_decimal(plan.first_pass_usd),
            "remaining_usd": canonical_decimal(Decimal(28) - plan.first_pass_usd),
            "lines": [
                {
                    "line_id": line.line_id,
                    "reserved_usd": canonical_decimal(line.amount_usd),
                }
                for line in plan.lines
            ],
        },
        "billing_before": unavailable,
        "billing_after": unavailable,
        "teardown_report": teardown,
    }


def build(path: Path, inputs: dict[str, Any]) -> None:
    path.mkdir()
    evidence.build_bundle(path, **inputs)


def write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.write_text(evidence.canonical_json(value), encoding="utf-8")


def refresh_checksums(path: Path) -> None:
    lines = []
    for name in sorted(set(evidence.BUNDLE_FILES) - {"SHA256SUMS"}):
        digest = hashlib.sha256((path / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (path / "SHA256SUMS").write_text("".join(lines))


def test_render_is_deterministic_and_observed_break_even(
    artifact_root: Path,
) -> None:
    inputs = synthetic()
    first, second = artifact_root / "first", artifact_root / "second"
    build(first, copy.deepcopy(inputs))
    build(second, copy.deepcopy(inputs))

    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    result = evidence.verify_bundle(first)
    assert result["requests_verified"] == 48
    crossing = json.loads((first / "break-even.json").read_text())
    assert [item["observed_requests"] for item in crossing["pairs"]] == [2, 2]
    correctness = json.loads((first / "correctness-report.json").read_text())
    assert correctness["failed"] > 0
    assert correctness["model_output_executed"] is False
    claims = json.loads((first / "claim-matrix.json").read_text())
    assert all("mlx" not in item.lower() for item in claims["ranking_scope"])
    assert all(
        item["relation"] == "uses_workload_contract" for item in claims["claims"]
    )


def test_builds_from_complete_orchestrator_directory(artifact_root: Path) -> None:
    inputs = synthetic()
    raw = artifact_root / "raw"
    bundle = artifact_root / "bundle"
    raw.mkdir()
    bundle.mkdir()
    for filename, key in (
        ("evidence-contract-input.json", "execution_contract"),
        ("pricing-snapshot-input.json", "pricing_snapshot"),
        ("billing-before-input.json", "billing_before"),
        ("ledger-projection-input.json", "ledger_snapshot"),
    ):
        write_canonical(raw / filename, seal(inputs[key], "artifact_sha256"))
    write_canonical(raw / "staging-receipt.json", inputs["staging_receipt"])
    for name, cell, lifecycle in zip(
        evidence._FUNCTIONS,
        inputs["cell_records"],
        inputs["lifecycle_records"],
        strict=True,
    ):
        write_canonical(raw / f"{name}-terminal.json", cell)
        write_canonical(raw / f"{name}-lifecycle.json", lifecycle)
    write_canonical(raw / "teardown-report.json", inputs["teardown_report"])

    evidence.build_from_execution_directory(raw, bundle)

    assert evidence.verify_bundle(bundle)["requests_verified"] == 48


@pytest.mark.parametrize(
    ("compiled_latency", "expected"),
    [
        (Decimal("1.95"), 40),
        (Decimal("2.1"), None),
    ],
)
def test_no_observed_crossing_extrapolates_only_positive_cycle(
    artifact_root: Path, compiled_latency: Decimal, expected: int | None
) -> None:
    inputs = synthetic()
    for cell_index in (1, 3):
        record = inputs["cell_records"][cell_index]
        record.pop("cell_sha256")
        for request in record["requests"]:
            request["wall_clock_seconds"] = float(compiled_latency)
            request["output_tokens_per_second"] = float(Decimal(2) / compiled_latency)
            started = datetime.fromisoformat(request["started_at"])
            request["ended_at"] = (
                started + timedelta(seconds=float(compiled_latency))
            ).isoformat()
        inputs["cell_records"][cell_index] = seal(record, "cell_sha256")
    bundle = artifact_root / "bundle"
    build(bundle, inputs)
    pairs = json.loads((bundle / "break-even.json").read_text())["pairs"]
    assert all(item["observed_requests"] is None for item in pairs)
    assert all(item["observed_lower_bound_requests"] == 12 for item in pairs)
    assert all(item["extrapolated_requests"] == expected for item in pairs)


def test_output_count_divergence_suppresses_break_even(artifact_root: Path) -> None:
    inputs = synthetic()
    compiled = inputs["cell_records"][1]
    compiled.pop("cell_sha256")
    request = compiled["requests"][0]
    request["output_token_ids"] = [1]
    request["output_token_count"] = 1
    request["output_tokens_per_second"] = 1.0
    request["evaluator_input"]["output_token_ids"] = [1]
    inputs["cell_records"][1] = seal(compiled, "cell_sha256")

    bundle = artifact_root / "bundle"
    build(bundle, inputs)
    pair = json.loads((bundle / "break-even.json").read_text())["pairs"][0]
    assert pair["comparable"] is False
    assert pair["paired_output_count_parity"] is False
    assert pair["identical_output_token_ids"] is False
    assert pair["observed_requests"] is None
    assert pair["extrapolated_requests"] is None


def test_equal_length_output_divergence_is_reported(artifact_root: Path) -> None:
    inputs = synthetic()
    compiled = inputs["cell_records"][1]
    compiled.pop("cell_sha256")
    request = compiled["requests"][0]
    request["output_token_ids"] = [1, 3]
    request["evaluator_input"]["output_token_ids"] = [1, 3]
    inputs["cell_records"][1] = seal(compiled, "cell_sha256")

    bundle = artifact_root / "bundle"
    build(bundle, inputs)
    pair = json.loads((bundle / "break-even.json").read_text())["pairs"][0]
    assert pair["comparable"] is True
    assert pair["paired_output_count_parity"] is True
    assert pair["identical_output_token_ids"] is False
    assert pair["first_divergent_request_ordinal"] == 1


def test_request_clock_mismatch_is_rejected(artifact_root: Path) -> None:
    inputs = synthetic()
    cell = inputs["cell_records"][0]
    cell.pop("cell_sha256")
    cell["requests"][0]["ended_at"] = iso(100)
    inputs["cell_records"][0] = seal(cell, "cell_sha256")

    with pytest.raises(evidence.VLLMCompileEvidenceError, match="does not reconcile"):
        build(artifact_root / "bundle", inputs)


def test_negative_initialization_delta_is_labeled_as_no_penalty() -> None:
    requests = [
        {
            "latency_seconds": "2",
            "output_token_count": 2,
            "output_token_ids": [1, 2],
            "correctness": True,
        }
        for _ in range(12)
    ]
    result = evidence._break_even_pair(
        {"initialization_seconds": "10"},
        {"initialization_seconds": "5"},
        requests,
        requests,
        eager_rate=Decimal("1"),
        compiled_rate=Decimal("1"),
    )
    assert result["observed_requests"] == 1
    assert result["initialization_delta_seconds"] == "-5"
    assert result["initialization_penalty_seconds"] == "0"
    assert result["no_measured_cold_start_penalty"] is True


def test_large_token_arrays_and_hyphenated_prose_pass_privacy_scan() -> None:
    token_ids = list(range(20_000))
    assert len(evidence._json_text(token_ids).encode()) > evidence.MAX_STRING_BYTES
    evidence._walk_safe(token_ids)
    evidence._walk_safe("A risk-adjusted task-oriented approach can remain disk-bound.")


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        (
            "workload-contract.json",
            lambda value: value["prompts"][0]["prompt_token_ids"].append(999),
        ),
        (
            "request-records.jsonl",
            lambda value: value[0].__setitem__("latency_seconds", "99"),
        ),
        (
            "cost-ledger.json",
            lambda value: value.__setitem__("inferred_cell_total_usd", "9"),
        ),
        (
            "break-even.json",
            lambda value: value["pairs"][0].__setitem__("observed_requests", 9),
        ),
    ],
)
def test_semantic_tampering_rejects_even_with_refreshed_checksums(
    artifact_root: Path, filename: str, mutation: Any
) -> None:
    bundle = artifact_root / "bundle"
    build(bundle, synthetic())
    path = bundle / filename
    if filename.endswith(".jsonl"):
        value = [json.loads(line) for line in path.read_text().splitlines()]
        mutation(value)
        path.write_text(evidence._jsonl_text(value))
    else:
        value = json.loads(path.read_text())
        mutation(value)
        path.write_text(evidence._json_text(value))
    refresh_checksums(bundle)
    with pytest.raises(evidence.VLLMCompileEvidenceError):
        evidence.verify_bundle(bundle)


def test_checksum_missing_extra_and_symlink_reject(artifact_root: Path) -> None:
    bundle = artifact_root / "bundle"
    build(bundle, synthetic())
    (bundle / "README.md").write_text("tampered")
    with pytest.raises(evidence.VLLMCompileEvidenceError, match="checksum"):
        evidence.verify_bundle(bundle)

    build(artifact_root / "missing", synthetic())
    (artifact_root / "missing" / "report.html").unlink()
    with pytest.raises(evidence.VLLMCompileEvidenceError, match="allowlist"):
        evidence.verify_bundle(artifact_root / "missing")

    build(artifact_root / "extra", synthetic())
    (artifact_root / "extra" / "extra.txt").write_text("extra")
    with pytest.raises(evidence.VLLMCompileEvidenceError, match="allowlist"):
        evidence.verify_bundle(artifact_root / "extra")

    build(artifact_root / "symlink", synthetic())
    target = artifact_root / "target"
    target.write_text("target")
    report = artifact_root / "symlink" / "report.html"
    report.unlink()
    os.symlink(target.resolve(), report)
    with pytest.raises(evidence.VLLMCompileEvidenceError, match="non-symlink"):
        evidence.verify_bundle(artifact_root / "symlink")


@pytest.mark.parametrize("kind", ["private", "credential", "nonfinite", "zero"])
def test_unsafe_or_ambiguous_inputs_reject(artifact_root: Path, kind: str) -> None:
    inputs = synthetic()
    record = inputs["cell_records"][0]
    record.pop("cell_sha256")
    if kind == "private":
        record["requests"][0]["decoded_output"] = "/Users/private/model"
    elif kind == "credential":
        record["requests"][0]["decoded_output"] = "hf_secretvalue"
    elif kind == "nonfinite":
        record["requests"][0]["wall_clock_seconds"] = float("nan")
    else:
        record["peak_gpu_memory_mib"] = 0
    inputs["cell_records"][0] = (
        record if kind == "nonfinite" else seal(record, "cell_sha256")
    )
    path = artifact_root / "bundle"
    path.mkdir()
    with pytest.raises(evidence.VLLMCompileEvidenceError):
        evidence.build_bundle(path, **inputs)


def test_incomplete_teardown_rejects(artifact_root: Path) -> None:
    inputs = synthetic()
    teardown = inputs["teardown_report"]
    teardown.pop("artifact_sha256")
    teardown["complete"] = False
    inputs["teardown_report"] = seal(teardown, "artifact_sha256")
    path = artifact_root / "bundle"
    path.mkdir()
    with pytest.raises(evidence.VLLMCompileEvidenceError, match="teardown"):
        evidence.build_bundle(path, **inputs)
