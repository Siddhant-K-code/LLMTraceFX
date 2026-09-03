"""Offline contract tests for the approved vLLM compilation experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as contract
from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    CELLS,
    HARD_CAP_USD,
    HardwareIdentity,
    LatencyRecord,
    LifecycleBudgetLedger,
    TerminalRequest,
    VLLMCompileContractError,
    VLLMCompilePlan,
    build_plan,
    calculate_break_even,
    validate_hardware_identity,
    validate_model_identity,
    workload_descriptors,
)

RATES = {
    "l40s_gpu_second_usd": "0.000542",
    "h100_gpu_second_usd": "0.001097",
    "cpu_core_second_usd": "0.0000131",
    "memory_gib_second_usd": "0.00000222",
    "volume_gib_month_usd": "0.09",
}
RUNTIME_PINS = {
    "python_version": "3.12.11",
    "vllm_version": "0.10.1.1",
    "torch_version": "2.8.0",
    "cuda_version": "12.8",
}
IMAGE_DIGEST = contract.OFFICIAL_VLLM_IMAGE_DIGEST
GIT_HEAD = "b" * 40
NOW = "2026-09-03T06:57:01Z"


def plan(**overrides: object) -> VLLMCompilePlan:
    values: dict[str, object] = {
        "prices": RATES,
        "effective_date": "2026-09-01",
        "price_source": "https://modal.com/pricing/2026-09-01",
        "price_source_sha256": "sha256:" + "a" * 64,
        "image_digest": IMAGE_DIGEST,
        "runtime_pins": RUNTIME_PINS,
        "as_of_date": "2026-09-03",
    }
    values.update(overrides)
    return build_plan(**values)  # type: ignore[arg-type]


def latency_records(
    eager: str, compiled: str
) -> tuple[list[LatencyRecord], list[LatencyRecord]]:
    return (
        [LatencyRecord(str(index), Decimal(eager)) for index in range(12)],
        [LatencyRecord(str(index), Decimal(compiled)) for index in range(12)],
    )


def terminal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "2k-structured-json-profile-extraction-rep-01",
        "finish_reason": "stop",
        "token_ids": [101, 202],
        "started_at": "2026-09-03T06:00:00Z",
        "ended_at": "2026-09-03T06:00:01.250000Z",
        "correctness": True,
    }
    payload.update(overrides)
    return payload


def test_exact_plan_math_and_canonical_round_trip() -> None:
    approved = plan()

    assert approved.first_pass_usd == Decimal("11.03358391012090444564819336")
    assert approved.full_retry_usd == approved.first_pass_usd
    assert approved.contingency_usd == Decimal("5.93283217975819110870361328")
    assert approved.envelope_usd == HARD_CAP_USD
    assert [line.kind for line in approved.lines] == [
        "image_allowance",
        "staging",
        "cell",
        "cell",
        "cell",
        "cell",
        "storage",
    ]
    assert [line.amount_usd for line in approved.lines] == [
        Decimal("0.33328800"),
        Decimal("0.33328800"),
        Decimal("1.79668800"),
        Decimal("1.79668800"),
        Decimal("3.29518800"),
        Decimal("3.29518800"),
        Decimal("0.1832559101209044456481933593"),
    ]
    assert VLLMCompilePlan.from_json(approved.to_json()) == approved
    assert approved.to_dict()["first_pass_usd"] == ("11.03358391012090444564819336")


def test_model_and_cell_contract_is_exact_and_ordered() -> None:
    payload = plan().to_dict()

    assert payload["model"] == {
        "id": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "expected_file_count": 15,
        "expected_bytes": 16_397_461_266,
    }
    assert [
        (cell.accelerator, cell.execution_mode, cell.compile_enabled) for cell in CELLS
    ] == [
        ("L40S", "eager", False),
        ("L40S", "compiled", True),
        ("H100!", "eager", False),
        ("H100!", "compiled", True),
    ]
    for cell in CELLS:
        assert (
            cell.gpu_count,
            cell.cpu_cores,
            cell.memory_gib,
            cell.max_containers,
            cell.min_containers,
            cell.concurrency,
            cell.retries,
            cell.allowance_seconds,
        ) == (1, 4, 32, 1, 0, 1, 0, 2700)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image_digest": "latest"}, "image digest"),
        ({"runtime_pins": {}}, "runtime pins"),
        (
            {
                "runtime_pins": {
                    **RUNTIME_PINS,
                    "vllm_version": "latest",
                }
            },
            "immutable",
        ),
        ({"price_source_sha256": "not-a-digest"}, "immutable"),
        ({"effective_date": "2026-07-01"}, "stale"),
        ({"hard_cap_usd": Decimal("28.01")}, "exactly USD 28"),
    ],
)
def test_missing_mutable_stale_or_over_cap_inputs_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(VLLMCompileContractError, match=message):
        plan(**overrides)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-0.1", "0", "1e999999"])
def test_nonfinite_nonpositive_or_wrong_prices_refuse(bad: str) -> None:
    rates = dict(RATES)
    rates["l40s_gpu_second_usd"] = bad

    with pytest.raises(VLLMCompileContractError):
        plan(prices=rates)


def test_changed_current_rates_are_recalculated_when_the_envelope_still_fits() -> None:
    rates = dict(RATES)
    rates["l40s_gpu_second_usd"] = "0.0006"

    changed = plan(prices=rates)

    assert changed.first_pass_usd > Decimal("11.03358391012090444564819336")
    assert changed.envelope_usd == HARD_CAP_USD


def test_changed_rates_refuse_when_first_pass_and_retry_exceed_cap() -> None:
    rates = {
        key: contract.canonical_decimal(Decimal(value) * 2)
        for key, value in RATES.items()
    }

    with pytest.raises(VLLMCompileContractError, match="full retry"):
        plan(prices=rates)


def test_tampered_or_overflowing_serialized_plan_refuses() -> None:
    payload = plan().to_dict()
    payload["envelope_usd"] = "28.00000000000000000000000001"

    with pytest.raises(VLLMCompileContractError, match="canonical contract"):
        VLLMCompilePlan.from_dict(payload)

    payload = plan().to_dict()
    payload["first_pass_lines"][0]["amount_usd"] = "1e999999"
    with pytest.raises(VLLMCompileContractError, match="canonical contract"):
        VLLMCompilePlan.from_dict(payload)


def test_hardware_identity_refuses_substitution_and_wrong_l40s() -> None:
    validate_hardware_identity(CELLS[0], HardwareIdentity("NVIDIA L40S", 1))
    validate_hardware_identity(CELLS[2], HardwareIdentity("NVIDIA H100 80GB HBM3", 1))
    validate_hardware_identity(CELLS[2], HardwareIdentity("NVIDIA H100 SXM5", 1))

    with pytest.raises(VLLMCompileContractError, match="NVIDIA L40S"):
        validate_hardware_identity(CELLS[0], HardwareIdentity("NVIDIA L40", 1))
    with pytest.raises(VLLMCompileContractError, match="NVIDIA H100"):
        validate_hardware_identity(CELLS[2], HardwareIdentity("NVIDIA A100", 1))
    with pytest.raises(VLLMCompileContractError, match="exactly 1"):
        validate_hardware_identity(
            CELLS[2], HardwareIdentity("NVIDIA H100 80GB HBM3", 2)
        )


def test_model_identity_requires_exact_revision_inventory() -> None:
    validate_model_identity(
        observed_revision=contract.MODEL_REVISION,
        observed_file_count=contract.EXPECTED_MODEL_FILE_COUNT,
        observed_bytes=contract.EXPECTED_MODEL_BYTES,
    )
    with pytest.raises(VLLMCompileContractError, match="model identity"):
        validate_model_identity(
            observed_revision="0" * 40,
            observed_file_count=contract.EXPECTED_MODEL_FILE_COUNT,
            observed_bytes=contract.EXPECTED_MODEL_BYTES,
        )


def test_packaged_workloads_have_exact_count_order_and_hashes() -> None:
    descriptors = workload_descriptors()

    assert len(descriptors) == 12
    assert all(not item.warmup for item in descriptors)
    assert [
        (item.context_tier, item.workload_id, item.repetition) for item in descriptors
    ] == [
        (tier, workload, repetition)
        for tier in ("2k", "8k", "16k")
        for workload in (
            "structured-json-profile-extraction",
            "prose-reasoning-two-train-gap",
        )
        for repetition in (1, 2)
    ]
    assert descriptors[0].prompt_sha256 == (
        "sha256:4a4a49e8368a99919fff1d1e68586466e279267d9de8d081ba8c08adac592fcd"
    )
    assert descriptors[-1].prompt_sha256 == (
        "sha256:4ddc616d1772e3ead314a5247729778b21bec55376465a276ab805bcde29ce41"
    )


def test_supplied_prompt_hashes_must_verify_payloads() -> None:
    payloads = {
        workload: {tier: f"{workload}-{tier}" for tier in contract.CONTEXT_TIERS}
        for workload in contract.WORKLOAD_IDS
    }
    hashes = {
        workload: {
            tier: "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            for tier, text in tiers.items()
        }
        for workload, tiers in payloads.items()
    }
    assert len(workload_descriptors(hashes, prompt_payloads=payloads)) == 12

    hashes["structured-json-profile-extraction"]["2k"] = "sha256:" + "0" * 64
    with pytest.raises(VLLMCompileContractError, match="hash mismatch"):
        workload_descriptors(hashes, prompt_payloads=payloads)
    with pytest.raises(VLLMCompileContractError, match="require payloads"):
        workload_descriptors(hashes)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("finish_reason", None, "finish_reason"),
        ("token_ids", None, "token_ids"),
        ("token_ids", [], "token_ids"),
        ("started_at", None, "started_at"),
        ("ended_at", None, "ended_at"),
        ("correctness", None, "correctness"),
    ],
)
def test_incomplete_terminal_evidence_refuses(
    field: str, value: object, message: str
) -> None:
    payload = terminal_payload()
    payload[field] = value

    with pytest.raises(VLLMCompileContractError, match=message):
        TerminalRequest.from_dict(payload)


def test_correctness_false_is_terminal_and_optionals_stay_none() -> None:
    record = TerminalRequest.from_dict(
        terminal_payload(correctness=False, ttft_seconds=None)
    )

    assert record.terminal is True
    assert record.correctness is False
    assert record.ttft_seconds is None
    assert record.output_tokens_per_second is None
    assert record.gpu_memory_gib is None
    assert record.latency_seconds == Decimal("1.25")
    assert record.to_dict()["ttft_seconds"] is None


def test_nonterminal_finish_reason_refuses() -> None:
    with pytest.raises(VLLMCompileContractError, match="complete stop"):
        TerminalRequest.from_dict(terminal_payload(finish_reason="abort"))


@pytest.mark.parametrize("metric", ["NaN", "Infinity", "-1", "1e999999"])
def test_terminal_optional_metrics_reject_nonfinite_or_invalid(metric: str) -> None:
    with pytest.raises(VLLMCompileContractError):
        TerminalRequest.from_dict(terminal_payload(ttft_seconds=metric))


def test_break_even_observed_prefix_crossing() -> None:
    eager, compiled = latency_records("2", "1")

    result = calculate_break_even(
        eager,
        compiled,
        eager_cell=CELLS[0],
        compiled_cell=CELLS[1],
        compilation_overhead_seconds=Decimal("3"),
    )

    assert result.observed_requests == 3
    assert result.observed_lower_bound_requests is None
    assert result.extrapolated_requests is None
    assert result.full_cycle_saving_seconds == Decimal("12")


def test_break_even_extrapolates_repeated_exact_cycle() -> None:
    eager, compiled = latency_records("2", "1")

    result = calculate_break_even(
        eager,
        compiled,
        eager_cell=CELLS[0],
        compiled_cell=CELLS[1],
        compilation_overhead_seconds=Decimal("25"),
    )

    assert result.observed_requests is None
    assert result.observed_lower_bound_requests == 12
    assert result.extrapolated_requests == 25


def test_break_even_no_cycle_saving_is_null() -> None:
    eager, compiled = latency_records("1", "1")

    result = calculate_break_even(
        eager,
        compiled,
        eager_cell=CELLS[0],
        compiled_cell=CELLS[1],
        compilation_overhead_seconds=Decimal("1"),
    )

    assert result.observed_requests is None
    assert result.observed_lower_bound_requests == 12
    assert result.extrapolated_requests is None
    assert result.full_cycle_saving_seconds == 0


def test_break_even_requires_exact_pairing_and_12_records() -> None:
    eager, compiled = latency_records("2", "1")
    compiled[0] = LatencyRecord("different", Decimal("1"))
    with pytest.raises(VLLMCompileContractError, match="identities"):
        calculate_break_even(
            eager, compiled, eager_cell=CELLS[0], compiled_cell=CELLS[1]
        )
    with pytest.raises(VLLMCompileContractError, match="exactly 12"):
        calculate_break_even(
            eager[:-1],
            compiled[:-1],
            eager_cell=CELLS[0],
            compiled_cell=CELLS[1],
        )


def test_break_even_refuses_cross_hardware_pairing() -> None:
    eager, compiled = latency_records("2", "1")
    with pytest.raises(VLLMCompileContractError, match="same accelerator"):
        calculate_break_even(
            eager, compiled, eager_cell=CELLS[0], compiled_cell=CELLS[3]
        )


def initialize_ledger(
    tmp_path: Path, *, workspace: Path | None = None
) -> LifecycleBudgetLedger:
    return LifecycleBudgetLedger.initialize(
        tmp_path / "ledger.json",
        plan=plan(),
        git_head=GIT_HEAD,
        workspace_path=workspace or tmp_path,
    )


def test_ledger_is_path_bound_and_reservation_is_pre_command(tmp_path: Path) -> None:
    ledger = initialize_ledger(tmp_path)
    event = ledger.reserve(
        "image-build",
        line_id="image-allowance",
        ceiling_usd=plan().lines[0].amount_usd,
        argv=("modal", "run", "image"),
        reserved_at=NOW,
    )
    snapshot = ledger.snapshot()

    assert event["stage"] == "pre_command"
    assert snapshot["plan_sha256"] == plan().content_sha256
    assert snapshot["pricing_sha256"] == plan().prices.content_sha256
    assert snapshot["git_head"] == GIT_HEAD
    assert snapshot["reserved_usd"] == "0.333288"
    assert snapshot["remaining_usd"] == "27.666712"
    assert snapshot["ledger_sha256"].startswith("sha256:")

    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    moved_context = LifecycleBudgetLedger(
        tmp_path / "ledger.json",
        plan=plan(),
        git_head=GIT_HEAD,
        workspace_path=other_workspace,
    )
    with pytest.raises(VLLMCompileContractError, match="workspace_path_sha256"):
        moved_context.snapshot()


def test_ledger_chain_tamper_is_detected_even_if_outer_seal_is_recomputed(
    tmp_path: Path,
) -> None:
    ledger = initialize_ledger(tmp_path)
    for command in ("first", "second"):
        ledger.reserve(
            command,
            line_id="image-allowance",
            ceiling_usd=plan().lines[0].amount_usd,
            argv=("runner", command),
            reserved_at=NOW,
        )
    raw = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    raw["events"][0]["command_id"] = "tampered"
    raw = contract._seal(raw)
    (tmp_path / "ledger.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(VLLMCompileContractError, match="event hash"):
        ledger.snapshot()


def test_duplicate_reservation_refuses_without_mutation(tmp_path: Path) -> None:
    ledger = initialize_ledger(tmp_path)
    ledger.reserve(
        "cell-l40s-eager",
        line_id="cell-l40s-eager",
        ceiling_usd=plan().lines[2].amount_usd,
        argv=("runner", "l40s-eager"),
        reserved_at=NOW,
    )
    before = copy.deepcopy(ledger.snapshot())

    with pytest.raises(VLLMCompileContractError, match="already reserved"):
        ledger.reserve(
            "cell-l40s-eager",
            line_id="cell-l40s-eager",
            ceiling_usd=plan().lines[2].amount_usd,
            argv=("runner", "l40s-eager"),
            reserved_at=NOW,
        )
    assert ledger.snapshot() == before


def test_third_lifecycle_attempt_refuses_and_preserves_ledger(
    tmp_path: Path,
) -> None:
    ledger = initialize_ledger(tmp_path)
    line = plan().lines[5]
    for command in ("authorized", "retry"):
        ledger.reserve(
            command,
            line_id=line.line_id,
            ceiling_usd=line.amount_usd,
            argv=("runner", command),
            reserved_at=NOW,
        )
    before = ledger.snapshot()

    with pytest.raises(VLLMCompileContractError, match="cannot be reserved more"):
        ledger.reserve(
            "third-attempt",
            line_id=line.line_id,
            ceiling_usd=line.amount_usd,
            argv=("runner", "third-attempt"),
            reserved_at=NOW,
        )
    assert ledger.snapshot() == before


def test_ledger_refuses_understated_or_unknown_plan_line(tmp_path: Path) -> None:
    ledger = initialize_ledger(tmp_path)
    with pytest.raises(VLLMCompileContractError, match="exactly match"):
        ledger.reserve(
            "understated",
            line_id="cell-h100-compiled",
            ceiling_usd=Decimal("0.01"),
            argv=("runner", "understated"),
            reserved_at=NOW,
        )
    with pytest.raises(VLLMCompileContractError, match="not in the plan"):
        ledger.reserve(
            "unknown",
            line_id="unknown",
            ceiling_usd=Decimal("0.01"),
            argv=("runner", "unknown"),
            reserved_at=NOW,
        )


def test_ledger_cannot_be_reset_or_opened_at_another_path(tmp_path: Path) -> None:
    ledger = initialize_ledger(tmp_path)
    with pytest.raises(VLLMCompileContractError, match="already exists"):
        initialize_ledger(tmp_path)

    moved = tmp_path / "moved.json"
    moved.write_bytes((tmp_path / "ledger.json").read_bytes())
    other = LifecycleBudgetLedger(
        moved, plan=plan(), git_head=GIT_HEAD, workspace_path=tmp_path
    )
    with pytest.raises(VLLMCompileContractError, match="ledger_path_sha256"):
        other.snapshot()
    assert ledger.snapshot()["revision"] == 0
