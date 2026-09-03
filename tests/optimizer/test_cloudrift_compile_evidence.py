"""Load-bearing tests for the CloudRift vLLM compilation evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_compile_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_runner as runner

ROOT = Path(__file__).parents[2]
PUBLIC = ROOT / "examples" / "optimizer" / "qwen3-8b-vllm-compile-break-even"


def load(name: str) -> dict:
    return json.loads((PUBLIC / name).read_text(encoding="utf-8"))


def test_committed_bundle_verifies() -> None:
    evidence.verify_bundle(PUBLIC)


def test_verifier_rejects_resealed_execution_config_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    shutil.copytree(PUBLIC, bundle)
    contract = json.loads(
        (bundle / "experiment-contract.json").read_text(encoding="utf-8")
    )
    contract["cells"][1]["compilation_mode"] = "NONE"
    evidence._write_json(bundle / "experiment-contract.json", contract)
    checksums = [
        f"{evidence._sha256_bytes((bundle / name).read_bytes())}  {name}"
        for name in evidence.HASHED_FILES
    ]
    (bundle / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(evidence.CloudRiftEvidenceError, match="contract binding"):
        evidence.verify_bundle(bundle)


def test_exact_two_cell_contract_and_break_even() -> None:
    contract = load("experiment-contract.json")
    result = load("break-even.json")

    assert [cell["mode"] for cell in contract["cells"]] == ["eager", "compiled"]
    assert contract["request_count_per_cell"] == 12
    assert contract["isolation"]["max_live_cells"] == 1
    assert contract["isolation"]["hard_timeout_seconds_per_cell"] == 2700
    assert contract["isolation"]["model_warmup_requests"] == 0
    assert result["observed_break_even_request_count"] is None
    assert result["observed_lower_bound_request_count"] == 12
    assert result["modeled_repeated_cycle_break_even_request_count"] == 113


def test_cost_scopes_and_teardown_are_not_overstated() -> None:
    cost = load("cost-ledger.json")
    teardown = load("teardown-report.json")

    assert cost["inferred_spend_usd_through_scheduled_shutdown_boundary"] == "0.393033"
    assert cost["provider_reported_spend_usd"] is None
    assert cost["final_inferred_spend_through_console_termination_usd"] == "0.484358"
    assert teardown["experiment_containers_remaining"] == 0
    assert teardown["gpu_processes_remaining"] == 0
    assert teardown["temporary_public_key_removed"] is True
    assert teardown["os_shutdown_observed"] is None
    assert teardown["os_shutdown_observation_unavailable_reason"]
    assert teardown["provider_console_termination_confirmed"] is True
    assert teardown["provider_console_terminated_at"] == "2026-09-03T22:19:00+05:30"


def test_all_requests_are_terminal_and_have_real_ttft() -> None:
    requests = [
        json.loads(line)
        for line in (PUBLIC / "request-records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(requests) == 24
    assert all(item["terminal"] for item in requests)
    assert sum(item["correctness"] for item in requests) == 22
    assert all(0 < item["ttft_seconds"] <= item["latency_seconds"] for item in requests)
    assert all(item["output_token_ids"] for item in requests)
    assert all(item["finish_reason"] in {"stop", "length"} for item in requests)


def test_missing_compile_components_remain_null() -> None:
    lifecycle = [
        json.loads(line)
        for line in (PUBLIC / "lifecycle-records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    compiled = lifecycle[1]

    assert compiled["compilation_seconds"] is None
    assert compiled["cuda_graph_seconds"] is None
    assert compiled["compilation_seconds_unobservable_reason"]
    assert compiled["cuda_graph_seconds_unobservable_reason"]


def test_correctness_and_output_identity_are_mode_specific() -> None:
    correctness = load("correctness-report.json")

    assert correctness["successful_requests"] == 22
    assert correctness["all_requests_correct"] is False
    assert correctness["successful_requests_by_mode"] == {
        "compiled": 12,
        "eager": 10,
    }
    assert correctness["paired_output_token_identity_matches"] == 8
    assert correctness["paired_output_token_identity_mismatched_ordinals"] == [
        7,
        8,
        11,
        12,
    ]


def test_documented_verifier_command_works() -> None:
    completed = subprocess.run(
        [sys.executable, str(PUBLIC / "evidence_bundle.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CloudRift vLLM compilation evidence verified" in completed.stdout


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/Users/private/run.json", "private home path"),
        ("/home/private/run.json", "private home path"),
        ("203.0.113.7", "IP address"),
        ("private@example.com", "email address"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private key"),
        ("GPU-00000000-0000-0000-0000-000000000000", "GPU UUID"),
    ],
)
def test_privacy_scan_rejects_private_values(
    tmp_path: Path, value: str, message: str
) -> None:
    (tmp_path / "artifact.txt").write_text(value, encoding="utf-8")
    with pytest.raises(evidence.CloudRiftEvidenceError, match=message):
        evidence._scan_privacy(tmp_path)


def test_runner_parser_allows_canary_without_state() -> None:
    args = runner.build_parser().parse_args(
        ["tokenizer-canary", "--model-path", "/model", "--output", "/output.json"]
    )
    assert args.state_path is None


def test_vllm_metric_ttft_rejects_invalid_values() -> None:
    assert (
        runner._metric_ttft(SimpleNamespace(first_token_ts=3.25, arrival_time=3.0))
        == 0.25
    )
    assert (
        runner._metric_ttft(SimpleNamespace(first_token_ts=2.0, arrival_time=3.0))
        is None
    )
    assert runner._metric_ttft(SimpleNamespace(first_token_latency=None)) is None


def test_resolved_configuration_is_mode_specific() -> None:
    eager = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=True),
                compilation_config=SimpleNamespace(
                    mode=SimpleNamespace(name="NONE"),
                    cudagraph_mode=SimpleNamespace(name="NONE"),
                ),
            )
        )
    )
    compiled = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=False),
                compilation_config=SimpleNamespace(
                    mode=SimpleNamespace(name="VLLM_COMPILE"),
                    cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"),
                ),
            )
        )
    )

    assert runner._resolved(eager, False)["enforce_eager"] is True
    assert runner._resolved(compiled, True)["enforce_eager"] is False


def test_runner_binds_prompts_and_live_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory = [{"path": "config.json", "size_bytes": 1, "sha256": "a" * 64}]
    staging = {
        "model_revision": runner.MODEL_REVISION,
        "prompt_ids_sha256": "sha256:" + "b" * 64,
        "inventory": inventory,
    }
    prompts = {"prompt_ids_sha256": staging["prompt_ids_sha256"]}
    monkeypatch.setattr(runner, "_inventory", lambda _: inventory)

    runner._verify_staging_binding(staging, prompts, tmp_path)
    with pytest.raises(runner.VLLMCompileContractError, match="prompt receipts"):
        runner._verify_staging_binding(
            staging,
            {"prompt_ids_sha256": "sha256:" + "c" * 64},
            tmp_path,
        )
