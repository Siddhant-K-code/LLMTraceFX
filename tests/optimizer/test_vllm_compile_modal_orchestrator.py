"""Offline execution-controller tests; no provider or Modal operation is real."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import modal_orchestrator as controller
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
    CELLS,
    EXPECTED_MODEL_BYTES,
    MODEL_ID,
    MODEL_REVISION,
    workload_descriptors,
)

HEAD = "a" * 40
PAGE_HASH = "sha256:" + "b" * 64
VOLUME_HASH = "sha256:" + "c" * 64
APPROVAL_BYTES = b"Coordinator-approved bounded execution plan\n"
APPROVAL_SHA256 = controller._sha256_bytes(APPROVAL_BYTES)


@pytest.fixture(autouse=True)
def approved_plan_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller, "APPROVED_PLAN_SHA256", APPROVAL_SHA256)


@pytest.fixture
def paths(request: pytest.FixtureRequest) -> Iterator[dict[str, Path]]:
    root = Path(".cache/llmtracefx-tests/modal-controller") / request.node.name
    shutil.rmtree(root, ignore_errors=True)
    workspace = root / "repo"
    output = root / "output"
    workspace.mkdir(parents=True)
    output.mkdir()
    result = {
        "root": root,
        "workspace": workspace,
        "output": output,
        "ledger": root / "ledger.json",
        "approval": root / "approval.json",
    }
    try:
        yield result
    finally:
        shutil.rmtree(root, ignore_errors=True)


class FakeGit:
    def root(self, workspace: Path) -> Path:
        return workspace

    def head(self, workspace: Path) -> str:
        return HEAD

    def is_clean(self, workspace: Path) -> bool:
        return True


class NeverProvider:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"provider method called during offline refusal: {name}")


def raw(payload: Any, marker: bytes = b"provider-json") -> controller.RawJSON:
    return controller.RawJSON(payload, controller._sha256_bytes(marker))


def rates(multiplier: Decimal = Decimal(1)) -> controller.RawJSON:
    baseline = {
        "gpu_hour_cost_l40s": Decimal("1.95000"),
        "gpu_hour_cost_h100": Decimal("3.95000"),
        "cpu_hour_cost": Decimal("0.04730"),
        "mem_gib_hour_cost": Decimal("0.00800"),
        "volume_storage_gib_month_cost": Decimal("0.09000"),
    }
    return raw(
        {resource: str(value * multiplier) for resource, value in baseline.items()},
        b"rates",
    )


def empty_inventory() -> controller.RawJSON:
    return raw([])


def approval(paths: dict[str, Path], experiment_id: str = "run-01") -> str:
    paths["approval"].write_bytes(APPROVAL_BYTES)
    return APPROVAL_SHA256


def config(
    paths: dict[str, Path], experiment_id: str = "run-01"
) -> controller.ExecutionConfig:
    return controller.ExecutionConfig(
        approval_path=paths["approval"],
        approval_sha256=approval(paths, experiment_id),
        git_head=HEAD,
        workspace_path=paths["workspace"],
        output_dir=paths["output"],
        ledger_path=paths["ledger"],
        experiment_id=experiment_id,
    )


def pages() -> controller.PagePolicySnapshot:
    return controller.PagePolicySnapshot(200, PAGE_HASH, 200, VOLUME_HASH)


class FakePagePolicy:
    def __init__(self, snapshot: controller.PagePolicySnapshot | None = None) -> None:
        self.snapshot = snapshot or pages()

    def fetch(self) -> controller.PagePolicySnapshot:
        return self.snapshot


class FakeProvider:
    def __init__(self, rate_response: controller.RawJSON | None = None) -> None:
        self.calls: list[str] = []
        self.rate_response = rate_response or rates()
        self.created = False
        self.deleted = False
        self.fail_create = False
        self.after_ambiguous = False

    def version(self) -> str:
        self.calls.append("version")
        return "1.5.4"

    def authenticate(self) -> str:
        self.calls.append("authenticate")
        return "sha256:" + "f" * 64

    def billing_rates(self) -> controller.RawJSON:
        self.calls.append("rates")
        return self.rate_response

    def billing_summary(self) -> controller.OptionalProviderJSON:
        self.calls.append("billing")
        return controller.OptionalProviderJSON(None, "unsupported")

    def app_inventory(self) -> controller.RawJSON:
        self.calls.append("apps")
        if self.after_ambiguous and self.deleted:
            return raw({"wrong": []})
        return empty_inventory()

    def volume_inventory(self) -> controller.RawJSON:
        self.calls.append("volumes")
        return empty_inventory()

    def container_inventory(self) -> controller.RawJSON:
        self.calls.append("containers")
        return empty_inventory()

    def secret_inventory(self) -> controller.RawJSON:
        self.calls.append("secrets")
        return empty_inventory()

    def create_volume(self, name: str) -> None:
        self.calls.append("create")
        if self.fail_create:
            raise RuntimeError("provider private account ac-private-secret")
        self.created = True

    def stop_app(self, name: str) -> None:
        self.calls.append("stop")

    def delete_volume(self, name: str, *, allow_missing: bool) -> None:
        assert allow_missing
        self.calls.append("delete")
        self.deleted = True


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = controller._sha256_json(result)
    return result


def staging(plan: Any) -> dict[str, Any]:
    source_hashes = {
        f"{item.context_tier}/{item.workload_id}": item.prompt_sha256
        for item in workload_descriptors()
    }
    prompts = []
    for index, (tier, workload) in enumerate(
        (tier, workload)
        for tier in ("2k", "8k", "16k")
        for workload in (
            "structured-json-profile-extraction",
            "prose-reasoning-two-train-gap",
        )
    ):
        token_ids = [index, index + 1]
        prompts.append(
            {
                "key": f"{tier}/{workload}",
                "prompt_sha256": source_hashes[f"{tier}/{workload}"],
                "prompt_token_ids_sha256": controller._sha256_json(token_ids),
                "prompt_token_ids": token_ids,
                "input_token_count": len(token_ids),
                "decoded_prompt_sha256": "sha256:" + f"{index + 10:064x}",
            }
        )
    return _seal(
        {
            "schema_version": "1",
            "plan_sha256": plan.content_sha256,
            "workload_sha256": controller._sha256_json(
                controller._HARNESS_WORKLOAD_PAYLOAD
            ),
            "output_contract_sha256": controller._sha256_json(
                controller._HARNESS_OUTPUT_PAYLOAD
            ),
            "runtime_sha256": controller._sha256_json(controller.RUNTIME_PINS),
            "image_sha256": controller._sha256_json(
                {
                    "reference": (
                        "vllm/vllm-openai:v0.28.0@"
                        + controller.OFFICIAL_VLLM_IMAGE_DIGEST
                    )
                }
            ),
            "image_digest": controller.OFFICIAL_VLLM_IMAGE_DIGEST,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_file_count": 15,
            "model_bytes": EXPECTED_MODEL_BYTES,
            "inventory": controller._expected_model_inventory(),
            "prompts": prompts,
            "staged_at": "2026-09-03T00:00:00+00:00",
            "prompt_ids_sha256": controller._sha256_json(
                {
                    "schema_version": "1",
                    "workload_sha256": controller._sha256_json(
                        controller._HARNESS_WORKLOAD_PAYLOAD
                    ),
                    "prompts": {
                        item["key"]: item["prompt_token_ids"] for item in prompts
                    },
                }
            ),
        },
        "receipt_sha256",
    )


def terminal(plan: Any, receipt: dict[str, Any], index: int) -> dict[str, Any]:
    counts = {item["key"]: item["input_token_count"] for item in receipt["prompts"]}
    requests = []
    base = datetime(2026, 9, 3, tzinfo=timezone.utc)
    for descriptor in workload_descriptors():
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        requests.append(
            {
                **descriptor.to_dict(),
                "terminal": True,
                "finish_reason": "length",
                "correctness": None,
                "input_token_count": counts[key],
                "input_token_ids_sha256": next(
                    item["prompt_token_ids_sha256"]
                    for item in receipt["prompts"]
                    if item["key"] == key
                ),
                "output_token_ids": [1, 2],
                "output_token_count": 2,
                "output_tokens_per_second": 2_000_000.0,
                "output_rate_basis": "output_tokens_per_complete_response_second",
                "decoded_output": "evaluate locally",
                "evaluator_input": {
                    "workload_id": descriptor.workload_id,
                    "context_tier": descriptor.context_tier,
                    "decoded_output": "evaluate locally",
                    "output_token_ids": [1, 2],
                },
                "provenance": "model_reported",
                "started_at": (
                    base + timedelta(microseconds=descriptor.ordinal + 1)
                ).isoformat(),
                "ended_at": (
                    base + timedelta(microseconds=descriptor.ordinal + 2)
                ).isoformat(),
                "wall_clock_seconds": 0.000001,
                "ttft_seconds": None,
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
    record = _seal(
        {
            "schema_version": "1",
            "cell": CELLS[index].to_dict(),
            "plan_sha256": plan.content_sha256,
            "staging_receipt_sha256": receipt["receipt_sha256"],
            "workload_sha256": controller._sha256_json(
                controller._HARNESS_WORKLOAD_PAYLOAD
            ),
            "output_contract_sha256": controller._sha256_json(
                controller._HARNESS_OUTPUT_PAYLOAD
            ),
            "runtime_sha256": controller._sha256_json(controller.RUNTIME_PINS),
            "image_sha256": controller._sha256_json(
                {
                    "reference": (
                        "vllm/vllm-openai:v0.28.0@"
                        + controller.OFFICIAL_VLLM_IMAGE_DIGEST
                    )
                }
            ),
            "hardware": {
                "gpu_name": ("NVIDIA L40S" if index < 2 else "NVIDIA H100 80GB HBM3"),
                "gpu_count": 1,
                "driver_version": "570.1",
                "memory_total_mib": 48_000.0 if index < 2 else 80_000.0,
                "memory_used_mib": 100.0,
            },
            "runtime": controller.RUNTIME_PINS,
            "resolved_execution_config": (
                {
                    "enforce_eager": False,
                    "compilation_mode": "VLLM_COMPILE",
                    "cuda_graph_mode": "FULL_AND_PIECEWISE",
                }
                if CELLS[index].compile_enabled
                else {
                    "enforce_eager": True,
                    "compilation_mode": "NONE",
                    "cuda_graph_mode": "NONE",
                }
            ),
            "initialization_started_at": base.isoformat(),
            "initialization_ready_at": (base + timedelta(microseconds=1)).isoformat(),
            "compilation_seconds": None,
            "compilation_seconds_unobservable_reason": (
                "vllm_compilation_time_not_exposed_or_nonpositive"
                if CELLS[index].compile_enabled
                else "not_applicable_eager_mode"
            ),
            "cuda_graph_seconds": None,
            "cuda_graph_seconds_unobservable_reason": (
                "stable_component_timing_not_exposed"
                if CELLS[index].compile_enabled
                else "not_applicable_eager_mode"
            ),
            "peak_gpu_memory_mib": 1024.0,
            "terminal": True,
            "correctness_evaluated_remotely": False,
            "requests": requests,
        },
        "cell_sha256",
    )
    return {
        "event": "cell_terminal",
        "provenance": "derived",
        "record": record,
    }


class FakeHarnessLoader:
    def __init__(
        self, *, bad_cell: int | None = None, fail_stage: bool = False
    ) -> None:
        self.sequence: list[str] = []
        self.bad_cell = bad_cell
        self.fail_stage = fail_stage

    def load(self, environment: Mapping[str, str]) -> Any:
        assert all(os.environ.get(key) == value for key, value in environment.items())
        plan = controller.VLLMCompilePlan.from_json(
            environment["LLMTRACEFX_QWEN3_COMPILE_PLAN_JSON"]
        )
        receipt = staging(plan)
        sequence = self.sequence

        class Stage:
            def remote(self) -> dict[str, Any]:
                sequence.append("stage")
                if self_outer.fail_stage:
                    raise RuntimeError("stage failure")
                return receipt

        class Cell:
            def __init__(self, index: int) -> None:
                self.index = index

            def remote_gen(self) -> Iterator[dict[str, Any]]:
                sequence.append(controller.CELL_FUNCTIONS[self.index])
                event = terminal(plan, receipt, self.index)
                if self_outer.bad_cell == self.index:
                    event["record"]["requests"][0]["finish_reason"] = ""
                yield {
                    "event": "container_started",
                    "provenance": "modal_provider",
                }
                yield {
                    "event": "hardware_validated",
                    "provenance": "cuda",
                }
                yield {
                    "event": "initialization_started",
                    "provenance": "client_observed",
                }
                yield {
                    "event": "initialization_ready",
                    "provenance": "vllm",
                }
                for request in event["record"]["requests"]:
                    yield {
                        "event": "request_terminal",
                        "provenance": "model_reported",
                        "request": request,
                    }
                yield event

        self_outer = self
        values = {
            name: Cell(index) for index, name in enumerate(controller.CELL_FUNCTIONS)
        }
        return SimpleNamespace(
            APP_NAME=f"qwen3-compile-{environment['LLMTRACEFX_QWEN3_COMPILE_EXPERIMENT_TAG']}",
            CELL_FUNCTIONS=tuple(values.values()),
            app=SimpleNamespace(run=lambda **_: nullcontext()),
            stage_qwen3=Stage(),
            **values,
        )


def execute_ok(
    paths: dict[str, Path],
    provider: FakeProvider,
    harness: FakeHarnessLoader,
) -> dict[str, Any]:
    return controller.execute(
        config(paths),
        provider=provider,
        page_policy=FakePagePolicy(),
        harness_loader=harness,
        git=FakeGit(),
        environ={},
        today=lambda: date(2026, 9, 3),
    )


def test_offline_preflight_never_accesses_provider(paths: dict[str, Path]) -> None:
    result = controller.offline_preflight(
        config(paths),
        harness_loader=FakeHarnessLoader(),
        git=FakeGit(),
        environ={},
    )

    assert result["provider_accessed"] is False
    assert result["cells"] == [cell.cell_id for cell in CELLS]
    assert not paths["ledger"].exists()
    assert not any(paths["output"].iterdir())


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "head",
        "dirty",
        "inside-output",
        "existing-ledger",
        "unsafe-id",
        "forbidden-env",
        "page-failure",
    ],
)
def test_all_offline_gates_precede_auth(paths: dict[str, Path], mutation: str) -> None:
    cfg = config(paths)
    git: Any = FakeGit()
    environ: dict[str, str] = {}
    policy = FakePagePolicy()
    if mutation == "hash":
        cfg = dataclass_replace(cfg, approval_sha256="sha256:" + "0" * 64)
    elif mutation == "head":
        cfg = dataclass_replace(cfg, git_head="f" * 40)
    elif mutation == "dirty":
        git = SimpleNamespace(
            root=lambda path: path, head=lambda _: HEAD, is_clean=lambda _: False
        )
    elif mutation == "inside-output":
        inside = paths["workspace"] / "output"
        inside.mkdir()
        cfg = dataclass_replace(cfg, output_dir=inside)
    elif mutation == "existing-ledger":
        paths["ledger"].write_text("existing")
    elif mutation == "unsafe-id":
        cfg = dataclass_replace(cfg, experiment_id="../bad")
    elif mutation == "forbidden-env":
        environ["HF_TOKEN"] = "never-read"
    else:
        policy = FakePagePolicy(
            controller.PagePolicySnapshot(None, None, None, None, "unavailable")
        )
    with pytest.raises(controller.ModalOrchestratorError):
        controller.execute(
            cfg,
            provider=NeverProvider(),
            page_policy=policy,
            harness_loader=SimpleNamespace(),
            git=git,
            environ=environ,
        )


def dataclass_replace(
    value: controller.ExecutionConfig, **changes: Any
) -> controller.ExecutionConfig:
    data = dict(value.__dict__)
    data.update(changes)
    return controller.ExecutionConfig(**data)


def test_reserves_every_line_before_create_and_runs_sequentially(
    paths: dict[str, Path],
) -> None:
    provider = FakeProvider()
    harness = FakeHarnessLoader()
    state = execute_ok(paths, provider, harness)

    assert provider.calls.index("create") > provider.calls.index("containers")
    ledger = json.loads(paths["ledger"].read_text())
    assert [item["line_id"] for item in ledger["events"]] == [
        "image-allowance",
        "staging",
        "cell-l40s-eager",
        "cell-l40s-compiled",
        "cell-h100-eager",
        "cell-h100-compiled",
        "storage",
    ]
    assert Decimal(ledger["reserved_usd"]) + Decimal(ledger["remaining_usd"]) == 28
    assert harness.sequence == ["stage", *controller.CELL_FUNCTIONS]
    assert state["status"] == "complete"
    assert provider.calls[-7:] == [
        "apps",
        "delete",
        "apps",
        "volumes",
        "containers",
        "secrets",
        "billing",
    ]


def test_real_orchestrator_output_builds_public_bundle(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evidence, "APPROVED_PLAN_SHA256", APPROVAL_SHA256)
    execute_ok(paths, FakeProvider(), FakeHarnessLoader())
    bundle = paths["output"].parent / "public-bundle"
    bundle.mkdir()

    evidence.build_from_execution_directory(paths["output"], bundle)

    assert evidence.verify_bundle(bundle)["requests_verified"] == 48


def test_teardown_inventory_excludes_unrelated_account_resources(
    paths: dict[str, Path],
) -> None:
    class ProviderWithUnrelatedResources(FakeProvider):
        def app_inventory(self) -> controller.RawJSON:
            self.calls.append("apps")
            return raw([{"description": "unrelated-app", "state": "deployed"}])

        def volume_inventory(self) -> controller.RawJSON:
            self.calls.append("volumes")
            return raw([{"name": "unrelated-volume"}])

        def container_inventory(self) -> controller.RawJSON:
            self.calls.append("containers")
            return raw([{"app_name": "unrelated-app"}])

        def secret_inventory(self) -> controller.RawJSON:
            self.calls.append("secrets")
            return raw([{"name": "unrelated-secret"}])

    execute_ok(paths, ProviderWithUnrelatedResources(), FakeHarnessLoader())
    teardown = json.loads((paths["output"] / "teardown-report.json").read_text())

    assert teardown["complete"] is True
    assert teardown["provider_inventory_after"] == {
        kind: {"count": 0, "status_counts": {}}
        for kind in ("apps", "volumes", "containers", "secrets")
    }


def test_teardown_waits_for_stopping_app_to_disappear(
    paths: dict[str, Path],
) -> None:
    class SettlingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.app_inventory_calls = 0

        def app_inventory(self) -> controller.RawJSON:
            self.calls.append("apps")
            self.app_inventory_calls += 1
            if self.app_inventory_calls in {2, 3}:
                return raw(
                    [
                        {
                            "description": "qwen3-compile-run-01",
                            "state": "stopping...",
                        }
                    ]
                )
            return empty_inventory()

    sleeps: list[float] = []
    provider = SettlingProvider()
    state = controller.execute(
        config(paths),
        provider=provider,
        page_policy=FakePagePolicy(),
        harness_loader=FakeHarnessLoader(),
        git=FakeGit(),
        environ={},
        today=lambda: date(2026, 9, 3),
        sleeper=sleeps.append,
    )

    assert state["status"] == "complete"
    assert sleeps == [controller.TEARDOWN_SETTLE_SECONDS]
    assert "stop" in provider.calls


def test_teardown_settle_window_is_bounded() -> None:
    class NeverSettles(FakeProvider):
        def app_inventory(self) -> controller.RawJSON:
            return raw(
                [{"description": "qwen3-compile-run-01", "state": "stopping..."}]
            )

    sleeps: list[float] = []
    with pytest.raises(controller.ModalOrchestratorError, match="did not settle"):
        controller._settle_teardown_inventory(
            NeverSettles(),
            app_name="qwen3-compile-run-01",
            volume_name="qwen3-compile-volume-run-01",
            experiment_id="run-01",
            sleeper=sleeps.append,
        )

    assert len(sleeps) == controller.TEARDOWN_SETTLE_ATTEMPTS - 1


def test_paired_hardware_must_match() -> None:
    eager = {
        "hardware": {
            "gpu_name": "NVIDIA L40S",
            "driver_version": "570.1",
            "memory_total_mib": 48_000.0,
        }
    }
    compiled = {
        "hardware": {
            "gpu_name": " NVIDIA  L40S ",
            "driver_version": "571.0",
            "memory_total_mib": 48_000.0,
        }
    }

    with pytest.raises(controller.ModalOrchestratorError, match="paired cell"):
        controller._validate_paired_hardware(eager, compiled)


def test_live_rate_increase_recalculates_and_refuses_before_paid_command(
    paths: dict[str, Path],
) -> None:
    provider = FakeProvider(rates(Decimal("2")))
    with pytest.raises(controller.ModalOrchestratorError):
        execute_ok(paths, provider, FakeHarnessLoader())
    assert "create" not in provider.calls


@pytest.mark.parametrize(
    "mutation",
    ["missing", "invalid", "not-object"],
)
def test_rate_parser_refuses_nonexact_modal_schema(mutation: str) -> None:
    response = rates()
    if mutation == "missing":
        response.payload.pop("gpu_hour_cost_l40s")
    elif mutation == "invalid":
        response.payload["gpu_hour_cost_l40s"] = "NaN"
    else:
        response = raw([])
    with pytest.raises(controller.ModalOrchestratorError):
        controller.parse_billing_rates(response)


def test_rate_parser_converts_hourly_modal_rates_to_plan_units() -> None:
    parsed = controller.parse_billing_rates(rates())

    assert parsed == {
        "l40s_gpu_second_usd": "0.0005416666666666666666666666667",
        "h100_gpu_second_usd": "0.001097222222222222222222222222",
        "cpu_core_second_usd": "0.00001313888888888888888888888889",
        "memory_gib_second_usd": "0.000002222222222222222222222222222",
        "volume_gib_month_usd": "0.09",
    }


def test_app_inventory_preserves_private_identifier_for_teardown_only() -> None:
    provider_id = "ap-" + "A1b2C3d4E5f6"
    parsed = controller._parse_inventory(
        raw(
            [
                {
                    "app_id": provider_id,
                    "description": "qwen3-compile-run-01",
                    "state": "ephemeral",
                }
            ]
        ),
        "app",
    )

    assert parsed[0].provider_id == provider_id
    assert controller._inventory_facts(parsed) == {
        "count": 1,
        "status_counts": {"ephemeral": 1},
    }


@pytest.mark.parametrize("prefix", ["ak-", "as-"])
def test_persistence_scanner_rejects_modal_credential_shapes(prefix: str) -> None:
    with pytest.raises(controller.ModalOrchestratorError, match="credential-shaped"):
        controller._scan_persistable({"diagnostic": prefix + "A1b2C3d4E5f6G7h8I9j0"})


@pytest.mark.parametrize("failure", ["stage", "cell", "provider"])
def test_teardown_is_mandatory_and_original_failure_preserved(
    paths: dict[str, Path], failure: str
) -> None:
    provider = FakeProvider()
    harness = FakeHarnessLoader(
        fail_stage=failure == "stage", bad_cell=0 if failure == "cell" else None
    )
    provider.fail_create = failure == "provider"
    with pytest.raises(controller.ModalOrchestratorError) as caught:
        execute_ok(paths, provider, harness)
    if failure == "provider":
        assert "stop" not in provider.calls
        assert provider.calls.count("delete") == 1
    else:
        assert provider.calls.count("stop") == 0
        assert provider.calls.count("delete") == 1
    assert caught.value.original is not None
    persisted = "".join(path.read_text() for path in paths["output"].glob("*.json"))
    assert "ac-private-secret" not in persisted
    assert "never-read" not in persisted


def test_invalid_cell_stops_before_next_and_correctness_remains_null(
    paths: dict[str, Path],
) -> None:
    provider = FakeProvider()
    harness = FakeHarnessLoader(bad_cell=1)
    with pytest.raises(controller.ModalOrchestratorError):
        execute_ok(paths, provider, harness)
    assert harness.sequence == ["stage", "l40s_eager", "l40s_compiled"]
    first = json.loads((paths["output"] / "l40s_eager-terminal.json").read_text())
    assert all(item["correctness"] is None for item in first["requests"])
    assert not (paths["output"] / "h100_eager-lifecycle.json").exists()


def test_ambiguous_post_delete_inventory_makes_result_incomplete(
    paths: dict[str, Path],
) -> None:
    provider = FakeProvider()
    provider.after_ambiguous = True
    with pytest.raises(controller.ModalOrchestratorError, match="teardown"):
        execute_ok(paths, provider, FakeHarnessLoader())
    report = json.loads((paths["output"] / "teardown-report.json").read_text())
    assert report["complete"] is False
    assert report["inventory_status"] == "incomplete"
    assert report["billing_after"] is None
    assert report["billing_after_unavailable_reason"] == "unsupported"
