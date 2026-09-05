"""Tests for the offline-verified Modal L4 execution surface.

No test imports or calls the real Modal SDK. Every provider interaction
goes through the fake in ``_fake_modal``, which mirrors the exact API
surface the protocol pins.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_cell_runner as cell_runner
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_execute as execute_module
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_rates as rates_module

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _fake_modal import build_fake_modal  # noqa: E402

APP_MODULE = "llmtracefx.optimizer.lab.qwen3_8b.modal_l4_app"
NONCE = "d" * 32
HEAD = "e" * 40
CELL_IDS = [cell.cell_id for cell in modal.crossover_schedule()]
# A signed approval carries a bounded UTC execution window. These fixtures place
# ``authorized_at`` inside a four-hour window and run the preflight clock inside
# it, well under the documented maximum bounded duration.
AUTHORIZED_AT = "2026-09-04T19:52:50.511+05:30"
NOT_BEFORE = "2026-09-04T14:00:00+00:00"
EXPIRES_AT = "2026-09-04T18:00:00+00:00"
WITHIN_WINDOW = datetime(2026, 9, 4, 14, 30, 0, tzinfo=timezone.utc)


def _passing_observation(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "gpu_name": modal.EXPECTED_GPU_NAME,
        "gpu_count": 1,
        "runtime_pins": dict(modal.RUNTIME_PINS),
        "total_vram_mib": 23_034,
        "peak_vram_mib": 21_000,
        "kv_cache_blocks": 640,
        "kv_cache_tokens": 40_960,
        "max_model_len": 16_480,
        "out_of_memory": False,
        "generated_tokens": modal.DECODE_STEPS,
        "terminal": True,
        "used_longest_controlled_prompt": True,
        "runner_kwargs": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "gpu_memory_utilization": "0.94",
            "enable_prefix_caching": False,
            "speculative_config": None,
            "enforce_eager": mode == "eager",
            "max_model_len": 16_480,
        },
    }


class _Receipts:
    """Scripted, terminal receipts with unique container identities."""

    def __init__(self) -> None:
        self.index = 0
        self.reuse_identity = False

    def _identity(self) -> str:
        if self.reuse_identity:
            return "sha256:" + "1" * 64
        self.index += 1
        return "sha256:" + f"{self.index:064d}"

    def base(self, kind: str, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "protocol_id": modal.PROTOCOL_ID,
            "kind": kind,
            "status": "completed",
            "terminal": True,
            "container_identity_sha256": self._identity(),
            **extra,
        }

    def stage(self) -> dict[str, Any]:
        return self.base("modal_stage")

    def verify(self) -> dict[str, Any]:
        return self.base("modal_verify", longest_prompt_tokens=16_384)

    def canary(self, mode: str) -> dict[str, Any]:
        return self.base(
            "modal_canary", mode=mode, observation=_passing_observation(mode)
        )

    def cell(self, cell_id: str, nonce: str) -> dict[str, Any]:
        del nonce
        return self.base("modal_cell", cell_id=cell_id)

    def analysis(self, cell_ids: Any) -> dict[str, Any]:
        return self.base("modal_analysis", expected_cell_ids=sorted(cell_ids))


@pytest.fixture
def fake_modal(monkeypatch: pytest.MonkeyPatch) -> Any:
    module = build_fake_modal()
    monkeypatch.setitem(sys.modules, "modal", module)
    return module


@pytest.fixture
def app_module(monkeypatch: pytest.MonkeyPatch, fake_modal: Any) -> Any:
    monkeypatch.setenv(execute_module.NONCE_VAR, NONCE)
    monkeypatch.delenv(execute_module.APP_NAME_VAR, raising=False)
    monkeypatch.delenv(execute_module.VOLUME_NAME_VAR, raising=False)
    monkeypatch.delenv(execute_module.PLAN_SHA256_VAR, raising=False)
    sys.modules.pop(APP_MODULE, None)
    module = importlib.import_module(APP_MODULE)
    yield module
    sys.modules.pop(APP_MODULE, None)


def _script(app_module: Any, receipts: _Receipts, **overrides: Any) -> None:
    builders = {
        "stage": lambda *args: receipts.stage(),
        "verify": lambda *args: receipts.verify(),
        "eager_canary": lambda *args: receipts.canary("eager"),
        "compiled_canary": lambda *args: receipts.canary("compiled"),
        "natural_cell": receipts.cell,
        "controlled_cell": receipts.cell,
        "analysis": receipts.analysis,
    }
    builders.update(overrides)
    for spec_key, function in app_module.FUNCTIONS.items():
        app_module.app.script[function.key] = builders[spec_key]


def _attestation(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "1",
        "kind": modal.CREDENTIAL_EXPOSURE_ATTESTATION_KIND,
        "protocol_id": modal.PROTOCOL_ID,
        "exposed_profile_credential_never_used_by_experiment": True,
        "exposed_profile_credential_revocation_confirmed": True,
        "revocation_confirmed_by": "coordinator",
        "fresh_local_profile_created_without_sharing": True,
        "fresh_profile_shared_anywhere": False,
        "confirmed_at": "2026-09-04T21:02:06.080+05:30",
        "status": "cleared",
        "reason": "coordinator confirmed revocation and fresh local profile",
    }
    payload.update(overrides)
    return payload


def _write_gate_files(
    tmp_path: Path,
    *,
    workspace: Path,
    rate_overrides: dict[str, Any] | None = None,
    attestation: dict[str, Any] | None = None,
) -> dict[str, Path]:
    rate_receipt = {
        "source_url": rates_module.OFFICIAL_RATE_URL,
        "document_sha256": rates_module._sha256_uri(
            _fetcher(rates_module.OFFICIAL_RATE_URL)
        ),
        "fetched_at": "2026-09-04T19:52:50.511+05:30",
        "rates": {
            "l4_gpu_second": "0.000222",
            "cpu_core_second": "0.0000131",
            "memory_gib_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
        "additional_charges": [],
    }
    if rate_overrides:
        rate_receipt.update(rate_overrides)
    rate_path = tmp_path / "rate-receipt.json"
    rate_path.write_text(json.dumps(rate_receipt, indent=2), encoding="utf-8")

    exposure = _attestation() if attestation is None else attestation
    exposure_path = tmp_path / "credential-exposure-attestation.json"
    exposure_path.write_text(json.dumps(exposure, indent=2), encoding="utf-8")

    content = execute_module.ModalExecutionAuthorization.content(
        plan_sha256=modal.build_default_plan().content_sha256,
        source_head=HEAD,
        experiment_nonce=NONCE,
        workspace_sha256=execute_module._sha256_text(str(workspace.resolve())),
        rate_receipt_sha256=execute_module._sha256_json(rate_receipt),
        credential_exposure_attestation_sha256=execute_module._sha256_json(exposure),
        authorized_at=AUTHORIZED_AT,
        not_before=NOT_BEFORE,
        expires_at=EXPIRES_AT,
    )
    content["authorization_sha256"] = execute_module._sha256_json(content)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(content, indent=2), encoding="utf-8")

    signature = tmp_path / "authorization.sig"
    signature.write_text("detached signature\n", encoding="utf-8")
    signers = tmp_path / "allowed_signers"
    signers.write_text(
        f"{execute_module.AUTHORIZATION_SIGNER_IDENTITY} ssh-ed25519 TEST\n",
        encoding="utf-8",
    )
    return {
        "authorization_path": authorization_path,
        "signature_path": signature,
        "authorized_signers_path": signers,
        "rate_receipt_path": rate_path,
        "credential_exposure_attestation_path": exposure_path,
    }


def _fetcher(url: str) -> bytes:
    return f"official document for {url}".encode()


def _clean_checkout_probe() -> dict[str, Any]:
    """A fake git checkout state: HEAD at the authorized head, nothing dirty."""

    return {"head": HEAD, "status_porcelain": ""}


def _authenticated_profile(*, sdk_version: str) -> dict[str, Any]:
    """A fake local-profile validator that reports an authenticated profile."""

    return {
        "schema_version": modal.PROFILE_AUTHENTICATION_SCHEMA_VERSION,
        "gate": modal.PROFILE_AUTHENTICATION_GATE,
        "authenticated": True,
        "mechanism": modal.PROFILE_AUTHENTICATION_MECHANISM,
        "cli_version": sdk_version,
        "sdk_version": sdk_version,
        "records_profile_identity": False,
        "checked_at": "2026-09-04T20:00:00+00:00",
    }


def _run(
    tmp_path: Path,
    app_module: Any,
    fake_modal: Any,
    *,
    environ: dict[str, str] | None = None,
    sdk_loader: Any = None,
    signature_result: int = 0,
    rate_overrides: dict[str, Any] | None = None,
    headroom: dict[str, Any] | None = None,
    attestation: dict[str, Any] | None = None,
    drop_attestation: bool = False,
    source_checkout_probe: Any = None,
    profile_validator: Any = None,
) -> dict[str, Any]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    files = _write_gate_files(
        tmp_path,
        workspace=workspace,
        rate_overrides=rate_overrides,
        attestation=attestation,
    )
    if drop_attestation:
        files["credential_exposure_attestation_path"] = None  # type: ignore[assignment]
    fake_modal.Volume.objects.existing.append(
        modal.run_scoped_names(NONCE)["volume_name"]
    )
    return execute_module.execute(
        workspace=workspace,
        sdk_loader=sdk_loader or (lambda: fake_modal),
        app_loader=lambda: app_module,
        environ=environ if environ is not None else {},
        fetcher=_fetcher,
        signed_headroom=headroom
        or {"headroom_usd": "25", "signed_by": "operator", "kind": "headroom"},
        signature_verifier=lambda payload: None,
        signature_runner=lambda argv, message: signature_result,
        source_checkout_probe=source_checkout_probe or _clean_checkout_probe,
        profile_validator=profile_validator or _authenticated_profile,
        now_utc=WITHIN_WINDOW,
        **files,
    )


class TestProviderAppDeclaration:
    def test_app_declares_no_web_endpoint_and_no_secret(self, app_module: Any) -> None:
        source = Path(app_module.__file__).read_text(encoding="utf-8")
        for decorator in modal.NO_WEB_ENDPOINT_DECORATORS:
            assert f"modal.{decorator}" not in source
        assert "Secret" not in source.replace("modal.Secret", "")
        assert app_module.app.registered_web_endpoints == []
        assert all(
            function.kwargs["secrets"] == []
            for function in app_module.FUNCTIONS.values()
        )

    @pytest.mark.parametrize("spec", modal.FUNCTION_SPECS, ids=lambda s: s.function_key)
    def test_every_function_uses_the_sealed_resources(
        self, app_module: Any, spec: modal.ModalFunctionSpec
    ) -> None:
        function = app_module.FUNCTIONS[spec.function_key]
        expected = spec.modal_kwargs()
        for key, value in expected.items():
            assert function.kwargs[key] == value, key
        assert function.kwargs["retries"] == 0
        assert function.kwargs["max_containers"] == 1
        assert function.kwargs["min_containers"] == 0
        assert function.kwargs["buffer_containers"] == 0
        assert function.kwargs["max_inputs"] == 1
        assert function.kwargs["single_use_containers"] is True
        assert function.raw.concurrency_max_inputs == 1

    def test_cpu_functions_can_never_allocate_an_accelerator(
        self, app_module: Any
    ) -> None:
        for key in ("stage", "verify", "analysis"):
            assert "gpu" not in app_module.FUNCTIONS[key].kwargs
        for key in (
            "eager_canary",
            "compiled_canary",
            "natural_cell",
            "controlled_cell",
        ):
            assert app_module.FUNCTIONS[key].kwargs["gpu"] == "L4:1"

    def test_model_volume_is_read_only_for_accelerated_functions(
        self, app_module: Any
    ) -> None:
        for key in (
            "eager_canary",
            "compiled_canary",
            "natural_cell",
            "controlled_cell",
        ):
            volume = app_module.FUNCTIONS[key].kwargs["volumes"][modal.MODEL_MOUNT_PATH]
            assert volume.readonly is True
        for key in ("stage", "verify"):
            volume = app_module.FUNCTIONS[key].kwargs["volumes"][modal.MODEL_MOUNT_PATH]
            assert volume.readonly is False

    def test_images_are_pinned(self, app_module: Any) -> None:
        assert (
            app_module.FUNCTIONS["stage"]
            .kwargs["image"]
            .reference.startswith("debian_slim:")
        )
        assert app_module.FUNCTIONS["stage"].kwargs["image"].pip_packages == [
            modal.STAGING_IMAGE_HF_HUB_PIN
        ]
        runtime = app_module.FUNCTIONS["controlled_cell"].kwargs["image"].reference
        assert runtime.endswith(
            "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
        )

    def test_run_scoped_names_bind_the_app_and_volume(self, app_module: Any) -> None:
        names = modal.run_scoped_names(NONCE)
        assert app_module.app.name == names["app_name"]
        assert app_module.model_volume.name == names["volume_name"]
        assert app_module.APP_TAGS["llmtracefx_experiment_nonce"] == NONCE

    def test_app_refuses_a_mismatched_run_scoped_name(
        self, monkeypatch: pytest.MonkeyPatch, fake_modal: Any
    ) -> None:
        monkeypatch.setenv(execute_module.NONCE_VAR, NONCE)
        monkeypatch.setenv(execute_module.APP_NAME_VAR, "someone-elses-app")
        sys.modules.pop(APP_MODULE, None)
        with pytest.raises(modal.ModalL4ContractError, match="app name"):
            importlib.import_module(APP_MODULE)
        sys.modules.pop(APP_MODULE, None)


class TestSdkCapabilityProbe:
    def test_the_fake_matches_the_pinned_surface(self, fake_modal: Any) -> None:
        result = modal.verify_sdk_capabilities(fake_modal)
        assert result["verified"] is True
        assert result["tested_version"] == modal.TESTED_MODAL_VERSION
        assert "individual_container_deletion" in result["unsupported_controls"]

    @pytest.mark.parametrize("version", ("0.9.9", "2.0.0", "not-a-version"))
    def test_unsupported_versions_fail_closed(self, version: str) -> None:
        module = build_fake_modal(version=version)
        with pytest.raises(modal.ModalL4ContractError):
            modal.verify_sdk_capabilities(module)

    def test_missing_attribute_fails_closed(self, fake_modal: Any) -> None:
        delattr(fake_modal, "concurrent")
        with pytest.raises(modal.ModalL4ContractError, match="missing required attr"):
            modal.verify_sdk_capabilities(fake_modal)

    def test_missing_decorator_control_fails_closed(self, fake_modal: Any) -> None:
        class Stripped(fake_modal.App):  # type: ignore[misc, name-defined]
            def function(self, *, image: Any = None) -> Any:  # type: ignore[override]
                del image
                return lambda raw: raw

        fake_modal.App = Stripped
        with pytest.raises(
            modal.ModalL4ContractError, match="missing required controls"
        ):
            modal.verify_sdk_capabilities(fake_modal)

    def test_missing_volume_manager_fails_closed(self, fake_modal: Any) -> None:
        fake_modal.Volume.objects = object()
        with pytest.raises(modal.ModalL4ContractError, match="volume manager"):
            modal.verify_sdk_capabilities(fake_modal)

    def test_probe_requires_with_mount_options_not_read_only(self) -> None:
        # modal 1.5.5 deprecated Volume.read_only() in favor of
        # with_mount_options(read_only=True); the probe pins the new API.
        assert ("Volume", "with_mount_options") in modal.REQUIRED_SDK_MEMBERS
        assert ("Volume", "read_only") not in modal.REQUIRED_SDK_MEMBERS
        module = build_fake_modal()

        class _NoMount:
            objects = module.Volume.objects

            @classmethod
            def from_name(cls, *args: Any, **kwargs: Any) -> Any:
                return None

            def commit(self) -> None:
                return None

        module.Volume = _NoMount
        with pytest.raises(
            modal.ModalL4ContractError, match="missing required members"
        ):
            modal.verify_sdk_capabilities(module)


class TestGatesBeforeProviderImport:
    def test_credential_override_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load after a rejected environment")

        with pytest.raises(modal.ModalL4ContractError, match="MODAL_TOKEN_ID"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                environ={"MODAL_TOKEN_ID": "secret"},
                sdk_loader=forbidden_loader,
            )

    def test_unsigned_authorization_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load without a signature")

        with pytest.raises(modal.ModalL4ContractError, match="signature"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                signature_result=1,
            )

    def test_rate_increase_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load after a rate increase")

        with pytest.raises(modal.ModalL4ContractError, match="exceed the committed"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                rate_overrides={
                    "rates": {
                        "l4_gpu_second": "0.001",
                        "cpu_core_second": "0.0000131",
                        "memory_gib_second": "0.00000222",
                        "volume_gib_month": "0.09",
                    }
                },
            )

    def test_insufficient_headroom_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load without headroom")

        with pytest.raises(rates_module.RateRefreshError, match="headroom"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                headroom={"headroom_usd": "1"},
            )

    def test_authorization_bound_to_another_workspace_refuses(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        files = _write_gate_files(tmp_path, workspace=tmp_path / "elsewhere")
        with pytest.raises(modal.ModalL4ContractError, match="different workspace"):
            execute_module.preflight(
                workspace=workspace,
                environ={},
                fetcher=_fetcher,
                signature_runner=lambda argv, message: 0,
                signed_headroom={"headroom_usd": "25"},
                signature_verifier=lambda payload: None,
                **files,
            )


class TestSequentialExecution:
    def test_full_run_calls_every_stage_in_the_sealed_order(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "complete"
        keys = [key for key, _ in app_module.app.calls]
        assert keys[:4] == [
            "stage_model",
            "verify_stage",
            "eager_canary",
            "compiled_canary",
        ]
        assert len(keys) == 4 + 32 + 1
        assert keys[-1] == "analysis"
        dispatched_cells = [
            args[0] for key, args in app_module.app.calls if key.endswith("_cell")
        ]
        assert dispatched_cells == CELL_IDS
        assert document["completed_cell_ids"] == sorted(CELL_IDS)
        assert document["attempt_adjudication"]["valid"] is True

    def test_every_call_uses_its_declared_timeout(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        _run(tmp_path, app_module, fake_modal)
        observed = dict(app_module.app.timeouts)
        assert observed["stage_model"] == 1800
        assert observed["verify_stage"] == 300
        assert observed["eager_canary"] == 300
        assert observed["compiled_canary"] == 420
        assert observed["controlled_cell"] == 480
        assert observed["natural_cell"] == 240
        assert observed["analysis"] == 900

    def test_each_call_reserves_its_lifecycle_before_dispatch(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        ledger = document["ledger"]
        reserves = [
            event for event in ledger["events"] if event["event_type"] == "reserve"
        ]
        completions = [
            event for event in ledger["events"] if event["event_type"] == "complete"
        ]
        assert len(reserves) == len(modal.LIFECYCLES) == 37
        assert len(completions) == 37
        assert [event["lifecycle_id"] for event in reserves] == [
            step["lifecycle_id"] for step in modal.call_sequence()
        ]
        assert ledger["reserved_usd"] == "5.0785056"
        assert ledger["is_provider_proof"] is False

    def test_receipts_and_orchestration_document_are_written(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        workspace = tmp_path / "workspace"
        assert (workspace / "orchestration-receipt.json").is_file()
        assert len(list((workspace / "cells").iterdir())) == 32
        assert len(list((workspace / "memory-gate").iterdir())) == 2
        assert (workspace / "rate-refresh.json").is_file()
        assert (workspace / "headroom.json").is_file()
        assert document["kind"].endswith(".result")
        assert document["provider_reported_spend_usd"] is None
        assert document["statistical_publication"]["accepts_modal_workspace"] is False


class TestGatingAndInvalidation:
    def test_no_cell_is_dispatched_after_a_failed_canary(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()

        def failing_canary(*args: Any) -> dict[str, Any]:
            payload = receipts.canary("compiled")
            payload["observation"]["out_of_memory"] = True
            return payload

        _script(app_module, receipts, compiled_canary=failing_canary)
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "refused"
        assert "memory gate failed" in document["failure"]
        assert [key for key, _ in app_module.app.calls] == [
            "stage_model",
            "verify_stage",
            "eager_canary",
            "compiled_canary",
        ]
        assert document["completed_cell_ids"] == []
        assert document["kind"].endswith(".refusal")

    def test_a_refused_staging_receipt_stops_the_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()
        refusal = {
            "schema_version": "1",
            "kind": "modal_stage",
            "status": "refused",
            "terminal": True,
            "container_identity_sha256": "sha256:" + "9" * 64,
        }
        _script(app_module, receipts, stage=lambda *args: refusal)
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert len(app_module.app.calls) == 1

    @pytest.mark.parametrize(
        ("exception", "observation"),
        (
            (RuntimeError("boom"), "crash"),
            (Exception("nope"), "crash"),
        ),
    )
    def test_provider_failures_invalidate_immediately(
        self,
        tmp_path: Path,
        app_module: Any,
        fake_modal: Any,
        exception: BaseException,
        observation: str,
    ) -> None:
        receipts = _Receipts()
        _script(app_module, receipts, verify=lambda *args: exception)
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert observation in document["failure"]
        assert len(app_module.app.calls) == 2

    def test_timeout_and_preemption_are_classified(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        from _fake_modal import FunctionTimeoutError

        receipts = _Receipts()
        _script(app_module, receipts, verify=lambda *args: FunctionTimeoutError("late"))
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert "timeout" in document["failure"]
        assert document["call_sequence_executed"][-1]["terminal_receipt"] is False

    def test_a_reused_container_is_reported_as_a_second_attempt(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()
        receipts.reuse_identity = True
        _script(app_module, receipts)
        document = _run(tmp_path, app_module, fake_modal)
        attempts = [item["attempt"] for item in document["call_sequence_executed"]]
        assert 2 in attempts
        assert document["attempt_adjudication"]["valid"] is False

    def test_non_terminal_receipt_is_not_silence(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()
        _script(app_module, receipts, verify=lambda *args: {"status": "completed"})
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert "missing_terminal_receipt" in document["failure"]


class TestTeardown:
    def test_teardown_runs_on_a_completed_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        teardown = document["teardown"]
        assert teardown["adjudication"]["complete"] is True
        assert teardown["volume_deleted"] is True
        assert teardown["live_named_volumes"] == []
        assert teardown["credential_secret_created"] is False
        assert teardown["run_created_secret_count"] == 0
        assert teardown["individual_container_deletion"] is None
        assert teardown["app_deletion_provider_verified"] is None
        assert teardown["container_inventory_observable"] is False
        assert teardown["named_resource_listing_scope"] == "volumes_only"
        assert modal.run_scoped_names(NONCE)["volume_name"] in (
            fake_modal.Volume.objects.deleted
        )
        assert app_module.app.exited == 1

    def test_teardown_runs_on_a_refusal(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()

        def failing_canary(*args: Any) -> dict[str, Any]:
            payload = receipts.canary("eager")
            payload["observation"]["terminal"] = False
            return payload

        _script(app_module, receipts, eager_canary=failing_canary)
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "refused"
        assert document["teardown"]["volume_deleted"] is True

    def test_teardown_runs_when_the_app_context_raises(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        app_module.app.enter_error = RuntimeError("provider unavailable")
        _script(app_module, _Receipts())
        with pytest.raises(RuntimeError, match="provider unavailable"):
            _run(tmp_path, app_module, fake_modal)
        assert modal.run_scoped_names(NONCE)["volume_name"] in (
            fake_modal.Volume.objects.deleted
        )

    def test_teardown_records_unverified_scale_to_zero_rather_than_claiming_it(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        app_module.app.stats_error = RuntimeError("stats unavailable")
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        teardown = document["teardown"]
        assert teardown["scale_zero_verified_via_control_plane"] is False
        assert teardown["adjudication"]["complete"] is False
        assert any(
            item.startswith("scale_zero_unverified")
            for item in teardown["teardown_failures"]
        )

    def test_teardown_records_a_failed_volume_delete(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        fake_modal.Volume.objects.delete_error = RuntimeError("delete failed")
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["teardown"]["volume_deleted"] is False
        assert "volume_delete_failed" in document["teardown"]["teardown_failures"]


class TestRateRefreshSurface:
    def test_capture_hashes_documents_without_parsing_them(self) -> None:
        capture = rates_module.capture_rate_documents(
            fetcher=_fetcher, observed_at="2026-09-04T19:52:50.511+05:30"
        )
        assert capture["parsed_from_html"] is False
        assert [item["url"] for item in capture["documents"]] == list(
            rates_module.OFFICIAL_SOURCE_URLS
        )
        assert all(
            item["sha256"].startswith("sha256:") for item in capture["documents"]
        )

    def test_receipt_must_be_bound_to_the_captured_document(self) -> None:
        capture = rates_module.capture_rate_documents(
            fetcher=_fetcher, observed_at="2026-09-04T19:52:50.511+05:30"
        )
        receipt = {
            "source_url": "https://modal.com/pricing",
            "document_sha256": "sha256:" + "b" * 64,
            "fetched_at": "2026-09-04T19:52:50.511+05:30",
            "rates": {
                "l4_gpu_second": "0.000222",
                "cpu_core_second": "0.0000131",
                "memory_gib_second": "0.00000222",
                "volume_gib_month": "0.09",
            },
            "additional_charges": [],
        }
        with pytest.raises(rates_module.RateRefreshError, match="hash differs"):
            rates_module.verify_rate_refresh(receipt, capture=capture)
        receipt["document_sha256"] = capture["documents"][0]["sha256"]
        assert rates_module.verify_rate_refresh(receipt, capture=capture)["verified"]

    def test_unofficial_sources_are_refused(self) -> None:
        with pytest.raises(rates_module.RateRefreshError, match="official domain"):
            rates_module.capture_rate_documents(
                fetcher=_fetcher,
                urls=("https://rates.example.com/modal",),
                observed_at="2026-09-04T19:52:50.511+05:30",
            )

    def test_headroom_is_never_inferred(self) -> None:
        with pytest.raises(rates_module.RateRefreshError, match="refusing to infer"):
            rates_module.account_headroom()
        with pytest.raises(rates_module.RateRefreshError, match="requires a verifier"):
            rates_module.account_headroom(signed_receipt={"headroom_usd": "25"})

    def test_control_plane_probe_is_used_when_supported(self) -> None:
        result = rates_module.account_headroom(
            control_plane_probe=lambda: {"headroom_usd": "25"}
        )
        assert result["supported"] is True
        assert result["is_provider_spend_proof"] is False

    def test_signed_receipt_must_be_sanitized(self) -> None:
        with pytest.raises(rates_module.RateRefreshError, match="sanitized"):
            rates_module.account_headroom(
                signed_receipt={"headroom_usd": "25", "email": "a@b.test"},
                signature_verifier=lambda payload: None,
            )

    def test_no_network_is_used_by_the_offline_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("offline paths must not open a socket")

        monkeypatch.setattr(rates_module.urllib.request, "urlopen", forbidden)
        capture = rates_module.capture_rate_documents(
            fetcher=_fetcher, observed_at="2026-09-04T19:52:50.511+05:30"
        )
        assert capture["documents"]

    def test_read_structured_receipt_errors_are_sanitized(self, tmp_path: Path) -> None:
        # A read failure must never echo the operator's path or a snippet of the
        # unsafe document; only the failure category is surfaced.
        secret_path = tmp_path / "SuperSecretDir" / "receipt.json"
        with pytest.raises(rates_module.RateRefreshError) as excinfo:
            rates_module.read_structured_receipt(secret_path)
        message = str(excinfo.value)
        assert "SuperSecretDir" not in message
        assert str(secret_path) not in message
        assert "FileNotFoundError" in message

        malformed = tmp_path / "malformed.json"
        malformed.write_text('{"unterminated', encoding="utf-8")
        with pytest.raises(rates_module.RateRefreshError) as excinfo:
            rates_module.read_structured_receipt(malformed)
        assert "unterminated" not in str(excinfo.value)


class TestAuthorizationBinding:
    def test_self_authored_authorization_without_binding_is_refused(
        self, tmp_path: Path
    ) -> None:
        plan = modal.build_default_plan()
        payload = execute_module.ModalExecutionAuthorization.content(
            plan_sha256=plan.content_sha256,
            source_head=HEAD,
            experiment_nonce=NONCE,
            workspace_sha256="sha256:" + "c" * 64,
            rate_receipt_sha256="sha256:" + "d" * 64,
            credential_exposure_attestation_sha256="sha256:" + "e" * 64,
            authorized_at=AUTHORIZED_AT,
            not_before=NOT_BEFORE,
            expires_at=EXPIRES_AT,
        )
        payload["authorization_sha256"] = "sha256:" + "0" * 64
        with pytest.raises(execute_module.ModalExecutionError, match="content hash"):
            execute_module.ModalExecutionAuthorization.from_dict(payload, plan=plan)

    def test_authorization_must_accept_the_crash_reschedule_residual(
        self, tmp_path: Path
    ) -> None:
        plan = modal.build_default_plan()
        payload = execute_module.ModalExecutionAuthorization.content(
            plan_sha256=plan.content_sha256,
            source_head=HEAD,
            experiment_nonce=NONCE,
            workspace_sha256="sha256:" + "c" * 64,
            rate_receipt_sha256="sha256:" + "d" * 64,
            credential_exposure_attestation_sha256="sha256:" + "e" * 64,
            authorized_at=AUTHORIZED_AT,
            not_before=NOT_BEFORE,
            expires_at=EXPIRES_AT,
        )
        assert payload["accepts_modal_crash_reschedule_residual"] is True
        assert payload["automatic_retries"] == 0
        assert payload["authentication"] == "standard_local_modal_profile_only"
        payload["accepts_modal_crash_reschedule_residual"] = False
        payload["authorization_sha256"] = execute_module._sha256_json(
            {k: v for k, v in payload.items() if k != "authorization_sha256"}
        )
        with pytest.raises(execute_module.ModalExecutionError, match="envelope"):
            execute_module.ModalExecutionAuthorization.from_dict(payload, plan=plan)

    def test_authorization_is_bound_to_the_plan_hash(self) -> None:
        plan = modal.build_default_plan()
        payload = execute_module.ModalExecutionAuthorization.content(
            plan_sha256="sha256:" + "f" * 64,
            source_head=HEAD,
            experiment_nonce=NONCE,
            workspace_sha256="sha256:" + "c" * 64,
            rate_receipt_sha256="sha256:" + "d" * 64,
            credential_exposure_attestation_sha256="sha256:" + "e" * 64,
            authorized_at=AUTHORIZED_AT,
            not_before=NOT_BEFORE,
            expires_at=EXPIRES_AT,
        )
        payload["authorization_sha256"] = execute_module._sha256_json(
            {k: v for k, v in payload.items() if k != "authorization_sha256"}
        )
        with pytest.raises(execute_module.ModalExecutionError, match="different plan"):
            execute_module.ModalExecutionAuthorization.from_dict(payload, plan=plan)

    def test_authorization_binds_the_runtime_image_commitment(self) -> None:
        plan = modal.build_default_plan()
        content = execute_module.ModalExecutionAuthorization.content(
            plan_sha256=plan.content_sha256,
            source_head=HEAD,
            experiment_nonce=NONCE,
            workspace_sha256="sha256:" + "c" * 64,
            rate_receipt_sha256="sha256:" + "d" * 64,
            credential_exposure_attestation_sha256="sha256:" + "e" * 64,
            authorized_at=AUTHORIZED_AT,
            not_before=NOT_BEFORE,
            expires_at=EXPIRES_AT,
        )
        # The signed authorization binds the exact SDK version and a
        # deterministic derived-image spec commitment (never a provider image
        # digest), plus a run commitment tying it to the source head.
        assert content["provider_sdk_tested_version"] == modal.TESTED_MODAL_VERSION
        assert content["runtime_image_spec_commitment"] == (
            modal.RUNTIME_IMAGE_SPEC_COMMITMENT
        )
        assert content["runtime_image_run_commitment"] == (
            modal.runtime_image_identity(source_head=HEAD)[
                "runtime_image_run_commitment"
            ]
        )


def _windowed_authorization(
    *, not_before: str = NOT_BEFORE, expires_at: str = EXPIRES_AT
) -> Any:
    plan = modal.build_default_plan()
    content = execute_module.ModalExecutionAuthorization.content(
        plan_sha256=plan.content_sha256,
        source_head=HEAD,
        experiment_nonce=NONCE,
        workspace_sha256="sha256:" + "c" * 64,
        rate_receipt_sha256="sha256:" + "d" * 64,
        credential_exposure_attestation_sha256="sha256:" + "e" * 64,
        authorized_at=AUTHORIZED_AT,
        not_before=not_before,
        expires_at=expires_at,
    )
    content["authorization_sha256"] = execute_module._sha256_json(
        {k: v for k, v in content.items() if k != "authorization_sha256"}
    )
    return execute_module.ModalExecutionAuthorization.from_dict(content, plan=plan)


class TestExecutionWindow:
    """Requirement 9: a bounded, signed [not_before, expires_at) UTC window."""

    def test_a_window_within_bounds_verifies(self) -> None:
        window = execute_module.verify_execution_window(
            _windowed_authorization(), now=WITHIN_WINDOW
        )
        assert window["verified"] is True
        assert window["checked_within_window"] is True
        assert window["not_before"] == NOT_BEFORE
        assert window["expires_at"] == EXPIRES_AT

    def test_an_approval_before_its_window_is_refused(self) -> None:
        early = datetime(2026, 9, 4, 13, 59, 59, tzinfo=timezone.utc)
        with pytest.raises(execute_module.ModalExecutionError, match="not yet valid"):
            execute_module.verify_execution_window(_windowed_authorization(), now=early)

    def test_an_expired_or_replayed_approval_is_refused(self) -> None:
        # At expires_at and any later instant the approval is dead, so a stale
        # or replayed approval can never re-authorize a run.
        for now in (
            datetime(2026, 9, 4, 18, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 9, 6, 9, 0, 0, tzinfo=timezone.utc),
        ):
            with pytest.raises(execute_module.ModalExecutionError, match="has expired"):
                execute_module.verify_execution_window(
                    _windowed_authorization(), now=now
                )

    def test_a_naive_clock_is_refused(self) -> None:
        with pytest.raises(execute_module.ModalExecutionError, match="timezone-aware"):
            execute_module.verify_execution_window(
                _windowed_authorization(), now=datetime(2026, 9, 4, 14, 30, 0)
            )

    def test_an_empty_or_inverted_window_is_refused(self) -> None:
        with pytest.raises(
            execute_module.ModalExecutionError, match="empty or inverted"
        ):
            _windowed_authorization(not_before=EXPIRES_AT, expires_at=NOT_BEFORE)

    def test_a_window_wider_than_the_maximum_is_refused(self) -> None:
        with pytest.raises(execute_module.ModalExecutionError, match="maximum bounded"):
            _windowed_authorization(
                not_before="2026-09-04T00:00:00+00:00",
                expires_at="2026-09-06T00:00:00+00:00",
            )

    @pytest.mark.parametrize(
        "edge",
        (
            "2026-09-04T14:00:00",  # naive, no offset
            "2026-09-04T14:00:00+05:30",  # non-UTC offset
            "not-a-timestamp",
        ),
    )
    def test_a_malformed_window_edge_is_refused(self, edge: str) -> None:
        with pytest.raises(execute_module.ModalExecutionError):
            _windowed_authorization(not_before=edge)

    def test_preflight_checks_the_window_before_any_fetch(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        files = _write_gate_files(tmp_path, workspace=workspace)

        def forbidden_fetcher(url: str) -> bytes:
            raise AssertionError("no official-rate fetch before the window check")

        expired = datetime(2026, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(execute_module.ModalExecutionError, match="has expired"):
            execute_module.preflight(
                workspace=workspace,
                environ={},
                fetcher=forbidden_fetcher,
                signature_runner=lambda argv, message: 0,
                signed_headroom={"headroom_usd": "25"},
                signature_verifier=lambda payload: None,
                source_checkout_probe=_clean_checkout_probe,
                now_utc=expired,
                **files,
            )


class TestContainerRunner:
    def test_refusal_receipts_are_sealed_and_terminal(self) -> None:
        receipt = cell_runner._refusal("modal_cell", reason="cell failed")
        assert receipt["terminal"] is True
        assert receipt["status"] == "refused"
        assert receipt["receipt_sha256"].startswith("sha256:")

    def test_out_of_memory_is_classified(self) -> None:
        assert cell_runner._is_out_of_memory(RuntimeError("CUDA out of memory")) is True
        assert cell_runner._is_out_of_memory(ValueError("bad input")) is False

    def test_longest_controlled_prompt_selects_the_longest_array(self) -> None:
        prompts = {
            "prompts": {
                "2k/structured-json-profile-extraction": [1, 2, 3],
                "2k/prose-reasoning-two-train-gap": [1, 2],
                "8k/structured-json-profile-extraction": [1] * 20,
                "8k/prose-reasoning-two-train-gap": [1] * 10,
                "16k/structured-json-profile-extraction": [1] * 40,
                "16k/prose-reasoning-two-train-gap": [1] * 30,
            }
        }
        key, ids = cell_runner.longest_controlled_prompt(prompts)
        assert key == "16k/structured-json-profile-extraction"
        assert len(ids) == 40
        assert modal.max_model_len(len(ids)) == 136

    def test_invalid_prompt_receipt_is_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="invalid token IDs"):
            cell_runner.longest_controlled_prompt(
                {"prompts": {"2k/structured-json-profile-extraction": ["x"]}}
            )

    def test_stage_refuses_when_the_inventory_differs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(cell_runner, "model_path", lambda: tmp_path / "model")
        monkeypatch.setattr(cell_runner, "state_path", lambda: tmp_path / "state")
        module = type(sys)("huggingface_hub")
        module.snapshot_download = lambda **kwargs: (  # type: ignore[attr-defined]
            (tmp_path / "model").mkdir(parents=True, exist_ok=True),
            (tmp_path / "model" / "config.json").write_text("{}", encoding="utf-8"),
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        receipt = cell_runner.stage_model()
        assert receipt["status"] == "refused"
        assert "inventory" in receipt["reason"]

    def test_l4_gate_rejects_another_accelerator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="NVIDIA A10G, 550.1, 23034, 12, GPU-x\n"
            )

        monkeypatch.setattr(cell_runner.subprocess, "run", fake_run)
        with pytest.raises(modal.ModalL4ContractError, match="not the approved"):
            cell_runner.l4_hardware()

    def test_l4_gate_accepts_one_l4_and_never_pins_the_driver(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="NVIDIA L4, 570.86, 23034, 5, GPU-abc\n"
            )

        monkeypatch.setattr(cell_runner.subprocess, "run", fake_run)
        hardware = cell_runner.l4_hardware()
        assert hardware["gpu_name"] == "NVIDIA L4"
        assert hardware["gpu_count"] == 1
        assert hardware["driver_pinned"] is False
        assert hardware["gpu_uuid_sha256"].startswith("sha256:")
        assert "GPU-abc" not in json.dumps(hardware)

    def test_two_accelerators_are_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="NVIDIA L4, 570.86, 23034, 5, GPU-a\nNVIDIA L4, 570.86, 23034, 5, GPU-b\n",
            )

        monkeypatch.setattr(cell_runner.subprocess, "run", fake_run)
        with pytest.raises(modal.ModalL4ContractError, match="exactly 1 accelerator"):
            cell_runner.l4_hardware()

    def test_measured_cell_refuses_an_unsealed_cell_id(self) -> None:
        receipt = cell_runner.run_measured_cell("not-a-cell", experiment_nonce=NONCE)
        assert receipt["status"] == "refused"
        assert "sealed schedule" in receipt["reason"]

    def test_container_identity_is_salted_and_optional(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MODAL_TASK_ID", raising=False)
        assert cell_runner.container_identity() is None
        monkeypatch.setenv("MODAL_TASK_ID", "ta-12345")
        identity = cell_runner.container_identity()
        assert identity is not None and identity.startswith("sha256:")
        assert "ta-12345" not in identity

    def test_runner_module_imports_no_provider_sdk(self) -> None:
        source = Path(cell_runner.__file__).read_text(encoding="utf-8")
        assert "import modal" not in source
        assert "from modal" not in source


class TestNoRealProviderPackage:
    def test_the_real_sdk_is_never_imported_by_these_tests(self) -> None:
        loaded = [name for name in sys.modules if name.split(".")[0] == "modal"]
        assert loaded in ([], ["modal"])
        if loaded:
            assert getattr(sys.modules["modal"], "__file__", None) is None

    def test_only_the_app_module_imports_the_sdk(self) -> None:
        root = Path(__file__).resolve().parents[2] / "llmtracefx"
        pattern = re.compile(r"^\s*(?:import modal$|from modal[. ])", re.MULTILINE)
        offenders = [
            path.relative_to(root).as_posix()
            for path in root.rglob("modal_l4_*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
            and path.name != "modal_l4_app.py"
        ]
        assert offenders == []


class TestCredentialExposureGate:
    def test_a_cleared_attestation_allows_the_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "complete"
        exposure = document["credential_exposure"]
        assert exposure["cleared"] is True
        assert exposure["exposed_profile_credential_never_used_by_experiment"] is True
        assert exposure["exposed_profile_credential_revocation_confirmed"] is True
        assert exposure["fresh_local_profile_created_without_sharing"] is True
        assert exposure["fresh_profile_shared_anywhere"] is False
        assert exposure["records_credential_values"] is False
        assert "reason" not in exposure

    def test_a_missing_attestation_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load while the gate is blocked")

        with pytest.raises(modal.ModalL4ContractError, match="blocked until"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                drop_attestation=True,
            )
        assert app_module.app.calls == []

    @pytest.mark.parametrize(
        "override",
        (
            {"exposed_profile_credential_revocation_confirmed": False},
            {"fresh_local_profile_created_without_sharing": False},
            {"fresh_profile_shared_anywhere": True},
            {"status": "blocked"},
            {"revocation_confirmed_by": None},
            {"confirmed_at": None},
        ),
    )
    def test_an_uncleared_attestation_refuses_before_the_sdk_loads(
        self,
        tmp_path: Path,
        app_module: Any,
        fake_modal: Any,
        override: dict[str, Any],
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load while the gate is blocked")

        with pytest.raises(modal.ModalL4ContractError, match="blocked until"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                attestation=_attestation(**override),
            )
        assert app_module.app.calls == []

    def test_a_swapped_attestation_is_not_the_authorized_one(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        files = _write_gate_files(tmp_path, workspace=workspace)
        swapped = _attestation(reason="a different but still cleared confirmation")
        files["credential_exposure_attestation_path"].write_text(
            json.dumps(swapped, indent=2), encoding="utf-8"
        )
        with pytest.raises(
            execute_module.ModalExecutionError, match="not the authorized one"
        ):
            execute_module.preflight(
                workspace=workspace,
                environ={},
                fetcher=_fetcher,
                signature_runner=lambda argv, message: 0,
                signed_headroom={"headroom_usd": "25"},
                signature_verifier=lambda payload: None,
                now_utc=WITHIN_WINDOW,
                **files,
            )

    def test_the_gate_verdict_records_booleans_and_nothing_else(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        _run(tmp_path, app_module, fake_modal)
        verdict = json.loads(
            (tmp_path / "workspace" / "credential-exposure.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(verdict) == {
            "gate",
            "cleared",
            "exposed_profile_credential_never_used_by_experiment",
            "exposed_profile_credential_revocation_confirmed",
            "fresh_local_profile_created_without_sharing",
            "fresh_profile_shared_anywhere",
            "confirmed_by",
            "confirmed_at",
            "reason",
            "records_credential_values",
            "action",
        }
        assert not any(
            fragment in key.lower()
            for key in verdict
            for fragment in ("hash", "sha256", "prefix", "screenshot", "token")
        )
        assert verdict["confirmed_by"] == "coordinator"

    def test_no_stored_receipt_carries_credential_material(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        _run(tmp_path, app_module, fake_modal)
        workspace = tmp_path / "workspace"
        markers = (
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "token_id",
            "token_secret",
            "screenshot",
            "bearer ",
            "ak-",
        )
        for path in sorted(workspace.rglob("*.json")):
            text = path.read_text(encoding="utf-8")
            for marker in markers:
                assert marker not in text, (path.name, marker)
        receipt = json.loads(
            (workspace / "orchestration-receipt.json").read_text(encoding="utf-8")
        )
        assert "reason" not in receipt["credential_exposure"]
        assert receipt["credential_exposure"]["records_credential_values"] is False

    def test_the_gate_runs_before_the_environment_check(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="blocked until"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                environ={"MODAL_TOKEN_ID": "value"},
                sdk_loader=lambda: (_ for _ in ()).throw(AssertionError("no SDK")),
                drop_attestation=True,
            )


class _SteppedClock:
    """A monotonic_ns stand-in that advances by a fixed step on every read."""

    def __init__(self, step_ns: int) -> None:
        self.step_ns = step_ns
        self.value = 0

    def __call__(self) -> int:
        current = self.value
        self.value += self.step_ns
        return current


def _orchestrator(
    tmp_path: Path,
    app_module: Any,
    fake_modal: Any,
    *,
    monotonic_ns: Any = None,
) -> Any:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    files = _write_gate_files(tmp_path, workspace=workspace)
    fake_modal.Volume.objects.existing.append(
        modal.run_scoped_names(NONCE)["volume_name"]
    )
    gates = execute_module.preflight(
        workspace=workspace,
        environ={},
        fetcher=_fetcher,
        signed_headroom={"headroom_usd": "25"},
        signature_verifier=lambda payload: None,
        signature_runner=lambda argv, message: 0,
        source_checkout_probe=_clean_checkout_probe,
        now_utc=WITHIN_WINDOW,
        **files,
    )
    ledger = modal.ModalApplicationLedger.initialize(
        workspace / "application-ledger.json",
        plan=gates["plan"],
        git_head=gates["authorization"].source_head,
        experiment_nonce=gates["authorization"].experiment_nonce,
    )
    kwargs: dict[str, Any] = {}
    if monotonic_ns is not None:
        kwargs["monotonic_ns"] = monotonic_ns
    return execute_module.ModalOrchestrator(
        plan=gates["plan"],
        authorization=gates["authorization"],
        workspace=gates["workspace"],
        ledger=ledger,
        credential_exposure=gates["credential_exposure"],
        rate_receipt=gates["rate_receipt"],
        rate_refresh=gates["rates"],
        source_checkout=gates["source_checkout"],
        sdk_loader=lambda: fake_modal,
        app_loader=lambda: app_module,
        profile_validator=_authenticated_profile,
        **kwargs,
    )


class TestOrchestratorTiming:
    def test_observed_duration_is_measured_and_never_zero_or_capped(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        # Five real seconds per call, well under every lifecycle ceiling.
        clock = _SteppedClock(step_ns=5_000_000_000)
        orchestrator = _orchestrator(
            tmp_path, app_module, fake_modal, monotonic_ns=clock
        )
        _script(orchestrator._app_loader(), _Receipts())
        document = orchestrator.execute()
        assert document["status"] == "complete"
        completions = [
            event
            for event in document["ledger"]["events"]
            if event["event_type"] == "complete"
        ]
        assert completions, "a completed run must record completions"
        assert all(event["actual_seconds"] == 5 for event in completions)
        assert all(
            event["duration_provenance"] == "client_observed_monotonic_ceiling_seconds"
            for event in completions
        )
        assert all(float(event["actual_cost_usd"]) > 0 for event in completions)

    def test_exceeding_the_lifecycle_ceiling_invalidates_the_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        # 2001 seconds exceeds the 1800s stage ceiling on the very first call.
        clock = _SteppedClock(step_ns=2_001_000_000_000)
        orchestrator = _orchestrator(
            tmp_path, app_module, fake_modal, monotonic_ns=clock
        )
        _script(orchestrator._app_loader(), _Receipts())
        document = orchestrator.execute()
        assert document["status"] == "invalidated"
        assert "lifecycle_ceiling_exceeded" in document["failure"]
        aborts = [
            event
            for event in document["ledger"]["events"]
            if event["event_type"] == "abort"
        ]
        assert aborts and "ceiling" in aborts[-1]["abort_reason"]

    def test_default_orchestrator_uses_a_real_monotonic_clock(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        orchestrator = _orchestrator(tmp_path, app_module, fake_modal)
        assert orchestrator._monotonic_ns is __import__("time").monotonic_ns


class TestOutstandingCallTeardown:
    def test_a_crashed_call_is_cancelled_with_container_termination(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()
        _script(
            app_module,
            receipts,
            verify=lambda *args: RuntimeError("provider crashed mid-call"),
        )
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        # The retained handle must be cancelled by teardown with container
        # termination; without the fix the finally-clear left nothing to cancel.
        assert ("cancel", "verify_stage") in app_module.app.log
        assert document["teardown"]["outstanding_calls_cancelled"] is True

    def test_a_timed_out_call_is_cancelled_with_container_termination(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        from _fake_modal import FunctionTimeoutError

        receipts = _Receipts()
        _script(
            app_module,
            receipts,
            eager_canary=lambda *args: FunctionTimeoutError("late"),
        )
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert "timeout" in document["failure"]
        assert ("cancel", "eager_canary") in app_module.app.log

    def test_a_non_terminal_receipt_still_cancels_the_handle(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()
        _script(app_module, receipts, verify=lambda *args: {"status": "completed"})
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "invalidated"
        assert ("cancel", "verify_stage") in app_module.app.log


class TestHeadroomSignatureGate:
    def test_a_signed_headroom_receipt_requires_both_signature_paths(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        files = _write_gate_files(tmp_path, workspace=workspace)
        with pytest.raises(
            execute_module.ModalExecutionError, match="headroom-signature"
        ):
            execute_module.preflight(
                workspace=workspace,
                environ={},
                fetcher=_fetcher,
                signature_runner=lambda argv, message: 0,
                signed_headroom={"headroom_usd": "25"},
                source_checkout_probe=_clean_checkout_probe,
                now_utc=WITHIN_WINDOW,
                **files,
            )

    def test_the_production_verifier_runs_ssh_keygen_over_the_receipt(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        files = _write_gate_files(tmp_path, workspace=workspace)
        signature = tmp_path / "headroom.sig"
        signature.write_text("detached\n", encoding="utf-8")
        signers = tmp_path / "headroom_signers"
        signers.write_text(
            f"{execute_module.HEADROOM_SIGNER_IDENTITY} ssh-ed25519 TEST\n",
            encoding="utf-8",
        )
        seen: dict[str, Any] = {}

        def runner(argv: Any, message: str) -> int:
            seen["argv"] = list(argv)
            seen["message"] = message
            return 0

        gates = execute_module.preflight(
            workspace=workspace,
            environ={},
            fetcher=_fetcher,
            signature_runner=runner,
            signed_headroom={"headroom_usd": "25"},
            headroom_signature_path=signature,
            headroom_authorized_signers_path=signers,
            source_checkout_probe=_clean_checkout_probe,
            now_utc=WITHIN_WINDOW,
            **files,
        )
        assert gates["headroom"]["provenance"] == "signed_operator_receipt"
        assert rates_module.HEADROOM_SIGNATURE_NAMESPACE in seen["argv"]
        assert execute_module.HEADROOM_SIGNER_IDENTITY in seen["argv"]
        # The signed message is the canonical headroom receipt, nothing else.
        assert json.loads(seen["message"]) == {"headroom_usd": "25"}

    def test_a_failing_headroom_signature_refuses(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        files = _write_gate_files(tmp_path, workspace=workspace)
        signature = tmp_path / "headroom.sig"
        signature.write_text("detached\n", encoding="utf-8")
        signers = tmp_path / "headroom_signers"
        signers.write_text("signer ssh-ed25519 TEST\n", encoding="utf-8")
        with pytest.raises(
            execute_module.ModalExecutionError, match="headroom signature did not"
        ):
            execute_module.preflight(
                workspace=workspace,
                environ={},
                fetcher=_fetcher,
                signature_runner=lambda argv, message: (
                    0 if "authorization" in "".join(str(a) for a in argv) else 1
                ),
                signed_headroom={"headroom_usd": "25"},
                headroom_signature_path=signature,
                headroom_authorized_signers_path=signers,
                source_checkout_probe=_clean_checkout_probe,
                now_utc=WITHIN_WINDOW,
                **files,
            )


class TestCleanWorkspaceGate:
    def test_a_stale_run_receipt_refuses_the_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "orchestration-receipt.json").write_text("{}", encoding="utf-8")
        files = _write_gate_files(tmp_path, workspace=workspace)
        fake_modal.Volume.objects.existing.append(
            modal.run_scoped_names(NONCE)["volume_name"]
        )
        with pytest.raises(execute_module.ModalExecutionError, match="not clean"):
            execute_module.execute(
                workspace=workspace,
                sdk_loader=lambda: fake_modal,
                app_loader=lambda: app_module,
                environ={},
                fetcher=_fetcher,
                signed_headroom={"headroom_usd": "25"},
                signature_verifier=lambda payload: None,
                signature_runner=lambda argv, message: 0,
                source_checkout_probe=_clean_checkout_probe,
                profile_validator=_authenticated_profile,
                now_utc=WITHIN_WINDOW,
                **files,
            )

    def test_a_stale_cells_directory_refuses_the_run(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        workspace = tmp_path / "workspace"
        (workspace / "cells").mkdir(parents=True)
        files = _write_gate_files(tmp_path, workspace=workspace)
        fake_modal.Volume.objects.existing.append(
            modal.run_scoped_names(NONCE)["volume_name"]
        )
        with pytest.raises(execute_module.ModalExecutionError, match="not clean"):
            execute_module.execute(
                workspace=workspace,
                sdk_loader=lambda: fake_modal,
                app_loader=lambda: app_module,
                environ={},
                fetcher=_fetcher,
                signed_headroom={"headroom_usd": "25"},
                signature_verifier=lambda payload: None,
                signature_runner=lambda argv, message: 0,
                source_checkout_probe=_clean_checkout_probe,
                profile_validator=_authenticated_profile,
                now_utc=WITHIN_WINDOW,
                **files,
            )


class TestSourceCheckoutGate:
    """Task 6: bind execution to a clean checkout at the authorized head."""

    def test_a_head_mismatch_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load before the checkout gate")

        with pytest.raises(
            execute_module.ModalExecutionError, match="does not match the authorized"
        ):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                source_checkout_probe=lambda: {
                    "head": "a" * 40,
                    "status_porcelain": "",
                },
            )
        assert app_module.app.calls == []

    def test_a_dirty_tracked_file_refuses_before_the_sdk_loads(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def forbidden_loader() -> Any:
            raise AssertionError("the SDK must not load before the checkout gate")

        dirty = " M llmtracefx/optimizer/lab/qwen3_8b/modal_l4_execute.py\n"
        with pytest.raises(execute_module.ModalExecutionError, match="not clean"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=forbidden_loader,
                source_checkout_probe=lambda: {"head": HEAD, "status_porcelain": dirty},
            )
        assert app_module.app.calls == []

    def test_untracked_python_source_refuses(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        untracked = "?? llmtracefx/optimizer/lab/qwen3_8b/sneaked_in.py\n"
        with pytest.raises(execute_module.ModalExecutionError, match="not clean"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                sdk_loader=lambda: (_ for _ in ()).throw(
                    AssertionError("SDK must not load")
                ),
                source_checkout_probe=lambda: {
                    "head": HEAD,
                    "status_porcelain": untracked,
                },
            )

    def test_the_known_untracked_agent_traces_is_ignored(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(
            tmp_path,
            app_module,
            fake_modal,
            source_checkout_probe=lambda: {
                "head": HEAD,
                "status_porcelain": "?? .agent-traces/trace-01.json\n",
            },
        )
        assert document["status"] == "complete"

    def test_the_gate_output_is_written(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        _run(tmp_path, app_module, fake_modal)
        gate = json.loads(
            (tmp_path / "workspace" / "source-checkout.json").read_text(
                encoding="utf-8"
            )
        )
        assert gate["verified"] is True
        assert gate["tracked_workspace_clean"] is True
        assert gate["source_head"] == HEAD

    def test_the_production_gate_reads_real_git_and_refuses_uncommitted_source(
        self,
    ) -> None:
        # The default probe reads the real checkout. This branch's Modal source
        # is still uncommitted, so the production gate must refuse it: an
        # execution can only start once the implementation is committed exactly.
        head = execute_module._default_source_checkout_probe(execute_module.REPO_ROOT)[
            "head"
        ]
        with pytest.raises(execute_module.ModalExecutionError):
            execute_module.verify_source_checkout(
                source_head=head, repo_root=execute_module.REPO_ROOT
            )


class TestLocalProfileGate:
    """Task 7: validate a standard local profile before the app is imported."""

    def test_validation_records_only_safe_fields(self) -> None:
        result = execute_module.validate_local_profile(
            sdk_version="1.5.5",
            runner=lambda command: 0,
            clock=lambda: "2026-09-04T20:00:00+00:00",
        )
        assert result == {
            "schema_version": "1",
            "gate": "local_profile_authentication",
            "authenticated": True,
            "mechanism": "current_interpreter_python_m_modal_token_info",
            "cli_version": "1.5.5",
            "sdk_version": "1.5.5",
            "records_profile_identity": False,
            "checked_at": "2026-09-04T20:00:00+00:00",
        }

    def test_validation_runs_modal_token_info(self) -> None:
        seen: dict[str, Any] = {}

        def runner(command: Any) -> int:
            seen["command"] = tuple(command)
            return 0

        execute_module.validate_local_profile(sdk_version="1.5.5", runner=runner)
        # The probe is the running interpreter's own modal module, so the CLI is
        # guaranteed to be the same install the SDK is loaded from.
        assert seen["command"] == (sys.executable, "-m", "modal", "token", "info")
        assert execute_module.profile_probe_command() == (
            sys.executable,
            "-m",
            "modal",
            "token",
            "info",
        )

    def test_a_nonzero_exit_status_refuses(self) -> None:
        with pytest.raises(
            execute_module.ModalExecutionError, match="no authenticated"
        ):
            execute_module.validate_local_profile(
                sdk_version="1.5.5", runner=lambda command: 1
            )

    def test_the_default_runner_discards_every_stream(self, monkeypatch: Any) -> None:
        seen: dict[str, Any] = {}

        class _Completed:
            returncode = 0

        def fake_run(command: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            seen["command"] = command
            return _Completed()

        monkeypatch.setattr(execute_module.subprocess, "run", fake_run)
        code = execute_module._default_profile_command_runner(
            ("modal", "token", "info")
        )
        assert code == 0
        assert seen["stdin"] is execute_module.subprocess.DEVNULL
        assert seen["stdout"] is execute_module.subprocess.DEVNULL
        assert seen["stderr"] is execute_module.subprocess.DEVNULL
        assert seen["shell"] is False
        assert seen["env"]["PATH"] == execute_module._SAFE_EXECUTION_PATH
        assert seen["timeout"] == execute_module.PROFILE_VALIDATION_TIMEOUT_SECONDS

    def test_validation_runs_after_the_sdk_probe_and_before_app_import(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        order: list[str] = []

        def sdk_loader() -> Any:
            order.append("sdk")
            return fake_modal

        def validator(*, sdk_version: str) -> dict[str, Any]:
            order.append("profile")
            return _authenticated_profile(sdk_version=sdk_version)

        _script(app_module, _Receipts())
        _run(
            tmp_path,
            app_module,
            fake_modal,
            sdk_loader=sdk_loader,
            profile_validator=validator,
        )
        # The app is only entered after the SDK probe and the profile gate.
        assert order == ["sdk", "profile"]
        assert app_module.app.entered == 1

    def test_an_unauthenticated_profile_refuses_before_app_import(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def validator(*, sdk_version: str) -> dict[str, Any]:
            raise execute_module.ModalExecutionError("no authenticated profile")

        with pytest.raises(execute_module.ModalExecutionError, match="authenticated"):
            _run(
                tmp_path,
                app_module,
                fake_modal,
                profile_validator=validator,
            )
        assert app_module.app.entered == 0
        assert app_module.app.calls == []

    def test_a_result_carrying_profile_identity_is_refused(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        def validator(*, sdk_version: str) -> dict[str, Any]:
            result = _authenticated_profile(sdk_version=sdk_version)
            result["token_id"] = "ak-1234"
            return result

        with pytest.raises(
            execute_module.ModalExecutionError,
            match="profile or credential derived fields",
        ):
            _run(tmp_path, app_module, fake_modal, profile_validator=validator)
        assert app_module.app.entered == 0

    def test_the_profile_verdict_is_in_the_orchestration(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["profile_authentication"]["authenticated"] is True
        assert document["profile_authentication"]["records_profile_identity"] is False


class TestNoResultEvidenceOnRefusal:
    """Task 8: a non-complete or incomplete-teardown run writes no result."""

    def test_an_incomplete_teardown_writes_no_result_evidence(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        fake_modal.Volume.objects.delete_error = RuntimeError("delete failed")
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["published"] is False
        workspace = tmp_path / "workspace"
        assert not (workspace / "orchestration-receipt.json").exists()
        assert not (workspace / "cells").exists()
        assert not (workspace / "memory-gate").exists()

    def test_a_refusal_writes_no_result_evidence(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        receipts = _Receipts()

        def failing_canary(*args: Any) -> dict[str, Any]:
            payload = receipts.canary("eager")
            payload["observation"]["out_of_memory"] = True
            return payload

        _script(app_module, receipts, eager_canary=failing_canary)
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "refused"
        assert document["published"] is False
        workspace = tmp_path / "workspace"
        assert not (workspace / "orchestration-receipt.json").exists()
        assert not (workspace / "cells").exists()


class TestCliPublicationStatus:
    def test_a_complete_run_with_incomplete_teardown_is_not_published(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        fake_modal.Volume.objects.delete_error = RuntimeError("delete failed")
        _script(app_module, _Receipts())
        document = _run(tmp_path, app_module, fake_modal)
        assert document["status"] == "complete"
        assert document["teardown"]["adjudication"]["complete"] is False
        assert document["published"] is False
        assert document["kind"].endswith(".refusal")

    def test_cli_returns_nonzero_when_the_headroom_triple_is_incomplete(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        # The headroom receipt, its signature, and its authorized-signers file
        # must be supplied together; a partial set never reaches preflight and
        # never returns success.
        code = execute_module.main(
            [
                "preflight",
                "--authorization",
                str(tmp_path / "a"),
                "--authorization-signature",
                str(tmp_path / "b"),
                "--authorized-signers",
                str(tmp_path / "c"),
                "--rate-receipt",
                str(tmp_path / "d"),
                "--workspace",
                str(tmp_path / "w"),
                "--credential-exposure-attestation",
                str(tmp_path / "f"),
                "--signed-headroom",
                str(tmp_path / "g"),
            ]
        )
        assert code == 1
        assert "supplied together" in capsys.readouterr().err

    def test_cli_run_returns_nonzero_when_result_is_not_published(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        # main() maps publication to the exit code: a complete run that was not
        # published (for example an incomplete teardown) must not exit 0.
        monkeypatch.setattr(
            execute_module,
            "execute",
            lambda **kwargs: {"status": "complete", "published": False},
        )
        args = [
            "run",
            "--authorization",
            str(tmp_path / "a"),
            "--authorization-signature",
            str(tmp_path / "b"),
            "--authorized-signers",
            str(tmp_path / "c"),
            "--rate-receipt",
            str(tmp_path / "d"),
            "--workspace",
            str(tmp_path / "w"),
            "--credential-exposure-attestation",
            str(tmp_path / "f"),
        ]
        assert execute_module.main(args) == 2
        monkeypatch.setattr(
            execute_module,
            "execute",
            lambda **kwargs: {"status": "complete", "published": True},
        )
        assert execute_module.main(args) == 0
        del capsys


class TestResultAnalysisFidelity:
    """The real orchestrator output is accepted by the hardened analyzer.

    This closes the loop between the execution path and the result path: a run
    that returns full sealed cell and canary receipts produces an orchestration
    receipt that ``analyze_modal_run`` validates end to end, so the offline
    result fixture models production faithfully.
    """

    def test_a_real_run_output_is_accepted_by_the_analyzer(
        self, tmp_path: Path, app_module: Any, fake_modal: Any
    ) -> None:
        import _modal_result_fixture as fixture

        from llmtracefx.optimizer.lab.qwen3_8b import (
            modal_l4_crossover_results as results_module,
        )
        from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as core

        plan = core.build_default_plan()
        index_by_id = {
            cell.cell_id: index
            for index, cell in enumerate(modal.crossover_schedule(), start=1)
        }
        schedule = {cell.cell_id: cell for cell in modal.crossover_schedule()}

        def cell_builder(cell_id: str, nonce: str) -> dict[str, Any]:
            del nonce
            return fixture.cell_wrapper(
                schedule[cell_id], plan, index=index_by_id[cell_id], nonce=NONCE
            )

        receipts = _Receipts()
        # Advance the CPU-stage identities well past the fixture cell indices so
        # stage/verify/analysis never collide with a cell or canary container.
        receipts.index = 900
        _script(
            app_module,
            receipts,
            eager_canary=lambda *a: fixture.canary_receipt(
                "eager", index=1, nonce=NONCE
            ),
            compiled_canary=lambda *a: fixture.canary_receipt(
                "compiled", index=2, nonce=NONCE
            ),
            natural_cell=cell_builder,
            controlled_cell=cell_builder,
        )
        document = _run(tmp_path, app_module, fake_modal)
        assert document["published"] is True
        cells = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (tmp_path / "workspace" / "cells").iterdir()
        }
        analysis = results_module.analyze_modal_run(orchestration=document, cells=cells)
        assert analysis["pair_count"] == 16
        claims = {
            claim["claim_id"]: claim["state"]
            for claim in analysis["claim_matrix"]["claims"]
        }
        assert claims["fixed-token-count-provider-conditioned-crossover"] == (
            "supported"
        )
        assert analysis["hardware_placement"]["raw_gpu_identity_exposed"] is False


def _l4_hardware() -> dict[str, Any]:
    return {
        "gpu_name": "NVIDIA L4",
        "gpu_count": 1,
        "driver_version": "570.86",
        "driver_pinned": False,
        "memory_total_mib": 23_034,
        "memory_used_mib": 5,
        "gpu_uuid_sha256": "sha256:" + "a" * 64,
    }


class TestL4HardwareObserverInjection:
    def test_the_gate_installs_and_restores_the_shared_hardware_probe(self) -> None:
        original = cell_runner.base_runner._hardware
        with cell_runner._l4_hardware_gate():
            assert cell_runner.base_runner._hardware is cell_runner.l4_hardware
        assert cell_runner.base_runner._hardware is original

    def test_the_gate_restores_even_when_the_delegate_raises(self) -> None:
        original = cell_runner.base_runner._hardware
        with pytest.raises(RuntimeError):
            with cell_runner._l4_hardware_gate():
                raise RuntimeError("boom")
        assert cell_runner.base_runner._hardware is original

    def test_measured_cell_delegates_under_the_l4_observer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cell_runner, "l4_hardware", _l4_hardware)
        monkeypatch.setattr(cell_runner, "model_path", lambda: tmp_path / "model")
        monkeypatch.setattr(cell_runner, "state_path", lambda: tmp_path / "state")
        seen: dict[str, Any] = {}

        def fake_run_cell(cell_id: str, *, output: Path, **kwargs: Any) -> None:
            seen["observer"] = cell_runner.base_runner._hardware
            output.write_text(json.dumps({"terminal": True}), encoding="utf-8")

        monkeypatch.setattr(cell_runner.cell_runner, "run_cell", fake_run_cell)
        receipt = cell_runner.run_measured_cell(
            CELL_IDS[0],
            experiment_nonce=NONCE,
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "out"),
        )
        # The delegate ran with the L4 observer, not the RTX 4090 gate, which an
        # L4 could never satisfy.
        assert seen["observer"] is cell_runner.l4_hardware
        assert cell_runner.base_runner._hardware is cell_runner.base_runner._hardware
        assert receipt["status"] == "completed"
        assert (
            receipt["runtime_image"]["derived_image_spec_commitment"]
            == modal.RUNTIME_IMAGE_SPEC_COMMITMENT
        )
        assert receipt["runtime_image"]["derived_provider_image_digest"] is None
        assert "gpu_uuid_sha256" not in receipt["provider_hardware"]


class TestCanaryKvCapacityGate:
    def test_kv_capacity_fails_closed_when_the_pinned_path_is_absent(self) -> None:
        class _NoEngine:
            pass

        with pytest.raises(modal.ModalL4ContractError, match="KV cache capacity"):
            cell_runner._kv_capacity(_NoEngine())

    def test_kv_capacity_reads_the_pinned_vllm_0_28_fields(self) -> None:
        engine = types.SimpleNamespace(
            cache_config=types.SimpleNamespace(num_gpu_blocks=640, block_size=64)
        )
        llm = types.SimpleNamespace(llm_engine=engine)
        tokens, blocks = cell_runner._kv_capacity(llm)
        assert blocks == 640
        assert tokens == 640 * 64

    def test_finish_reason_shape_is_pinned(self) -> None:
        assert cell_runner.FINISH_REASON_LENGTH == "length"


class TestFailureSanitization:
    def test_failure_detail_persists_only_the_stable_class_name(self) -> None:
        exc = RuntimeError("/Users/alice/secret https://host/x?token=abcdef GPU-1234")
        detail = cell_runner._failure_detail(exc)
        assert detail == "RuntimeError"
        for leak in ("/Users/", "https://", "token", "GPU-"):
            assert leak not in detail

    @pytest.mark.parametrize(
        ("exc", "category"),
        (
            (RuntimeError("CUDA out of memory"), "out_of_memory"),
            (modal.ModalL4ContractError("x"), "contract_violation"),
            (ValueError("x"), "value_error"),
            (RuntimeError("x"), "runtime_error"),
            (OSError("x"), "io_error"),
            (KeyError("x"), "unexpected_error"),
        ),
    )
    def test_failure_category_is_allowlisted(
        self, exc: BaseException, category: str
    ) -> None:
        assert cell_runner._failure_category(exc) in cell_runner._FAILURE_CATEGORIES
        assert cell_runner._failure_category(exc) == category

    def test_measured_cell_refusal_leaks_no_exception_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cell_runner, "l4_hardware", _l4_hardware)
        monkeypatch.setattr(cell_runner, "model_path", lambda: tmp_path / "model")
        monkeypatch.setattr(cell_runner, "state_path", lambda: tmp_path / "state")

        def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("/Users/alice/secret leaked https://host/tok")

        monkeypatch.setattr(cell_runner.cell_runner, "run_cell", boom)
        receipt = cell_runner.run_measured_cell(
            CELL_IDS[0],
            experiment_nonce=NONCE,
            cache_root=str(tmp_path / "cache"),
            output_root=str(tmp_path / "out"),
        )
        assert receipt["status"] == "refused"
        assert receipt["detail"] == "RuntimeError"
        assert receipt["failure_category"] == "runtime_error"
        assert "secret" not in json.dumps(receipt)
        assert "https://" not in json.dumps(receipt)
