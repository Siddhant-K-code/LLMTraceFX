"""Tests for ``gpu-energy``: NVML capability, the measured window, evidence
labelling, attribution and the CLI. NVML is always a fake here."""

from __future__ import annotations

import functools
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _fake_nvml import (
    FakeClock,
    FakeDevice,
    FakeLauncher,
    FakeNvml,
    FakeProcessTable,
)

from llmtracefx.optimizer import cli as optimizer_cli
from llmtracefx.optimizer.gpu_energy import cli as gpu_cli
from llmtracefx.optimizer.gpu_energy.capability import (
    check_capability,
    probe_capability,
    uuid_sha256,
)
from llmtracefx.optimizer.gpu_energy.evidence import build_evidence
from llmtracefx.optimizer.gpu_energy.nvml import (
    NvmlCallError,
    NvmlUnavailable,
    PynvmlBackend,
    load_backend,
)
from llmtracefx.optimizer.gpu_energy.window import (
    ALLOWED,
    ANCESTRY_UNKNOWN,
    FOREIGN,
    NOT_VISIBLE,
    ProcProcessTable,
    record_window,
)

CHILD_PID = 4242


def _child_listed(_t: float) -> tuple[int, ...]:
    return (CHILD_PID,)


def _measure(
    devices: list[FakeDevice],
    *,
    runtime_s: float = 1.0,
    exit_code: int = 0,
    interval_ms: float = 100,
    max_samples: int = 100_000,
    classes: dict[int, str] | None = None,
    assume_isolated: bool = False,
    gpu_pids: tuple[int, ...] = (),
    argv: tuple[str, ...] = ("python", "infer.py"),
) -> dict[str, Any]:
    clock = FakeClock()
    backend = FakeNvml(devices, clock)
    capability = probe_capability(backend)
    launcher = FakeLauncher(clock, runtime_s=runtime_s, exit_code=exit_code)
    record = record_window(
        backend,
        list(range(len(devices))),
        argv,
        interval_ms=interval_ms,
        max_samples=max_samples,
        gpu_pids=gpu_pids,
        clock=clock,
        sleep=clock.sleep,
        launcher=launcher,
        process_table=FakeProcessTable(classes),
    )
    return build_evidence(
        record,
        capability.devices,
        versions=capability.versions,
        argv=argv,
        started_at="2026-01-01T00:00:00.000000Z",
        ended_at="2026-01-01T00:00:01.000000Z",
        assume_isolated=assume_isolated,
        gpu_pids=gpu_pids,
    )


# Capability


def test_capability_supported_reports_devices_without_raw_uuid() -> None:
    device = FakeDevice()
    capability = check_capability(lambda: FakeNvml([device], FakeClock()))
    report = capability.as_dict()
    assert report["state"] == "supported"
    assert report["supported"] is True
    [entry] = report["devices"]
    assert entry["uuid_sha256"] == uuid_sha256(device.uuid)
    assert entry["enforced_power_limit_w"] == 72.0
    assert entry["energy_counter"] == {"supported": True, "reason": None}
    assert device.uuid not in json.dumps(report)


@pytest.mark.parametrize(
    ("backend", "state"),
    [
        (FakeNvml([], FakeClock()), "no_devices"),
        (
            FakeNvml([FakeDevice()], FakeClock(), count_error="call_failed"),
            "nvml_error",
        ),
        (
            FakeNvml(
                [FakeDevice(energy_error="not_supported", power_error="not_supported")],
                FakeClock(),
            ),
            "no_power_readings",
        ),
        (
            FakeNvml(
                [],
                FakeClock(),
                init_error=NvmlUnavailable("driver_unavailable", "no driver"),
            ),
            "driver_unavailable",
        ),
    ],
)
def test_capability_unsupported_states(backend: FakeNvml, state: str) -> None:
    capability = check_capability(lambda: backend)
    assert capability.state == state
    assert not capability.is_supported


def test_load_backend_reports_extra_missing_without_pynvml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    with pytest.raises(NvmlUnavailable) as excinfo:
        load_backend()
    assert excinfo.value.state == "extra_missing"


# The real backend against a stand-in for the pynvml module


class _NVMLError(Exception):
    def __init__(self, value: int) -> None:
        super().__init__(f"code {value}")
        self.value = value


def _fake_pynvml(**overrides: Any) -> types.SimpleNamespace:
    def raise_(code: int) -> Any:
        def call(*_args: Any) -> Any:
            raise _NVMLError(code)

        return call

    module = types.SimpleNamespace(
        NVMLError=_NVMLError,
        NVML_ERROR_NOT_SUPPORTED=3,
        NVML_ERROR_NO_PERMISSION=4,
        NVML_ERROR_DRIVER_NOT_LOADED=9,
        NVML_ERROR_LIBRARY_NOT_FOUND=12,
        NVML_ERROR_GPU_IS_LOST=15,
        NVML_DEVICE_MIG_ENABLE=1,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlSystemGetNVMLVersion=lambda: b"12.560.35.03",
        nvmlSystemGetDriverVersion=lambda: "560.35.03",
        nvmlSystemGetCudaDriverVersion=lambda: 12060,
        nvmlDeviceGetCount=lambda: 1,
        nvmlDeviceGetHandleByIndex=lambda index: f"handle-{index}",
        nvmlDeviceGetName=lambda _h: b"NVIDIA L4",
        nvmlDeviceGetUUID=lambda _h: "GPU-abc",
        nvmlDeviceGetTotalEnergyConsumption=lambda _h: 123_456,
        nvmlDeviceGetPowerUsage=raise_(3),
        nvmlDeviceGetEnforcedPowerLimit=raise_(4),
        nvmlDeviceGetMigMode=lambda _h: [1, 1],
        nvmlDeviceGetComputeRunningProcesses=lambda _h: [
            types.SimpleNamespace(pid=7),
            types.SimpleNamespace(pid=9),
        ],
        nvmlDeviceGetGraphicsRunningProcesses=lambda _h: [types.SimpleNamespace(pid=7)],
        raise_=raise_,
    )
    for name, value in overrides.items():
        setattr(module, name, value)
    return module


def test_pynvml_backend_translates_values_and_errors() -> None:
    module = _fake_pynvml(nvmlDeviceGetTotalEnergyConsumption=_fake_pynvml().raise_(15))
    backend = PynvmlBackend(module)
    backend.init()
    assert backend.versions()["cuda_driver_version"] == "12.6"
    assert backend.versions()["nvml_version"] == "12.560.35.03"
    assert backend.device_name(0) == "NVIDIA L4"
    assert backend.mig_mode_enabled(0) is True
    assert backend.running_pids(0) == (7, 9)
    for call, reason in (
        (backend.power_mw, "not_supported"),
        (backend.enforced_power_limit_mw, "no_permission"),
        (backend.total_energy_mj, "gpu_lost"),
    ):
        with pytest.raises(NvmlCallError) as excinfo:
            call(0)
        assert excinfo.value.reason == reason


@pytest.mark.parametrize(
    ("code", "state"),
    [(12, "library_unavailable"), (9, "driver_unavailable"), (999, "nvml_error")],
)
def test_pynvml_backend_init_failures(code: int, state: str) -> None:
    module = _fake_pynvml()
    module.nvmlInit = module.raise_(code)
    with pytest.raises(NvmlUnavailable) as excinfo:
        PynvmlBackend(module).init()
    assert excinfo.value.state == state


# Energy and power labelling


def test_counter_delta_is_measured_and_average_power_is_derived() -> None:
    evidence = _measure([FakeDevice(power_w=50.0, pids=_child_listed)])
    [device] = evidence["devices"]
    counter = device["energy_counter"]
    seconds = (counter["end_offset_ms"] - counter["start_offset_ms"]) / 1000
    assert device["energy"]["provenance"] == "measured_native"
    assert device["energy"]["method"] == "total_energy_counter_delta"
    assert device["energy"]["unit"] == "J"
    assert device["energy"]["value"] == pytest.approx(
        (counter["end"] - counter["start"]) / 1000
    )
    assert device["energy"]["value"] == pytest.approx(50.0 * seconds, abs=0.002)
    assert device["average_power"]["provenance"] == "derived"
    assert device["average_power"]["value"] == pytest.approx(50.0, rel=1e-3)
    assert evidence["window"]["provenance"] == "measured_wall_clock"


def test_sampled_integral_is_an_estimate_over_the_sample_span() -> None:
    evidence = _measure([FakeDevice(power_w=40.0, pids=_child_listed)])
    [device] = evidence["devices"]
    samples = device["power_samples"]
    assert samples["provenance"] == "measured_native"
    assert samples["count"] >= 10
    span_s = (samples["values"][-1][0] - samples["values"][0][0]) / 1000
    integral = device["sampled_power_integral"]
    assert integral["provenance"] == "estimated"
    assert integral["method"] == "trapezoid_over_sampled_power"
    assert integral["value"] == pytest.approx(40.0 * span_s)
    assert device["sampled_power_mean"]["value"] == pytest.approx(40.0)
    assert device["sampled_power_max"]["provenance"] == "derived"


@pytest.mark.parametrize(
    ("device", "reason"),
    [
        (FakeDevice(energy_script=[9, 5_000, 1_000]), "counter_reset"),
        (FakeDevice(energy_script=[9, 5_000, 5_000]), "stale_counter"),
        (FakeDevice(energy_error="not_supported"), "counter_not_supported"),
    ],
)
def test_counter_refusals_keep_only_the_estimate(
    device: FakeDevice, reason: str
) -> None:
    device.pids = _child_listed
    evidence = _measure([device])
    [entry] = evidence["devices"]
    assert entry["energy"] is None
    assert entry["average_power"] is None
    assert entry["energy_refused_reason"] == reason
    assert entry["sampled_power_integral"]["provenance"] == "estimated"
    assert "gpu_energy" not in evidence["highlights"]
    assert evidence["highlights"]["gpu_sampled_power_integral"]["provenance"] == (
        "estimated"
    )


def test_sample_cap_refuses_sampled_values_but_keeps_samples() -> None:
    evidence = _measure([FakeDevice(pids=_child_listed)], max_samples=3)
    [entry] = evidence["devices"]
    assert entry["power_samples"]["capped"] is True
    assert entry["power_samples"]["count"] == 3
    assert entry["sampled_power_integral"] is None
    assert entry["sampled_power_mean"] is None
    assert entry["sampled_power_refused_reason"] == "sample_cap_reached"
    assert entry["energy"] is not None


def test_failed_power_read_refuses_sampled_values() -> None:
    device = FakeDevice(pids=_child_listed, power_error_calls=frozenset({2}))
    evidence = _measure([device])
    [entry] = evidence["devices"]
    assert entry["power_samples"]["errors"] == {"call_failed": 1}
    assert entry["sampled_power_integral"] is None
    assert entry["sampled_power_refused_reason"] == "sample_errors"


def test_totals_sum_devices_only_when_every_device_has_a_value() -> None:
    evidence = _measure(
        [
            FakeDevice(power_w=50.0, pids=_child_listed),
            FakeDevice(power_w=30.0, uuid="GPU-2"),
        ]
    )
    totals = evidence["totals"]
    energies = [device["energy"]["value"] for device in evidence["devices"]]
    assert totals["energy"]["value"] == pytest.approx(sum(energies))
    assert totals["energy"]["provenance"] == "derived"
    assert totals["sampled_power_integral"]["provenance"] == "estimated"
    assert totals["sampled_power_max"]["value"] == pytest.approx(50.0)

    partial = _measure(
        [
            FakeDevice(pids=_child_listed),
            FakeDevice(uuid="GPU-2", energy_error="no_permission"),
        ]
    )
    assert partial["totals"]["energy"] is None
    assert "gpu_energy" not in partial["highlights"]


# Attribution and highlights


def test_window_with_only_the_command_is_attributable() -> None:
    evidence = _measure([FakeDevice(pids=_child_listed)])
    attribution = evidence["attribution"]
    assert attribution["status"] == "no_foreign_process_observed"
    assert attribution["attributable"] is True
    assert attribution["process_checks"] >= 10
    assert set(evidence["highlights"]) == {
        "gpu_energy",
        "gpu_average_power",
        "gpu_sampled_power_integral",
        "gpu_sampled_power_max",
    }
    assert evidence["highlights_withheld_reason"] is None


def test_foreign_process_contaminates_even_when_isolation_is_assumed() -> None:
    evidence = _measure(
        [FakeDevice(pids=lambda _t: (CHILD_PID, 777))],
        classes={777: FOREIGN},
        assume_isolated=True,
    )
    attribution = evidence["attribution"]
    assert attribution["status"] == "contaminated"
    assert "777" in attribution["reason"]
    assert evidence["highlights"] == {}
    assert evidence["highlights_withheld_reason"] == attribution["reason"]


@pytest.mark.parametrize(
    ("device", "classes", "fragment"),
    [
        (FakeDevice(), None, "never listed"),
        (FakeDevice(pids_error="no_permission"), None, "listing failed"),
        (
            FakeDevice(pids=lambda _t: (CHILD_PID, 31337)),
            {31337: NOT_VISIBLE},
            "not visible",
        ),
        (
            FakeDevice(pids=lambda _t: (CHILD_PID, 55)),
            {55: ANCESTRY_UNKNOWN},
            "ancestry",
        ),
        (FakeDevice(pids=_child_listed, mig=True), None, "MIG"),
    ],
)
def test_unverifiable_windows_withhold_highlights(
    device: FakeDevice, classes: dict[int, str] | None, fragment: str
) -> None:
    evidence = _measure([device], classes=classes)
    attribution = evidence["attribution"]
    assert attribution["status"] == "unverifiable"
    assert fragment in attribution["reason"]
    assert evidence["highlights"] == {}
    assert evidence["devices"][0]["energy"] is not None

    asserted = _measure([device], classes=classes, assume_isolated=True)
    assert asserted["attribution"]["status"] == "operator_asserted"
    assert asserted["attribution"]["attributable"] is True
    assert "gpu_energy" in asserted["highlights"]


def test_gpu_pid_attributes_a_server_the_command_drives() -> None:
    evidence = _measure([FakeDevice(pids=lambda _t: (900,))], gpu_pids=(900,))
    assert evidence["attribution"]["status"] == "no_foreign_process_observed"
    assert evidence["attribution"]["allowed_roots"] == [900, CHILD_PID]


def test_failed_command_withholds_highlights() -> None:
    evidence = _measure([FakeDevice(pids=_child_listed)], exit_code=1)
    assert evidence["command"]["exit_code"] == 1
    assert evidence["highlights"] == {}
    assert evidence["highlights_withheld_reason"] == "the command exited with code 1"
    assert evidence["devices"][0]["energy"] is not None


def test_evidence_redacts_argv_and_never_holds_a_raw_uuid() -> None:
    device = FakeDevice(pids=_child_listed)
    evidence = _measure(
        [device], argv=("python", "infer.py", "--api-key", "sk-live-secret")
    )
    text = json.dumps(evidence, allow_nan=False)
    assert "sk-live-secret" not in text
    assert device.uuid not in text
    assert evidence["devices"][0]["uuid_sha256"] == uuid_sha256(device.uuid)
    assert evidence["kind"] == "llmtracefx.gpu_energy.evidence"
    assert evidence["scope"] == "device_wide_window"


# The window loop


def test_interrupt_stops_the_child_and_propagates() -> None:
    clock = FakeClock()
    backend = FakeNvml([FakeDevice()], clock)
    launcher = FakeLauncher(clock, runtime_s=60.0)

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        record_window(
            backend,
            [0],
            ["sleep", "60"],
            clock=clock,
            sleep=interrupt,
            launcher=launcher,
            process_table=FakeProcessTable(),
        )
    assert launcher.children[0].terminated is True


def _write_stat(root: Path, pid: int, ppid: int, comm: str = "python") -> None:
    (root / str(pid)).mkdir()
    (root / str(pid) / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0\n")


def test_proc_process_table_walks_parent_pids(tmp_path: Path) -> None:
    _write_stat(tmp_path, 100, 1)
    _write_stat(tmp_path, 200, 100, comm="weird) name")
    _write_stat(tmp_path, 300, 200)
    _write_stat(tmp_path, 400, 1)
    _write_stat(tmp_path, 500, 600)
    table = ProcProcessTable(tmp_path)
    roots = frozenset({100})
    assert table.classify(100, roots) == ALLOWED
    assert table.classify(300, roots) == ALLOWED
    assert table.classify(400, roots) == FOREIGN
    assert table.classify(999, roots) == NOT_VISIBLE
    assert table.classify(500, roots) == ANCESTRY_UNKNOWN


# CLI


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    clock = FakeClock()
    state: dict[str, Any] = {
        "clock": clock,
        "devices": [FakeDevice(pids=_child_listed)],
        "launcher": FakeLauncher(clock),
        "classes": {},
        "init_error": None,
    }

    def factory() -> FakeNvml:
        state["backend"] = FakeNvml(
            state["devices"], clock, init_error=state["init_error"]
        )
        return state["backend"]

    def run_window(*args: Any, **kwargs: Any) -> Any:
        return functools.partial(
            record_window,
            clock=clock,
            sleep=clock.sleep,
            launcher=state["launcher"],
            process_table=FakeProcessTable(state["classes"]),
        )(*args, **kwargs)

    monkeypatch.setattr(gpu_cli, "backend_factory", factory)
    monkeypatch.setattr(gpu_cli, "run_window", run_window)
    return state


def _run_cli(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        optimizer_cli.main(argv)
    return int(excinfo.value.code or 0)


def test_cli_capability_writes_report(fake_cli: dict[str, Any], tmp_path: Path) -> None:
    output = tmp_path / "capability.json"
    assert _run_cli(["gpu-energy", "capability", "--output", str(output)]) == 0
    assert json.loads(output.read_text())["state"] == "supported"


def test_cli_capability_exits_3_when_extra_is_missing(
    fake_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli["init_error"] = NvmlUnavailable("extra_missing", "install nvml")
    assert _run_cli(["gpu-energy", "capability"]) == 3
    assert json.loads(capsys.readouterr().out)["state"] == "extra_missing"


def test_cli_record_attributable_window_exits_0(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "gpu.json"
    code = _run_cli(
        ["gpu-energy", "record", "--output", str(output), "--", "python", "x.py"]
    )
    assert code == 0
    evidence = json.loads(output.read_text())
    assert evidence["command"]["argv"] == ["python", "x.py"]
    assert fake_cli["launcher"].calls == [["python", "x.py"]]
    assert fake_cli["backend"].shutdowns == 1


def test_cli_record_unattributable_window_exits_2(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    fake_cli["devices"] = [FakeDevice()]
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 2
    assert json.loads(output.read_text())["attribution"]["status"] == "unverifiable"


def test_cli_record_failed_command_exits_1_with_evidence(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    fake_cli["launcher"].exit_code = 7
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 1
    assert json.loads(output.read_text())["command"]["exit_code"] == 7


def test_cli_record_exits_3_without_nvml_and_runs_nothing(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    fake_cli["init_error"] = NvmlUnavailable("library_unavailable", "no driver")
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 3
    assert fake_cli["launcher"].calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--device", "5", "--", "x"],
        ["--max-samples", "1", "--", "x"],
        ["--power-sample-interval-ms", "0", "--", "x"],
    ],
)
def test_cli_record_refuses_bad_input_before_running(
    fake_cli: dict[str, Any], tmp_path: Path, extra: list[str]
) -> None:
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), *extra]) == 1
    assert fake_cli["launcher"].calls == []
    assert not output.exists()


def test_cli_record_refuses_existing_output(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "gpu.json"
    output.write_text("keep")
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 1
    assert output.read_text() == "keep"
    assert fake_cli["launcher"].calls == []


def test_cli_record_launch_failure_writes_nothing(
    fake_cli: dict[str, Any], tmp_path: Path
) -> None:
    fake_cli["launcher"].error = FileNotFoundError("no such program")
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 1
    assert not output.exists()
    assert fake_cli["backend"].shutdowns == 1


def test_cli_record_interrupt_exits_130_and_writes_nothing(
    fake_cli: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(gpu_cli, "run_window", interrupted)
    output = tmp_path / "gpu.json"
    assert _run_cli(["gpu-energy", "record", "--output", str(output), "--", "x"]) == 130
    assert not output.exists()
