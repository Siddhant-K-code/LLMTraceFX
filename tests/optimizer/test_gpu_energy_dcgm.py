"""Tests for the dcgm-exporter source of ``gpu-energy``.

No GPU or exporter is needed. :class:`FakeExporter` behaves like
dcgm-exporter seen through scrapes: DCGM samples every ``collect_s`` on its
own schedule and each scrape returns the latest sample with no timestamp.
"""

from __future__ import annotations

import functools
import http.server
import json
import math
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from _fake_nvml import FakeClock, FakeLauncher

from llmtracefx.optimizer import cli as optimizer_cli
from llmtracefx.optimizer.gpu_energy import cli as gpu_cli
from llmtracefx.optimizer.gpu_energy.capability import uuid_sha256
from llmtracefx.optimizer.gpu_energy.dcgm_evidence import build_dcgm_evidence
from llmtracefx.optimizer.gpu_energy.dcgm_exporter import (
    AMBIGUOUS_SERIES,
    ENERGY_METRICS,
    ScrapeError,
    http_scraper,
    parse_gpus,
    scrape_gpus,
)
from llmtracefx.optimizer.gpu_energy.dcgm_window import (
    DcgmExporterUnavailable,
    DcgmWindowRecord,
    record_dcgm_window,
)

# Verbatim from dcgm-exporter 4.9.0 (4e663a526c4624ad0e18ad816f09b7be2c5348a7),
# tests/internal/exportercontract/contracts_test.go.
UPSTREAM_4_9_LINES = """\
# HELP DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL Total energy.
# TYPE DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL counter
DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{gpu="0",UUID="GPU-0",device="nvidia0",\
modelName="NVIDIA B200",Hostname="node-a",pod="cuda-workload",\
namespace="dcgm-exporter",container="cuda-workload"} 196764
# HELP DCGM_FI_DEV_PCIE_REPLAY_TOTAL PCIe replay counter.
# TYPE DCGM_FI_DEV_PCIE_REPLAY_TOTAL counter
DCGM_FI_DEV_PCIE_REPLAY_TOTAL{gpu="0",UUID="GPU-0",device="nvidia0",\
modelName="NVIDIA B200",Hostname="node-a",pod="cuda-workload",\
namespace="dcgm-exporter",container="cuda-workload"} 0
"""

T0 = 1000.0


def _labels(index: int, extra: str = "") -> str:
    return (
        f'gpu="{index}",UUID="GPU-{index}",pci_bus_id="00000000:0{index}:00.0",'
        f'device="nvidia{index}",modelName="NVIDIA H100 80GB HBM3",'
        f'Hostname="node-a"{extra}'
    )


def _family(name: str, kind: str, lines: list[str]) -> str:
    return f"# HELP {name} {name}.\n# TYPE {name} {kind}\n" + "".join(lines)


class FakeExporter:
    def __init__(
        self,
        clock: FakeClock,
        *,
        gpus: tuple[int, ...] = (0,),
        power_w: float = 200.0,
        collect_s: float = 1.0,
        phase_s: float = 0.3,
        latency_s: float = 0.01,
        style: str = "4.8",
        failing_calls: frozenset[int] = frozenset(),
        frozen_after: float | None = None,
        reset_at: float | None = None,
    ) -> None:
        self.clock = clock
        self.gpus = gpus
        self.power_w = power_w
        self.collect_s = collect_s
        self.phase_s = phase_s
        self.latency_s = latency_s
        self.style = style
        self.failing_calls = failing_calls
        self.frozen_after = frozen_after
        self.reset_at = reset_at
        self.calls = 0
        #: Rendered energy value to the time DCGM sampled it.
        self.sample_times: dict[float, float] = {}

    def sample_time(self) -> float:
        now = self.clock.now
        if self.frozen_after is not None:
            now = min(now, self.frozen_after)
        steps = math.floor((now - T0 - self.phase_s) / self.collect_s)
        return T0 + self.phase_s + steps * self.collect_s

    def energy_mj(self, t: float) -> float:
        if self.reset_at is not None and t >= self.reset_at:
            return math.floor(self.power_w * (t - self.reset_at) * 1000)
        return math.floor(5e9 + self.power_w * (t - T0) * 1000)

    def render(self) -> str:
        t = self.sample_time()
        self.sample_times[float(math.floor(self.energy_mj(t)))] = t
        if self.style == "4.9":
            energy = [
                f"DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{{{_labels(i)}}} "
                f"{math.floor(self.energy_mj(t) / 1000)}\n"
                for i in self.gpus
            ]
            power = [
                f"DCGM_FI_DEV_BOARD_POWER_WATTS{{{_labels(i)}}} {self.power_w}\n"
                for i in self.gpus
            ]
            return _family(
                "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL", "counter", energy
            ) + _family("DCGM_FI_DEV_BOARD_POWER_WATTS", "gauge", power)
        energy = [
            f"DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{{_labels(i)}}} "
            f"{self.energy_mj(t):.0f}\n"
            for i in self.gpus
        ]
        power = [
            f"DCGM_FI_DEV_POWER_USAGE{{{_labels(i)}}} {self.power_w}\n"
            for i in self.gpus
        ]
        clocks = [f"DCGM_FI_DEV_SM_CLOCK{{{_labels(i)}}} 1980\n" for i in self.gpus]
        return (
            _family("DCGM_FI_DEV_SM_CLOCK", "gauge", clocks)
            + _family("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "counter", energy)
            + _family("DCGM_FI_DEV_POWER_USAGE", "gauge", power)
        )

    def __call__(self) -> str:
        self.calls += 1
        self.clock.now += self.latency_s / 2
        try:
            if self.calls in self.failing_calls:
                raise ScrapeError("unreachable", "connection refused")
            return self.render()
        finally:
            self.clock.now += self.latency_s / 2


def _record(
    clock: FakeClock,
    exporter: FakeExporter,
    launcher: FakeLauncher,
    **kwargs: Any,
) -> DcgmWindowRecord:
    return record_dcgm_window(
        exporter,
        list(exporter.gpus),
        ["python", "infer.py"],
        clock=clock,
        sleep=clock.sleep,
        launcher=launcher,
        **kwargs,
    )


def _evidence(record: DcgmWindowRecord, **kwargs: Any) -> dict[str, Any]:
    return build_dcgm_evidence(
        record,
        endpoint="http://node-a:9400/metrics",
        argv=["python", "infer.py"],
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        min_run_share=kwargs.pop("min_run_share", 0.9),
        **kwargs,
    )


# Parsing -----------------------------------------------------------------


def test_parse_gpus_reads_upstream_contract_output() -> None:
    gpus = parse_gpus(UPSTREAM_4_9_LINES)
    assert list(gpus) == [0]
    gpu = gpus[0]
    assert gpu.name == "NVIDIA B200"
    assert gpu.uuid_sha256 == uuid_sha256("GPU-0")
    metric = gpu.energy_metric()
    assert metric is not None and metric.unit == "J"
    assert gpu.energy(metric) == (196764.0, None)
    assert not gpu.mig_seen


def test_parse_gpus_prefers_millijoule_counter() -> None:
    labels = _labels(0)
    text = _family(
        "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL",
        "counter",
        [f"DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL{{{labels}}} 1234\n"],
    ) + _family(
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        "counter",
        [f"DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION{{{labels}}} 1234567\n"],
    )
    metric = parse_gpus(text)[0].energy_metric()
    assert metric is not None
    assert (metric.name, metric.unit) == ("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "mJ")


def test_parse_gpus_accepts_equal_mig_series_and_refuses_disagreeing_ones() -> None:
    name = "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
    first, second = _labels(0, ',GPU_I_ID="1"'), _labels(0, ',GPU_I_ID="2"')
    same = _family(
        name, "counter", [f"{name}{{{first}}} 500\n", f"{name}{{{second}}} 500\n"]
    )
    gpu = parse_gpus(same)[0]
    assert gpu.mig_seen
    assert gpu.values[name] == 500

    different = same.replace("} 500\n", "} 900\n", 1)
    gpu = parse_gpus(different)[0]
    assert gpu.energy_metric() is None
    assert gpu.energy(ENERGY_METRICS[0]) == (None, AMBIGUOUS_SERIES)


# Window and evidence -----------------------------------------------------


def test_window_brackets_the_command_and_measures_energy() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, gpus=(0, 1), collect_s=1.0)
    launcher = FakeLauncher(clock, runtime_s=60.0)
    record = _record(clock, exporter, launcher, scrape_interval_ms=200)
    evidence = _evidence(record, stated_collect_interval_ms=1000)

    assert launcher.calls == [["python", "infer.py"]]
    assert evidence["source"] == "dcgm-exporter"
    assert evidence["schema_version"] == "1"
    for device in evidence["devices"]:
        bracket = device["bracket"]
        assert device["energy_refused_reason"] is None
        energy = device["energy"]
        assert energy["provenance"] == "measured_native"
        assert energy["method"] == "dcgm_exporter_counter_delta_over_bracket"
        # The counter delta is P x (s1 - s0) with s0 and s1 inside the bounds.
        joules = energy["value"]
        assert joules >= 200.0 * bracket["min_span_ms"] / 1000 - 0.002
        assert joules <= 200.0 * bracket["max_span_ms"] / 1000 + 0.002
        assert bracket["min_span_ms"] >= bracket["command_ms"]
        assert bracket["min_command_share"] >= 0.9
        assert device["average_power"] is None
        assert device["sampled_power_integral"] is None
        assert device["sampled_power_refused_reason"] == "sample_times_unknown"
        assert device["power_samples"]["values"][0][1] == 200.0
        assert device["energy_counter"]["unit"] == "mJ"
        assert device["uuid_sha256"] == uuid_sha256(f"GPU-{device['index']}")
    sampling = evidence["power_sampling"]
    assert sampling["interval_ms"] == 200
    assert sampling["collect_interval_stated"]["stated_by"] == "operator"
    observed = sampling["collect_interval_observed"]
    assert abs(observed["value"] - 1000) <= sampling["interval_ms"]
    assert evidence["totals"]["energy"]["value"] == pytest.approx(
        sum(d["energy"]["value"] for d in evidence["devices"])
    )
    assert evidence["attribution"]["status"] == "unverifiable"
    assert evidence["highlights"] == {}
    assert "node-a" not in json.dumps(evidence["devices"])


@pytest.mark.parametrize("phase_s", [i * 0.07 for i in range(15)])
def test_bounds_contain_the_true_sample_times(phase_s: float) -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, collect_s=1.0, phase_s=phase_s, latency_s=0.09)
    launcher = FakeLauncher(clock, runtime_s=7.33)
    record = _record(clock, exporter, launcher, scrape_interval_ms=300)
    exited_ms = (launcher.children[0].ends_at - T0) * 1000

    device = record.devices[0]
    assert device.start is not None and device.end is not None
    s0 = (exporter.sample_times[device.start.value] - T0) * 1000
    s1 = (exporter.sample_times[device.end.value] - T0) * 1000
    assert device.start.after_ms < s0 <= device.start.by_ms
    assert device.end.after_ms < s1 <= device.end.by_ms
    assert device.start.by_ms <= record.command_start_ms
    assert device.end.after_ms >= record.command_end_ms >= exited_ms
    assert s0 <= record.command_start_ms and s1 > exited_ms


def test_window_with_assume_isolated_is_operator_asserted() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, style="4.9", collect_s=1.0)
    record = _record(clock, exporter, FakeLauncher(clock, runtime_s=60.0))
    evidence = _evidence(record, assume_isolated=True)

    assert evidence["attribution"]["status"] == "operator_asserted"
    assert evidence["attribution"]["attributable"] is True
    assert evidence["devices"][0]["energy_counter"]["unit"] == "J"
    assert evidence["highlights"]["gpu_energy"]["unit"] == "J"


def test_default_collect_interval_makes_a_short_bracket_too_wide() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, collect_s=30.0)
    record = _record(clock, exporter, FakeLauncher(clock, runtime_s=10.0))
    device = _evidence(record)["devices"][0]

    assert device["energy"] is None
    assert device["energy_refused_reason"] == "bracket_too_wide"
    assert device["bracket"]["max_span_ms"] > 30_000
    assert device["energy_counter"]["end"] > device["energy_counter"]["start"]


def test_long_run_passes_with_default_collect_interval() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, collect_s=30.0)
    record = _record(clock, exporter, FakeLauncher(clock, runtime_s=600.0))
    device = _evidence(record)["devices"][0]

    assert device["energy_refused_reason"] is None
    assert device["bracket"]["min_command_share"] >= 0.9


def test_no_counter_change_before_the_command_runs_nothing() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, frozen_after=T0)
    launcher = FakeLauncher(clock)
    with pytest.raises(DcgmExporterUnavailable, match="was not run"):
        _record(clock, exporter, launcher, edge_timeout_ms=5000)
    assert launcher.calls == []


def test_first_scrape_failure_runs_nothing() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, failing_calls=frozenset({1}))
    launcher = FakeLauncher(clock)
    with pytest.raises(DcgmExporterUnavailable, match="unreachable"):
        _record(clock, exporter, launcher)
    assert launcher.calls == []


def test_no_counter_change_after_the_command_refuses_energy() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, frozen_after=T0 + 20.0)
    record = _record(
        clock, exporter, FakeLauncher(clock, runtime_s=30.0), edge_timeout_ms=5000
    )
    device = _evidence(record)["devices"][0]

    assert device["energy"] is None
    assert device["bracket"] is None
    assert device["energy_refused_reason"] == "no_counter_update_after_run"


def test_counter_reset_during_the_command_refuses_energy() -> None:
    clock = FakeClock(T0)
    exporter = FakeExporter(clock, reset_at=T0 + 30.0)
    record = _record(clock, exporter, FakeLauncher(clock, runtime_s=60.0))
    device = _evidence(record)["devices"][0]

    assert device["energy"] is None
    assert device["energy_refused_reason"] == "counter_reset"


def test_failed_scrapes_are_counted_and_keep_the_bounds_honest() -> None:
    clock = FakeClock(T0)
    failing = frozenset(range(5, 400, 3))
    exporter = FakeExporter(clock, failing_calls=failing)
    record = _record(clock, exporter, FakeLauncher(clock, runtime_s=60.0))
    evidence = _evidence(record)
    device = evidence["devices"][0]

    assert evidence["power_sampling"]["scrape_errors"]["unreachable"] > 0
    joules = device["energy"]["value"]
    bracket = device["bracket"]
    assert 200.0 * bracket["min_span_ms"] / 1000 - 0.002 <= joules
    assert joules <= 200.0 * bracket["max_span_ms"] / 1000 + 0.002


# HTTP scraper --------------------------------------------------------------


@pytest.fixture
def exporter_server() -> Iterator[str]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/metrics":
                self.send_error(404)
                return
            body = UPSTREAM_4_9_LINES.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_http_scraper_reads_a_local_exporter(exporter_server: str) -> None:
    gpus = scrape_gpus(http_scraper(f"{exporter_server}/metrics", timeout=5))
    assert gpus[0].values["DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL"] == 196764


def test_http_scraper_reports_http_errors(exporter_server: str) -> None:
    with pytest.raises(ScrapeError) as excinfo:
        http_scraper(f"{exporter_server}/nope", timeout=5)()
    assert excinfo.value.reason == "http_404"


# CLI ---------------------------------------------------------------------


@pytest.fixture
def fake_dcgm_cli(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    clock = FakeClock(T0)
    state: dict[str, Any] = {
        "clock": clock,
        "exporter": FakeExporter(clock, collect_s=1.0),
        "launcher": FakeLauncher(clock, runtime_s=60.0),
        "urls": [],
    }

    def factory(url: str) -> Any:
        state["urls"].append(url)
        return state["exporter"]

    def run(*args: Any, **kwargs: Any) -> Any:
        return functools.partial(
            record_dcgm_window,
            clock=clock,
            sleep=clock.sleep,
            launcher=state["launcher"],
        )(*args, **kwargs)

    monkeypatch.setattr(gpu_cli, "scraper_factory", factory)
    monkeypatch.setattr(gpu_cli, "run_dcgm_window", run)
    return state


def _run_cli(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        optimizer_cli.main(argv)
    return int(excinfo.value.code or 0)


URL = ["--source", "dcgm-exporter", "--dcgm-exporter-url", "http://node-a:9400/metrics"]


def test_cli_capability_reports_exporter_gpus(
    fake_dcgm_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(["gpu-energy", "capability", *URL]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"] == "dcgm-exporter"
    assert report["collector"]["endpoint"] == "http://node-a:9400/metrics"
    device = report["devices"][0]
    assert device["energy_counter"]["metric"] == "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION"
    assert device["process_listing"]["supported"] is False


def test_cli_capability_exits_3_when_the_exporter_is_unreachable(
    fake_dcgm_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    fake_dcgm_cli["exporter"].failing_calls = frozenset({1})
    assert _run_cli(["gpu-energy", "capability", *URL]) == 3
    assert json.loads(capsys.readouterr().out)["state"] == "unreachable"


def test_cli_record_writes_unverifiable_evidence_and_exits_2(
    fake_dcgm_cli: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "gpu.json"
    argv = ["gpu-energy", "record", "--output", str(output), *URL]
    assert _run_cli([*argv, "--", "python", "x.py"]) == 2
    evidence = json.loads(output.read_text())
    assert evidence["source"] == "dcgm-exporter"
    assert evidence["devices"][0]["energy"]["provenance"] == "measured_native"
    assert evidence["power_sampling"]["interval_ms"] == 500
    assert "The energy delta covers" in capsys.readouterr().out


def test_cli_record_with_assume_isolated_exits_0(
    fake_dcgm_cli: dict[str, Any], tmp_path: Path
) -> None:
    output = tmp_path / "gpu.json"
    argv = ["gpu-energy", "record", "--output", str(output), *URL]
    assert _run_cli([*argv, "--assume-isolated", "--", "x"]) == 0
    assert "gpu_energy" in json.loads(output.read_text())["highlights"]


def test_cli_record_exits_3_without_a_counter_edge_and_runs_nothing(
    fake_dcgm_cli: dict[str, Any], tmp_path: Path
) -> None:
    fake_dcgm_cli["exporter"].frozen_after = T0
    output = tmp_path / "gpu.json"
    argv = ["gpu-energy", "record", "--output", str(output), *URL]
    assert _run_cli([*argv, "--edge-timeout-ms", "3000", "--", "x"]) == 3
    assert fake_dcgm_cli["launcher"].calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "dcgm-exporter"],
        [*URL, "--gpu-pid", "12"],
        [*URL, "--power-sample-interval-ms", "50"],
        ["--scrape-interval-ms", "500"],
        ["--dcgm-exporter-url", "http://node-a:9400/metrics"],
        [*URL, "--min-run-share", "1.5"],
        [*URL, "--scrape-interval-ms", "nan"],
        [*URL, "--device", "7"],
        ["--source", "dcgm-exporter", "--dcgm-exporter-url", "http://t@h/m?x=1"],
    ],
)
def test_cli_record_refuses_bad_dcgm_input_before_running(
    fake_dcgm_cli: dict[str, Any], tmp_path: Path, extra: list[str]
) -> None:
    output = tmp_path / "gpu.json"
    argv = ["gpu-energy", "record", "--output", str(output), *extra, "--", "x"]
    assert _run_cli(argv) == 1
    assert fake_dcgm_cli["launcher"].calls == []
    assert not output.exists()
