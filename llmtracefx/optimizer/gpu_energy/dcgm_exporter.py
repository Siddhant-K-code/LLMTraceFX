"""Read GPU energy and power from an NVIDIA dcgm-exporter ``/metrics`` endpoint.

dcgm-exporter serves DCGM's latest cached field values in Prometheus text
format. Each request reads the cache again, but the output carries no
timestamps and DCGM refreshes the cache only every collect interval
(``--collect-interval``, default 30000 ms). A value read here is therefore
only known to have been sampled some time before the scrape that returned
it. :mod:`.dcgm_window` turns that into a bounded window.

Metric names follow dcgm-exporter's default counters. Up to 4.8.4 the
energy counter is ``DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`` (mJ) and power is
``DCGM_FI_DEV_POWER_USAGE`` (W). From 4.9.0 the defaults are
``DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL`` (J, truncated to whole joules by
DCGM) and ``DCGM_FI_DEV_BOARD_POWER_WATTS`` (W). The mJ counter is preferred
when both are present because it has finer resolution.

Only the standard library is used, through the server metrics scraper.
"""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..server_metrics.prometheus import ExpositionParseError, parse_exposition
from ..server_metrics.snapshot import (
    DEFAULT_SCRAPE_TIMEOUT_SECONDS,
    ServerMetricsError,
    urllib_fetcher,
    validate_metrics_url,
)
from .capability import CAPABILITY_KIND, uuid_sha256

SOURCE = "dcgm-exporter"

#: Labels dcgm-exporter adds to series that belong to a MIG GPU instance.
MIG_LABELS = ("GPU_I_ID", "GPU_I_PROFILE")

AMBIGUOUS_SERIES = "ambiguous_series"
METRIC_MISSING = "metric_missing"
GPU_MISSING = "gpu_missing"
PROCESS_LISTING_UNAVAILABLE = "not_available_from_dcgm_exporter"


@dataclass(frozen=True)
class EnergyMetric:
    name: str
    unit: str
    joules_per_unit: float
    #: Smallest step the counter can show, in its own unit.
    resolution: float


ENERGY_METRICS = (
    EnergyMetric("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "mJ", 0.001, 1.0),
    EnergyMetric("DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL", "J", 1.0, 1.0),
)
POWER_METRICS = ("DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_BOARD_POWER_WATTS")
POWER_LIMIT_METRICS = (
    "DCGM_FI_DEV_ENFORCED_POWER_LIMIT",
    "DCGM_FI_DEV_BOARD_POWER_LIMIT_ENFORCED_WATTS",
)
_WANTED = (
    tuple(metric.name for metric in ENERGY_METRICS)
    + POWER_METRICS
    + POWER_LIMIT_METRICS
)


class ScrapeError(RuntimeError):
    """A scrape that returned no usable exposition. ``reason`` is a short code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


Scraper = Callable[[], str]
"""Returns one exposition body or raises :class:`ScrapeError`."""


def http_scraper(url: str, timeout: float = DEFAULT_SCRAPE_TIMEOUT_SECONDS) -> Scraper:
    """A scraper for ``url``. Redirects are refused and the body size is capped."""

    def scrape() -> str:
        try:
            body, _ = urllib_fetcher(url, timeout)
        except urllib.error.HTTPError as exc:
            raise ScrapeError(f"http_{exc.code}", f"HTTP {exc.code}") from exc
        except TimeoutError as exc:
            raise ScrapeError("timeout", "the scrape timed out") from exc
        except urllib.error.URLError as exc:
            raise ScrapeError("unreachable", str(exc.reason)) from exc
        except ServerMetricsError as exc:
            raise ScrapeError("too_large", str(exc)) from exc
        except OSError as exc:
            raise ScrapeError("unreachable", str(exc)) from exc
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ScrapeError("parse_error", "the body is not UTF-8") from exc

    return scrape


@dataclass
class GpuReading:
    """What one scrape says about one ``gpu`` label value."""

    index: int
    name: str | None = None
    uuid_sha256: str | None = None
    mig_seen: bool = False
    values: dict[str, float] = field(default_factory=dict)
    #: Metrics whose series for this GPU disagree, so no value is trusted.
    ambiguous: set[str] = field(default_factory=set)

    def energy(self, metric: EnergyMetric) -> tuple[float | None, str | None]:
        """Return ``(value, error)`` for one energy metric."""
        if metric.name in self.ambiguous:
            return None, AMBIGUOUS_SERIES
        value = self.values.get(metric.name)
        if value is None:
            return None, METRIC_MISSING
        return value, None

    def first(self, names: tuple[str, ...]) -> tuple[str | None, float | None]:
        for name in names:
            if name in self.values:
                return name, self.values[name]
        return None, None

    def energy_metric(self) -> EnergyMetric | None:
        """The finest energy counter this GPU reports unambiguously."""
        for metric in ENERGY_METRICS:
            if metric.name in self.values:
                return metric
        return None


def parse_gpus(text: str) -> dict[int, GpuReading]:
    """Group the energy, power and power limit series by their ``gpu`` label.

    A GPU can have several series for one metric (MIG instances or pod
    labels). They are accepted only when they all carry the same value;
    otherwise the metric is marked ambiguous. Two different UUIDs under one
    ``gpu`` label make every metric for it ambiguous.
    """
    families = parse_exposition(text)
    gpus: dict[int, GpuReading] = {}
    seen: dict[tuple[int, str], set[float]] = {}
    uuids: dict[int, set[str]] = {}
    for name in _WANTED:
        family = families.get(name)
        if family is None:
            continue
        for sample in family.samples:
            labels = dict(sample.labels)
            try:
                index = int(labels.get("gpu", ""))
            except ValueError:
                continue
            if index < 0:
                continue
            gpu = gpus.setdefault(index, GpuReading(index))
            if gpu.name is None and labels.get("modelName"):
                gpu.name = labels["modelName"]
            if labels.get("UUID"):
                uuids.setdefault(index, set()).add(labels["UUID"])
            if any(label in labels for label in MIG_LABELS):
                gpu.mig_seen = True
            seen.setdefault((index, name), set()).add(sample.value)
    for (index, name), values in seen.items():
        gpu = gpus[index]
        if len(values) == 1:
            gpu.values[name] = next(iter(values))
        else:
            gpu.ambiguous.add(name)
    for index, found in uuids.items():
        gpu = gpus[index]
        if len(found) == 1:
            gpu.uuid_sha256 = uuid_sha256(next(iter(found)))
        else:
            gpu.ambiguous.update(gpu.values)
            gpu.values.clear()
    return gpus


def scrape_gpus(scraper: Scraper) -> dict[int, GpuReading]:
    """Scrape once and parse. Every failure is a :class:`ScrapeError`."""
    text = scraper()
    try:
        return parse_gpus(text)
    except ExpositionParseError as exc:
        raise ScrapeError("parse_error", str(exc)) from exc


def endpoint_label(url: str) -> str:
    """The URL as validated for artifacts: no credentials, query or fragment."""
    try:
        return validate_metrics_url(url)
    except ServerMetricsError as exc:
        raise ValueError(str(exc)) from exc


# Capability ---------------------------------------------------------------

SUPPORTED = "supported"
UNREACHABLE = "unreachable"
NO_ENERGY_METRIC = "no_energy_metric"
NO_GPUS = "no_gpus"

INTERVAL_NOTE = (
    "dcgm-exporter does not publish its collect interval or sample "
    "timestamps. Its default --collect-interval is 30000 ms."
)


def _device_dict(gpu: GpuReading) -> dict[str, Any]:
    metric = gpu.energy_metric()
    if metric is not None:
        energy: dict[str, Any] = {
            "supported": True,
            "reason": None,
            "metric": metric.name,
            "unit": metric.unit,
            "resolution": metric.resolution,
        }
    else:
        ambiguous = any(m.name in gpu.ambiguous for m in ENERGY_METRICS)
        energy = {
            "supported": False,
            "reason": AMBIGUOUS_SERIES if ambiguous else METRIC_MISSING,
            "metric": None,
            "unit": None,
            "resolution": None,
        }
    power_metric, _ = gpu.first(POWER_METRICS)
    if power_metric is not None:
        power = {"supported": True, "reason": None, "metric": power_metric}
    else:
        ambiguous = any(name in gpu.ambiguous for name in POWER_METRICS)
        power = {
            "supported": False,
            "reason": AMBIGUOUS_SERIES if ambiguous else METRIC_MISSING,
            "metric": None,
        }
    _, limit = gpu.first(POWER_LIMIT_METRICS)
    return {
        "index": gpu.index,
        "name": gpu.name,
        "uuid_sha256": gpu.uuid_sha256,
        "mig_enabled": True if gpu.mig_seen else None,
        "enforced_power_limit_w": limit,
        "energy_counter": energy,
        "power_reading": power,
        "process_listing": {
            "supported": False,
            "reason": PROCESS_LISTING_UNAVAILABLE,
        },
    }


@dataclass(frozen=True)
class DcgmExporterCapability:
    state: str
    message: str
    endpoint: str
    devices: tuple[dict[str, Any], ...] = ()

    @property
    def is_supported(self) -> bool:
        return self.state == SUPPORTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": CAPABILITY_KIND,
            "schema_version": "1",
            "source": SOURCE,
            "state": self.state,
            "supported": self.is_supported,
            "message": self.message,
            "collector": {"name": SOURCE, "endpoint": self.endpoint},
            "collect_interval_note": INTERVAL_NOTE,
            "devices": list(self.devices),
        }


def check_exporter_capability(
    endpoint: str, scraper: Scraper
) -> DcgmExporterCapability:
    """Scrape once and report which GPUs expose an energy counter."""
    try:
        gpus = scrape_gpus(scraper)
    except ScrapeError as exc:
        return DcgmExporterCapability(
            state=UNREACHABLE,
            message=f"could not read {endpoint} ({exc.reason}): {exc}",
            endpoint=endpoint,
        )
    if not gpus:
        return DcgmExporterCapability(
            state=NO_GPUS,
            message="the exposition has no energy or power series with a gpu label",
            endpoint=endpoint,
        )
    devices = tuple(_device_dict(gpus[index]) for index in sorted(gpus))
    usable = [d["index"] for d in devices if d["energy_counter"]["supported"]]
    if not usable:
        return DcgmExporterCapability(
            state=NO_ENERGY_METRIC,
            message=(
                "no GPU has an unambiguous energy counter. Enable "
                f"{' or '.join(m.name for m in ENERGY_METRICS)} in the "
                "exporter's counters file"
            ),
            endpoint=endpoint,
            devices=devices,
        )
    return DcgmExporterCapability(
        state=SUPPORTED,
        message=f"dcgm-exporter reports an energy counter for {len(usable)} GPU(s)",
        endpoint=endpoint,
        devices=devices,
    )
