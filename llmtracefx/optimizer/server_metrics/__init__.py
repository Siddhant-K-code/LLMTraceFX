"""Server-side evidence from an inference server's Prometheus endpoint.

Client timing includes the network and cannot split queueing from
prefill. vLLM, SGLang, llama.cpp and TGI record those phases themselves
and expose them on ``/metrics``. Scraping before and after a measured
request and differencing the two scrapes recovers the server's own view,
provided nothing else touched the server in between. :func:`diff_snapshots`
checks that and refuses to attribute a window it cannot vouch for.
"""

from .diff import (
    ATTRIBUTABLE_STATUSES,
    PROFILES,
    RUNTIME_CHOICES,
    capture_settled_after,
    detect_runtime,
    diff_snapshots,
    write_evidence,
)
from .prometheus import ExpositionParseError, MetricFamily, Sample, parse_exposition
from .snapshot import (
    ServerMetricsError,
    ServerMetricsSnapshot,
    capture_snapshot,
    snapshot_from_body,
    validate_metrics_url,
)

__all__ = [
    "ATTRIBUTABLE_STATUSES",
    "PROFILES",
    "RUNTIME_CHOICES",
    "ExpositionParseError",
    "MetricFamily",
    "Sample",
    "ServerMetricsError",
    "ServerMetricsSnapshot",
    "capture_settled_after",
    "capture_snapshot",
    "detect_runtime",
    "diff_snapshots",
    "parse_exposition",
    "snapshot_from_body",
    "validate_metrics_url",
    "write_evidence",
]
