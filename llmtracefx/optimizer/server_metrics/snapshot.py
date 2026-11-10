"""Capture and persist one scrape of a server's Prometheus endpoint."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text, sha256_bytes
from ..schema import utc_now_iso
from .prometheus import (
    METRIC_TYPES,
    MetricFamily,
    Sample,
    decode_value,
    encode_value,
    parse_exposition,
)

SNAPSHOT_KIND = "llmtracefx.server_metrics.snapshot"
SNAPSHOT_SCHEMA_VERSION = "1"
DEFAULT_SCRAPE_TIMEOUT_SECONDS = 5.0
MAX_SCRAPE_BYTES = 32 * 1024 * 1024
_ACCEPT = "text/plain;version=0.0.4;q=1.0, */*;q=0.1"

Fetcher = Callable[[str, float], tuple[bytes, str | None]]
"""Takes ``(url, timeout_seconds)`` and returns ``(body, content_type)``."""


class ServerMetricsError(RuntimeError):
    """Raised when a scrape, snapshot or diff cannot be produced honestly."""


def validate_metrics_url(url: str) -> str:
    """Return ``scheme://host[:port]/path`` or raise.

    Nothing secret is sent with a scrape, so plain HTTP is allowed to any
    host. Credentials, queries and fragments are refused because the
    endpoint is written into artifacts and a query string is a common place
    for a token.
    """
    if not isinstance(url, str) or not url or any(ord(c) < 33 for c in url):
        raise ServerMetricsError("metrics URL must be a non-empty URL without spaces")
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ServerMetricsError(f"metrics URL is not valid: {exc}") from exc
    if parts.scheme not in ("http", "https"):
        raise ServerMetricsError("metrics URL must use http or https")
    if not parts.hostname:
        raise ServerMetricsError("metrics URL must include a host")
    if parts.username is not None or parts.password is not None:
        raise ServerMetricsError("metrics URL must not embed credentials")
    if parts.query:
        raise ServerMetricsError("metrics URL must not include a query string")
    if parts.fragment:
        raise ServerMetricsError("metrics URL must not include a fragment")
    host = parts.hostname
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path or "/", "", ""))


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[no-untyped-def]
        self, req, fp, code, msg, headers, newurl
    ):
        raise urllib.error.HTTPError(
            req.full_url, code, "redirects are not followed", headers, fp
        )


def urllib_fetcher(url: str, timeout: float) -> tuple[bytes, str | None]:
    opener = urllib.request.build_opener(_RefuseRedirects)
    request = urllib.request.Request(url, headers={"Accept": _ACCEPT}, method="GET")
    with opener.open(request, timeout=timeout) as response:
        body = response.read(MAX_SCRAPE_BYTES + 1)
        content_type = response.headers.get("Content-Type")
    if len(body) > MAX_SCRAPE_BYTES:
        raise ServerMetricsError(
            f"metrics response exceeds the {MAX_SCRAPE_BYTES}-byte limit"
        )
    return body, content_type


@dataclass(frozen=True)
class ServerMetricsSnapshot:
    endpoint: str
    captured_at: str
    """UTC time the scrape finished."""
    scrape_duration_ms: float
    content_type: str | None
    body_sha256: str
    body_bytes: int
    families: dict[str, MetricFamily]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": SNAPSHOT_KIND,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "endpoint": self.endpoint,
            "captured_at": self.captured_at,
            "scrape_duration_ms": self.scrape_duration_ms,
            "content_type": self.content_type,
            "body_sha256": self.body_sha256,
            "body_bytes": self.body_bytes,
            "families": [
                {
                    "name": family.name,
                    "type": family.type,
                    "help": family.help,
                    "samples": [
                        {
                            "name": sample.name,
                            "labels": dict(sample.labels),
                            "value": encode_value(sample.value),
                        }
                        for sample in family.samples
                    ],
                }
                for family in self.families.values()
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    def write_json(self, path: Path) -> None:
        atomic_write_text(path, self.to_json() + "\n")

    @classmethod
    def from_dict(cls, data: Any) -> ServerMetricsSnapshot:
        try:
            if not isinstance(data, dict) or data.get("kind") != SNAPSHOT_KIND:
                raise ValueError(f"not a {SNAPSHOT_KIND} document")
            if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema_version {data.get('schema_version')!r}"
                )
            families: dict[str, MetricFamily] = {}
            for raw in data["families"]:
                name = _string(raw["name"], "family name")
                metric_type = _string(raw["type"], "family type")
                if metric_type not in METRIC_TYPES:
                    raise ValueError(f"unknown metric type {metric_type!r}")
                samples = tuple(
                    Sample(
                        name=_string(item["name"], "sample name"),
                        labels=tuple(
                            sorted(
                                (_string(k, "label"), _string(v, "label value"))
                                for k, v in dict(item["labels"]).items()
                            )
                        ),
                        value=decode_value(item["value"]),
                    )
                    for item in raw["samples"]
                )
                if name in families:
                    raise ValueError(f"duplicate family {name}")
                help_text = raw.get("help")
                families[name] = MetricFamily(
                    name=name,
                    type=metric_type,
                    help=None if help_text is None else _string(help_text, "help"),
                    samples=samples,
                )
            duration = data["scrape_duration_ms"]
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError("scrape_duration_ms must be a number")
            body_bytes = data["body_bytes"]
            if isinstance(body_bytes, bool) or not isinstance(body_bytes, int):
                raise ValueError("body_bytes must be an integer")
            content_type = data.get("content_type")
            return cls(
                endpoint=_string(data["endpoint"], "endpoint"),
                captured_at=_string(data["captured_at"], "captured_at"),
                scrape_duration_ms=float(duration),
                content_type=(
                    None if content_type is None else _string(content_type, "type")
                ),
                body_sha256=_string(data["body_sha256"], "body_sha256"),
                body_bytes=body_bytes,
                families=dict(sorted(families.items())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ServerMetricsError(f"invalid server metrics snapshot: {exc}") from exc

    @classmethod
    def read_json(cls, path: Path) -> ServerMetricsSnapshot:
        try:
            text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
            data = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ValueError) as exc:
            raise ServerMetricsError(f"cannot read snapshot {path}: {exc}") from exc
        return cls.from_dict(data)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def snapshot_from_body(
    endpoint: str,
    body: bytes,
    content_type: str | None,
    *,
    captured_at: str,
    scrape_duration_ms: float,
) -> ServerMetricsSnapshot:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ServerMetricsError("metrics response is not UTF-8") from exc
    try:
        families = parse_exposition(text)
    except ValueError as exc:
        raise ServerMetricsError(f"metrics response did not parse: {exc}") from exc
    return ServerMetricsSnapshot(
        endpoint=endpoint,
        captured_at=captured_at,
        scrape_duration_ms=scrape_duration_ms,
        content_type=content_type,
        body_sha256=sha256_bytes(body),
        body_bytes=len(body),
        families=families,
    )


def capture_snapshot(
    url: str,
    *,
    timeout_seconds: float = DEFAULT_SCRAPE_TIMEOUT_SECONDS,
    fetcher: Fetcher = urllib_fetcher,
    clock: Callable[[], float] = time.perf_counter,
    now: Callable[[], str] = utc_now_iso,
) -> ServerMetricsSnapshot:
    endpoint = validate_metrics_url(url)
    started = clock()
    try:
        body, content_type = fetcher(endpoint, timeout_seconds)
    except ServerMetricsError:
        raise
    except urllib.error.HTTPError as exc:
        raise ServerMetricsError(
            f"metrics endpoint returned HTTP {exc.code}: {exc.reason}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ServerMetricsError(f"metrics scrape failed: {exc}") from exc
    finished = clock()
    return snapshot_from_body(
        endpoint,
        body,
        content_type,
        captured_at=now(),
        scrape_duration_ms=max(0.0, finished - started) * 1000,
    )
