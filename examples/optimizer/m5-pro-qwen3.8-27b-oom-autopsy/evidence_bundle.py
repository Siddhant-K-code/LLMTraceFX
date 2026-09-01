#!/usr/bin/env python3
"""Generate and verify the committed clean-boot OOM evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PUBLIC_DIR = Path(__file__).resolve().parent / "publication"
MAX_PUBLIC_FILE_BYTES = 2 * 1024 * 1024
GIB = 1024**3

SOURCE_SHA256 = {
    "oom-autopsy-checkpoints.csv": (
        "325bc7aa37a2e1d92edd6305e5a8bc0cd343bffe20d5202f06b9194631644515"
    ),
    "oom-autopsy-report.html": (
        "d3408cf9220c9c168630374035cccd97b5a2edf4d1c0184f3214cf7ea20fcc39"
    ),
    "oom-autopsy-summary.json": (
        "2cb09a3e1e743cd70c7289388f00b92e5405ebba6027fe61d7005cc6965454c0"
    ),
}
PLAN_SHA256 = "91c5b983ec1c039f0af8c2abf74dabb844998e20a0adc48363974018e76ed830"
HASHED_FILES = (
    "autopsy-plan.json",
    "evidence-manifest.json",
    "mlx-memory-by-stage.svg",
    "oom-autopsy-checkpoints.csv",
    "oom-autopsy-report.html",
    "oom-autopsy-summary.json",
)
PUBLIC_FILES = (*HASHED_FILES, "SHA256SUMS")

CODE_COMMIT = "2519bc8da309656d2e2ce2a7063f19b0dfb4c9ed"
MODEL_ID = "mlx-community/Qwen3.8-27B-4bit"
MODEL_REVISION = "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
MODEL_BYTES = 16081490933
MODEL_FILE_COUNT = 15
STAGES = (
    "child_start",
    "before_model_load",
    "after_model_load",
    "after_prompt_tokenization",
    "immediately_before_prefill_generation",
    "caught_oom",
    "cleanup",
)

FORBIDDEN_PUBLIC_PATTERNS = (
    (re.compile(r"/(?:Users|home)/", re.IGNORECASE), "absolute home path"),
    (re.compile(r"\b[A-Za-z]:\\Users\\", re.IGNORECASE), "Windows home path"),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "email address"),
    (
        re.compile(r"\b(?:gh[pousr]_|sk-|hf_)[A-Za-z0-9_-]{8,}\b"),
        "credential-like token",
    ),
    (
        re.compile(r"siddhant-git-ai|siddhantkhare2694", re.IGNORECASE),
        "private username",
    ),
    (
        re.compile(r"(?:huggingface[/\\]hub|\.cache[/\\]models)", re.IGNORECASE),
        "model cache path",
    ),
    (
        re.compile(r"(?:^|[/\\])(?:journal|result|state)\.json\b", re.IGNORECASE),
        "private run artifact name",
    ),
    (re.compile(r"(?:^|[/\\])logs?[/\\]", re.IGNORECASE), "private log path"),
    (re.compile(r"\bfile://", re.IGNORECASE), "local file URL"),
)
FORBIDDEN_JSON_KEYS = {
    "absolute_path",
    "cache_path",
    "home",
    "host_name",
    "hostname",
    "local_path",
    "model_path",
    "pid",
    "process_id",
    "raw_prompt",
    "raw_response",
    "user_name",
    "username",
}
EXTERNAL_HTML_PATTERNS = (
    re.compile(r"<script\b", re.IGNORECASE),
    re.compile(r"<iframe\b", re.IGNORECASE),
    re.compile(r"<link\b", re.IGNORECASE),
    re.compile(r"\b(?:src|href)\s*=\s*[\"']\s*(?:https?:|//)", re.IGNORECASE),
    re.compile(r"url\s*\(", re.IGNORECASE),
    re.compile(r"@import\b", re.IGNORECASE),
)


class EvidenceError(ValueError):
    """Raised when the public evidence bundle is unsafe or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{path.name} must be a regular non-symlink file")
    size = path.stat().st_size
    if size > MAX_PUBLIC_FILE_BYTES:
        raise EvidenceError(
            f"{path.name} exceeds the {MAX_PUBLIC_FILE_BYTES}-byte public file limit"
        )
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise EvidenceError(f"{path.name} is not valid UTF-8") from exc


def _reject_json_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number {value!r} is not allowed")


def _check_json_value(value: Any, *, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{context} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_JSON_KEYS:
                raise EvidenceError(f"{context}.{key} is a private identity/path field")
            _check_json_value(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json_value(item, context=f"{context}[{index}]")


def _load_json(path: Path) -> dict[str, Any]:
    text = _read_regular_text(path)
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"{path.name} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path.name} must contain a JSON object")
    _check_json_value(value)
    return value


def scan_privacy(name: str, text: str) -> None:
    for pattern, label in FORBIDDEN_PUBLIC_PATTERNS:
        if pattern.search(text):
            raise EvidenceError(f"{name} contains {label}")


def _validate_csv(path: Path) -> None:
    text = _read_regular_text(path)
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except csv.Error as exc:
        raise EvidenceError(f"{path.name} is invalid CSV: {exc}") from exc
    if len(rows) != len(STAGES):
        raise EvidenceError("checkpoint CSV must contain exactly seven stage rows")
    if tuple(row.get("stage") for row in rows) != STAGES:
        raise EvidenceError("checkpoint CSV stage order drifted")
    numeric_columns = (
        "monotonic_offset_seconds",
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "host_rss_bytes",
        "host_max_rss_bytes",
        "swap_total_bytes",
        "swap_used_bytes",
    )
    for row in rows:
        for column in numeric_columns:
            raw = row.get(column)
            if raw in (None, ""):
                continue
            try:
                number = float(raw)
            except ValueError as exc:
                raise EvidenceError(f"checkpoint CSV {column} is not numeric") from exc
            if not math.isfinite(number):
                raise EvidenceError(f"checkpoint CSV {column} is non-finite")


def _manifest_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_id": "m5-pro-qwen3.8-27b-clean-boot-oom-autopsy-20260901",
        "run": {
            "completed_at_utc": "2026-09-01T17:45:36.921331Z",
            "code_checkout_commit": CODE_COMMIT,
            "mode": "publication",
            "clean_boot_operator_assertion": True,
            "terminal_outcome": "oom",
            "reason": "MLX/Metal reported insufficient memory",
            "child_exit_code": 2,
            "timed_out": False,
            "descendants_cleaned": True,
            "journal_complete": True,
            "journal_terminal": "oom",
        },
        "model": {
            "repository_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_bytes": MODEL_BYTES,
            "checkpoint_file_count": MODEL_FILE_COUNT,
        },
        "workload": {
            "tier": "t256",
            "requested_prompt_tokens": 256,
            "actual_prompt_tokens": 256,
            "first_token_observed": False,
            "generation_completion_observed": False,
            "evaluator_result_available": False,
            "quality_metrics_available": False,
            "throughput_metrics_available": False,
        },
        "source_plan": {
            "file": "autopsy-plan.json",
            "sha256": PLAN_SHA256,
        },
        "source_reports": SOURCE_SHA256,
        "scope_contract": {
            "mlx_active_cache_peak": "MLX allocator counters; bytes",
            "rss": "host process current/max RSS; bytes",
            "swap": "host system swap total/used; bytes",
            "available_memory_estimate": (
                "approximate system headroom from macOS memory_pressure; "
                "not GPU memory"
            ),
            "combination_rule": (
                "Scopes are non-additive and must not be summed or relabeled "
                "as GPU memory"
            ),
        },
        "limitations": [
            (
                "This is bounded evidence for one recorded machine state, exact "
                "checkpoint, runtime, and t256 workload."
            ),
            (
                "It is not a universal memory-capacity boundary and not a "
                "universal 24 GB boundary."
            ),
            (
                "Checkpoint deltas are observations, not causal allocation "
                "attribution."
            ),
            (
                "No first-token, completion, evaluator, quality, or throughput "
                "measurement exists."
            ),
            (
                "No GPU utilization, free GPU memory, bandwidth, power, energy, "
                "or kernel time is measured or inferred."
            ),
            (
                "Observer overhead from stage probes was not separately measured "
                "or subtracted."
            ),
            (
                "Periodic sampling was disabled; only discrete stage-boundary "
                "checkpoints were recorded."
            ),
        ],
    }


def _assert_subset(
    actual: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise EvidenceError(f"{context}.{key} is missing")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise EvidenceError(f"{context}.{key} must be an object")
            _assert_subset(actual_value, expected_value, f"{context}.{key}")
        elif actual_value != expected_value:
            raise EvidenceError(
                f"{context}.{key} drifted: expected {expected_value!r}, "
                f"got {actual_value!r}"
            )


def _validate_plan(plan: dict[str, Any]) -> None:
    expected = {
        "weights_loaded": False,
        "downloads_performed": False,
        "autopsy_id": "m5-pro-qwen3.8-27b-oom-autopsy-v1",
        "run_mode": "publication",
        "clean_boot_confirmed": True,
        "publication_ready": True,
        "model_present_by_size": True,
        "model": {"repository_id": MODEL_ID, "revision": MODEL_REVISION},
        "tier": {"name": "t256", "requested_tokens": 256},
        "sampling": {"periodic_sampling_enabled": False},
        "machine_state": {
            "os_name": "Darwin",
            "os_release": "25.6.0",
            "architecture": "arm64",
            "chip": "Apple M5 Pro",
            "physical_memory_bytes": 25769803776,
            "available_memory_estimate_bytes": 21131239096,
            "memory_free_percent": 82.0,
            "swap_used_bytes": 0,
            "package_versions": {
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "mlx-vlm": "0.6.8",
                "transformers": "5.16.1",
            },
        },
        "safety": {"safe": True, "blockers": []},
    }
    _assert_subset(plan, expected, "plan")


def _validate_summary(summary: dict[str, Any]) -> None:
    expected = {
        "schema_version": "1",
        "autopsy_id": "m5-pro-qwen3.8-27b-oom-autopsy-v1",
        "generated_at": "2026-09-01T17:45:36.921331Z",
        "run_mode": "publication",
        "clean_boot_confirmed": True,
        "synthetic": False,
        "model": {"repository_id": MODEL_ID, "revision": MODEL_REVISION},
        "tier": {"name": "t256", "requested_tokens": 256},
        "terminal_outcome": "oom",
        "reason": "MLX/Metal reported insufficient memory",
        "actual_tokens": 256,
        "journal_complete": True,
        "journal_terminal": "oom",
        "sampling": {"periodic_sampling_enabled": False},
        "pre_run_machine_state": {
            "os_name": "Darwin",
            "os_release": "25.6.0",
            "architecture": "arm64",
            "chip": "Apple M5 Pro",
            "physical_memory_bytes": 25769803776,
            "available_memory_estimate_bytes": 20358144983,
            "memory_free_percent": 79.0,
            "swap_used_bytes": 0,
            "package_versions": {
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "mlx-vlm": "0.6.8",
                "transformers": "5.16.1",
            },
        },
        "provenance": {
            "mlx_active_bytes": {"scope": "mlx_allocator", "unit": "bytes"},
            "mlx_cache_bytes": {"scope": "mlx_allocator", "unit": "bytes"},
            "mlx_peak_bytes": {"scope": "mlx_allocator", "unit": "bytes"},
            "host_rss_bytes": {"scope": "host_process", "unit": "bytes"},
            "host_max_rss_bytes": {"scope": "host_process", "unit": "bytes"},
            "swap_bytes": {"scope": "host_system_swap", "unit": "bytes"},
        },
    }
    _assert_subset(summary, expected, "summary")
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise EvidenceError("summary.checkpoints must be an array")
    if tuple(item.get("stage") for item in checkpoints) != STAGES:
        raise EvidenceError("summary checkpoint stage order drifted")
    if tuple(item.get("sequence") for item in checkpoints) != tuple(range(len(STAGES))):
        raise EvidenceError("summary checkpoint sequence drifted")
    if any(
        item.get("stage") in {"after_first_token", "completion"} for item in checkpoints
    ):
        raise EvidenceError("summary must not claim first-token or completion stages")

    deltas = summary.get("observed_deltas")
    if not isinstance(deltas, list):
        raise EvidenceError("summary.observed_deltas must be an array")
    boundary = next(
        (
            item
            for item in deltas
            if item.get("from_stage") == "immediately_before_prefill_generation"
            and item.get("to_stage") == "caught_oom"
        ),
        None,
    )
    _assert_subset(
        boundary if isinstance(boundary, dict) else {},
        {
            "mlx_active_bytes_delta": 2672187942,
            "mlx_cache_bytes_delta": 76420156,
            "mlx_peak_bytes_delta": 2839022214,
        },
        "summary.prefill_boundary_delta",
    )


def _svg_series(
    *,
    checkpoints: list[dict[str, Any]],
    key: str,
    label: str,
    color: str,
    x_positions: list[float],
    top: float,
    height: float,
    maximum: float,
) -> list[str]:
    parts: list[str] = []
    segment: list[tuple[float, float, int, str]] = []

    def flush() -> None:
        if not segment:
            return
        if len(segment) > 1:
            points = " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in segment)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2"/>'
            )
        for x, y, value, stage in segment:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" '
                f'data-series="{html.escape(label, quote=True)}" '
                f'data-stage="{html.escape(stage, quote=True)}" '
                f'data-bytes="{value}"><title>{html.escape(label)} at '
                f"{html.escape(stage)}: {value} bytes "
                f"({value / GIB:.6f} GiB)</title></circle>"
            )
        segment.clear()

    for checkpoint, x in zip(checkpoints, x_positions, strict=True):
        raw = checkpoint.get(key)
        if raw is None:
            flush()
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise EvidenceError(f"{key} must be a non-negative integer or null")
        y = top + height - (raw / maximum * height)
        segment.append((x, y, raw, str(checkpoint["stage"])))
    flush()
    return parts


def _panel(
    *,
    checkpoints: list[dict[str, Any]],
    title: str,
    scope: str,
    series: Iterable[tuple[str, str, str]],
    x_positions: list[float],
    top: float,
    left: float,
    width: float,
    height: float,
) -> list[str]:
    configured = tuple(series)
    values = [
        item[key]
        for item in checkpoints
        for key, _, _ in configured
        if isinstance(item.get(key), int) and not isinstance(item.get(key), bool)
    ]
    maximum = float(max(values, default=1))
    parts = [
        f'<text x="{left:.1f}" y="{top - 24:.1f}" fill="#17202a" '
        'font-family="-apple-system,BlinkMacSystemFont,sans-serif" '
        f'font-size="17" font-weight="600">{html.escape(title)}</text>',
        f'<text x="{left + width:.1f}" y="{top - 24:.1f}" text-anchor="end" '
        'fill="#5b6167" font-family="ui-monospace,monospace" font-size="12">'
        f"scope: {html.escape(scope)}; independent axis; bytes / 2^30</text>",
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" fill="#fffdf8" stroke="#d9d1c3"/>',
        f'<text x="{left - 12:.1f}" y="{top + 5:.1f}" text-anchor="end" '
        'fill="#5b6167" font-family="ui-monospace,monospace" font-size="11">'
        f"{maximum / GIB:.3f} GiB</text>",
        f'<text x="{left - 12:.1f}" y="{top + height:.1f}" text-anchor="end" '
        'fill="#5b6167" font-family="ui-monospace,monospace" font-size="11">'
        "0</text>",
    ]
    oom_index = STAGES.index("caught_oom")
    oom_x = x_positions[oom_index]
    parts.extend(
        [
            f'<line x1="{oom_x:.1f}" y1="{top:.1f}" x2="{oom_x:.1f}" '
            f'y2="{top + height:.1f}" stroke="#9b2c2c" stroke-width="1.5" '
            'stroke-dasharray="5 4"/>',
            f'<text x="{oom_x + 6:.1f}" y="{top + 16:.1f}" fill="#9b2c2c" '
            'font-family="ui-monospace,monospace" font-size="11">OOM boundary</text>',
        ]
    )
    for key, label, color in configured:
        parts.extend(
            _svg_series(
                checkpoints=checkpoints,
                key=key,
                label=label,
                color=color,
                x_positions=x_positions,
                top=top,
                height=height,
                maximum=maximum,
            )
        )
    legend_x = left
    for _, label, color in configured:
        parts.extend(
            [
                f'<rect x="{legend_x:.1f}" y="{top + height + 12:.1f}" '
                f'width="11" height="11" fill="{color}"/>',
                f'<text x="{legend_x + 17:.1f}" y="{top + height + 22:.1f}" '
                'fill="#17202a" font-family="ui-monospace,monospace" '
                f'font-size="11">{html.escape(label)}</text>',
            ]
        )
        legend_x += 170
    return parts


def render_chart(summary: dict[str, Any]) -> str:
    """Render the three-scope SVG solely from the sanitized summary object."""
    _validate_summary(summary)
    checkpoints = summary["checkpoints"]
    width, height = 1200, 920
    left, right = 150.0, 45.0
    plot_width = width - left - right
    x_positions = [
        left + index * plot_width / (len(checkpoints) - 1)
        for index in range(len(checkpoints))
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Qwen3.8-27B clean-boot OOM autopsy</title>',
        '<desc id="desc">Stage checkpoints shown on three independent axes: '
        "MLX allocator counters, host process RSS, and host system swap. "
        "Missing values are omitted, never converted to zero.</desc>",
        f'<rect width="{width}" height="{height}" fill="#f7f3ea"/>',
        '<text x="48" y="48" fill="#17202a" '
        'font-family="-apple-system,BlinkMacSystemFont,sans-serif" '
        'font-size="24" font-weight="700">Qwen3.8-27B clean-boot OOM autopsy</text>',
        '<text x="48" y="76" fill="#5b6167" '
        'font-family="ui-monospace,monospace" font-size="13">'
        "Exact bytes are embedded per point; GiB labels are binary conversions.</text>",
    ]
    panel_specs = (
        (
            "MLX allocator counters",
            "mlx_allocator",
            (
                ("mlx_active_bytes", "MLX active", "#c0392b"),
                ("mlx_cache_bytes", "MLX cache", "#d68910"),
                ("mlx_peak_bytes", "MLX peak", "#7d3c98"),
            ),
            130.0,
        ),
        (
            "Host process memory",
            "host_process",
            (
                ("host_rss_bytes", "current RSS", "#2471a3"),
                ("host_max_rss_bytes", "max RSS", "#148f77"),
            ),
            385.0,
        ),
        (
            "Host system swap",
            "host_system_swap",
            (
                ("swap_used_bytes", "swap used", "#7d6608"),
                ("swap_total_bytes", "swap total", "#5d6d7e"),
            ),
            640.0,
        ),
    )
    for title, scope, series, top in panel_specs:
        parts.extend(
            _panel(
                checkpoints=checkpoints,
                title=title,
                scope=scope,
                series=series,
                x_positions=x_positions,
                top=top,
                left=left,
                width=plot_width,
                height=170.0,
            )
        )
    for x, stage in zip(x_positions, STAGES, strict=True):
        display = {
            "before_model_load": "before load",
            "after_model_load": "after load",
            "after_prompt_tokenization": "tokenized",
            "immediately_before_prefill_generation": "before prefill",
            "caught_oom": "caught OOM",
            "child_start": "child start",
            "cleanup": "cleanup",
        }[stage]
        parts.append(
            f'<text x="{x:.1f}" y="893" text-anchor="middle" fill="#17202a" '
            'font-family="ui-monospace,monospace" font-size="11">'
            f"{html.escape(display)}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _parse_checksums(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise EvidenceError("SHA256SUMS contains a malformed line")
        digest, name = match.groups()
        if name in parsed:
            raise EvidenceError(f"SHA256SUMS repeats {name}")
        parsed[name] = digest
    if tuple(parsed) != HASHED_FILES:
        raise EvidenceError("SHA256SUMS file order or allowlist drifted")
    return parsed


def write_generated_files(public_dir: Path = PUBLIC_DIR) -> None:
    report_path = public_dir / "oom-autopsy-report.html"
    report_bytes = report_path.read_bytes()
    if (
        report_bytes.endswith(b"\n")
        and hashlib.sha256(report_bytes[:-1]).hexdigest()
        == SOURCE_SHA256["oom-autopsy-report.html"]
    ):
        report_path.write_bytes(report_bytes[:-1])
    summary = _load_json(public_dir / "oom-autopsy-summary.json")
    (public_dir / "mlx-memory-by-stage.svg").write_text(
        render_chart(summary), encoding="utf-8"
    )
    lines = [f"{_sha256(public_dir / name)}  {name}" for name in HASHED_FILES]
    (public_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_bundle(public_dir: Path = PUBLIC_DIR) -> None:
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise EvidenceError("publication directory must be a real directory")
    entries = sorted(path.name for path in public_dir.iterdir())
    if entries != sorted(PUBLIC_FILES):
        missing = sorted(set(PUBLIC_FILES) - set(entries))
        unexpected = sorted(set(entries) - set(PUBLIC_FILES))
        raise EvidenceError(
            f"public file allowlist drifted; missing={missing}, unexpected={unexpected}"
        )

    texts: dict[str, str] = {}
    for name in PUBLIC_FILES:
        text = _read_regular_text(public_dir / name)
        scan_privacy(name, text)
        texts[name] = text
    for pattern in EXTERNAL_HTML_PATTERNS:
        if pattern.search(texts["oom-autopsy-report.html"]):
            raise EvidenceError("HTML report contains an external/network resource")
        if pattern.search(texts["mlx-memory-by-stage.svg"]):
            raise EvidenceError("SVG chart contains an external/network resource")

    checksums = _parse_checksums(texts["SHA256SUMS"])
    for name, expected in checksums.items():
        if _sha256(public_dir / name) != expected:
            raise EvidenceError(f"{name} does not match SHA256SUMS")
    for name, expected in SOURCE_SHA256.items():
        if checksums.get(name) != expected:
            raise EvidenceError(f"{name} no longer matches the verified source hash")
    if checksums.get("autopsy-plan.json") != PLAN_SHA256:
        raise EvidenceError("autopsy-plan.json no longer matches the reviewed source")

    plan = _load_json(public_dir / "autopsy-plan.json")
    manifest = _load_json(public_dir / "evidence-manifest.json")
    summary = _load_json(public_dir / "oom-autopsy-summary.json")
    _validate_plan(plan)
    _validate_summary(summary)
    expected_manifest = _manifest_contract()
    if manifest != expected_manifest:
        _assert_subset(manifest, expected_manifest, "manifest")
        raise EvidenceError("manifest contains unsupported fields or claims")

    _validate_csv(public_dir / "oom-autopsy-checkpoints.csv")
    if render_chart(summary) != texts["mlx-memory-by-stage.svg"]:
        raise EvidenceError("SVG chart is not the deterministic summary rendering")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or verify the committed OOM evidence bundle"
    )
    parser.add_argument(
        "action", nargs="?", choices=("verify", "generate"), default="verify"
    )
    parser.add_argument("--public-dir", type=Path, default=PUBLIC_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.action == "generate":
            write_generated_files(args.public_dir)
        verify_bundle(args.public_dir)
    except (EvidenceError, OSError) as exc:
        print(f"OOM evidence bundle verification failed: {exc}")
        return 2
    print("OOM evidence bundle verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
