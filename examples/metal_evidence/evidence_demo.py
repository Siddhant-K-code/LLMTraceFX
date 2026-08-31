#!/usr/bin/env python3
"""Capture and package privacy-safe Metal System Trace evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from llmtracefx.optimizer.instruments.capability import (
    METAL_SYSTEM_TRACE_TEMPLATE,
    detect_xctrace_capability,
)
from llmtracefx.optimizer.instruments.export import (
    parse_exported_table,
    parse_table_of_contents,
    read_export_text,
    summarize_metal_gpu_intervals,
)
from llmtracefx.optimizer.instruments.process import (
    SubprocessCommandRunner,
    SubprocessProcessLauncher,
)
from llmtracefx.optimizer.instruments.workflow import import_trace, record_trace

SCHEMA_VERSION = 1
DEFAULT_DISPATCH_COUNTS = (400, 250, 120, 77, 133)
DEFAULT_TIME_LIMIT = "10s"
PUBLIC_DIR_NAME = "public"
PRIVATE_DIR_NAME = "private"
WORKLOAD_NAME = "metal-evidence-workload"
TABLE_SCHEMA = "metal-gpu-intervals"
PUBLIC_CONTENT_FILES = (
    "capture-summary.csv",
    "capture-summary.json",
    "dispatch-attribution.svg",
    "experiment-manifest.json",
    "unrelated-interval-share.svg",
)
PUBLIC_FILES = (*PUBLIC_CONTENT_FILES, "SHA256SUMS")
FORBIDDEN_PUBLIC_PATTERNS = (
    (re.compile(r"/Users/", re.IGNORECASE), "absolute macOS home path"),
    (
        re.compile(r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", re.IGNORECASE),
        "UUID",
    ),
    (
        re.compile(r"\b(?:gh[pousr]_|sk-|hf_)[A-Za-z0-9_-]{8,}\b"),
        "credential-like token",
    ),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "email address"),
    (re.compile(r"\.trace(?:/|\b)"), "raw trace path"),
)


@dataclass(frozen=True)
class CaptureSummary:
    expected_dispatch_count: int
    attributed_interval_count: int
    all_process_interval_count: int
    known_unrelated_interval_count: int
    known_unrelated_interval_share_percent: float
    unattributed_interval_count: int
    unattributed_interval_share_percent: float
    window_server_interval_count: int
    window_server_interval_share_percent: float
    schema_count: int
    exported_rows: int
    exported_columns: int
    exported_cells: int
    reference_cells: int
    dispatch_count_matches: bool
    attribution: str = "TOC target PID"


def _run_checked(argv: Sequence[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"{argv[0]} failed with exit {result.returncode}: {detail}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _direct_row_cells(root: ElementTree.Element) -> list[ElementTree.Element]:
    return [cell for row in root.findall("./node/row") for cell in list(row)]


def summarize_exports(
    toc_xml: str, table_xml: str, *, expected_dispatch_count: int
) -> CaptureSummary:
    """Derive the public allowlisted summary from private xctrace XML."""
    toc = parse_table_of_contents(toc_xml)
    run = toc.runs[0]
    if run.target_pid is None:
        raise ValueError("trace TOC does not identify a target PID")

    table = parse_exported_table(table_xml, expected_schema=TABLE_SCHEMA)
    summary = summarize_metal_gpu_intervals(table)
    target = summary.for_process(run.target_pid)
    if target is None:
        raise ValueError("target PID has no attributed Metal intervals")
    unattributed = summary.unattributed_interval_count
    known_unrelated = (
        summary.total_interval_count - target.interval_count - unattributed
    )
    window_server = sum(
        entry.interval_count
        for entry in summary.per_process
        if entry.pid != run.target_pid
        and entry.process_label.partition(" (")[0] == "WindowServer"
    )
    root = ElementTree.fromstring(table_xml)
    cells = _direct_row_cells(root)
    cell_count = len(cells)
    expected_cells = table.row_count * len(table.columns)
    if cell_count != expected_cells:
        raise ValueError(
            f"row-cell count {cell_count} does not match parsed shape {expected_cells}"
        )

    return CaptureSummary(
        expected_dispatch_count=expected_dispatch_count,
        attributed_interval_count=target.interval_count,
        all_process_interval_count=summary.total_interval_count,
        known_unrelated_interval_count=known_unrelated,
        known_unrelated_interval_share_percent=round(
            known_unrelated * 100.0 / summary.total_interval_count, 1
        ),
        unattributed_interval_count=unattributed,
        unattributed_interval_share_percent=round(
            unattributed * 100.0 / summary.total_interval_count, 1
        ),
        window_server_interval_count=window_server,
        window_server_interval_share_percent=round(
            window_server * 100.0 / summary.total_interval_count, 1
        ),
        schema_count=len(run.schemas),
        exported_rows=table.row_count,
        exported_columns=len(table.columns),
        exported_cells=cell_count,
        reference_cells=sum(cell.get("ref") is not None for cell in cells),
        dispatch_count_matches=target.interval_count == expected_dispatch_count,
    )


def _safe_host_metadata() -> dict[str, str]:
    brand = _run_checked(("sysctl", "-n", "machdep.cpu.brand_string"))
    macos_version = _run_checked(("sw_vers", "-productVersion"))
    macos_build = _run_checked(("sw_vers", "-buildVersion"))
    xcode_lines = _run_checked(("xcodebuild", "-version")).splitlines()
    instruments = _run_checked(
        (
            "defaults",
            "read",
            "/Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/Info.plist",
            "CFBundleShortVersionString",
        )
    )
    xctrace = _run_checked(("xcrun", "xctrace", "version"))
    return {
        "hardware": brand,
        "architecture": platform.machine(),
        "macos": macos_version,
        "macos_build": macos_build,
        "xcode": xcode_lines[0],
        "xcode_build": xcode_lines[1].removeprefix("Build version "),
        "instruments": instruments,
        "xctrace": xctrace,
    }


def _prepare_output(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"output directory must be absent or empty to prevent stale artifact mixing: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / PRIVATE_DIR_NAME
    public_dir = output_dir / PUBLIC_DIR_NAME
    private_dir.mkdir()
    public_dir.mkdir()
    return private_dir, public_dir


def _remove_private_artifacts(private_dir: Path) -> None:
    shutil.rmtree(private_dir)
    if private_dir.exists():
        raise RuntimeError(f"private capture artifacts remain at {private_dir}")


def _compile_workload(source: Path, binary: Path) -> None:
    _run_checked(
        (
            "xcrun",
            "swiftc",
            str(source),
            "-framework",
            "Metal",
            "-O",
            "-o",
            str(binary),
        )
    )


def _capture_one(
    *,
    runner: SubprocessCommandRunner,
    launcher: SubprocessProcessLauncher,
    private_dir: Path,
    binary: Path,
    dispatch_count: int,
    time_limit: str,
) -> CaptureSummary:
    trace_path = private_dir / f"dispatch-{dispatch_count}.trace"
    record_dir = private_dir / f"record-{dispatch_count}"
    import_dir = private_dir / f"import-{dispatch_count}"
    recorded = record_trace(
        runner=runner,
        launcher=launcher,
        command=(str(binary), str(dispatch_count)),
        output_trace=trace_path,
        output_dir=record_dir,
        time_limit=time_limit,
        table_schema=None,
    )
    if not recorded.succeeded:
        raise RuntimeError(f"xctrace recording failed: {recorded.message}")

    imported = import_trace(
        runner=runner,
        trace_path=trace_path,
        output_dir=import_dir,
        capability=recorded.capability,
        table_schema=TABLE_SCHEMA,
    )
    if imported.table is None:
        raise RuntimeError(f"xctrace import produced no {TABLE_SCHEMA!r} table")
    return summarize_exports(
        read_export_text(import_dir / "trace_toc.xml"),
        read_export_text(import_dir / "trace_table.xml"),
        expected_dispatch_count=dispatch_count,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, captures: Sequence[CaptureSummary]) -> None:
    fieldnames = list(asdict(captures[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(capture) for capture in captures)


def _svg_chart(
    *,
    title: str,
    subtitle: str,
    labels: Sequence[str],
    series: Sequence[tuple[str, str, Sequence[float]]],
    y_label: str,
) -> str:
    width, height = 960, 540
    left, top, right, bottom = 92, 112, 34, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(max(values) for _, _, values in series)
    maximum = max(maximum, 1.0)
    group_width = plot_width / len(labels)
    bar_width = min(54.0, group_width / (len(series) + 1))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{title}</title>',
        f'<desc id="desc">{subtitle}</desc>',
        '<rect width="960" height="540" fill="#f4f1ea"/>',
        '<rect x="20" y="20" width="920" height="500" fill="#fbfaf7" stroke="#d2cec5"/>',
        f'<text x="{left}" y="56" fill="#16181a" font-family="-apple-system, BlinkMacSystemFont, sans-serif" font-size="23" font-weight="600">{title}</text>',
        f'<text x="{left}" y="82" fill="#5b6167" font-family="ui-monospace, monospace" font-size="13">{subtitle}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#16181a"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#16181a"/>',
        f'<text x="25" y="{top + plot_height / 2}" transform="rotate(-90 25 {top + plot_height / 2})" fill="#5b6167" font-family="ui-monospace, monospace" font-size="12">{y_label}</text>',
    ]
    for index, label in enumerate(labels):
        center = left + group_width * (index + 0.5)
        parts.append(
            f'<text x="{center:.1f}" y="{top + plot_height + 28}" text-anchor="middle" fill="#16181a" font-family="ui-monospace, monospace" font-size="13">{label}</text>'
        )
        for series_index, (_, color, values) in enumerate(series):
            value = values[index]
            bar_height = value / maximum * (plot_height - 22)
            x = center - (len(series) * bar_width) / 2 + series_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 5:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + (bar_width - 5) / 2:.1f}" y="{max(top + 13, y - 7):.1f}" text-anchor="middle" fill="#16181a" font-family="ui-monospace, monospace" font-size="12">{value:g}</text>'
            )
    legend_x = left
    for name, color, _ in series:
        parts.extend(
            (
                f'<rect x="{legend_x}" y="94" width="10" height="10" fill="{color}"/>',
                f'<text x="{legend_x + 16}" y="103" fill="#5b6167" font-family="ui-monospace, monospace" font-size="11">{name}</text>',
            )
        )
        legend_x += 190
    parts.append("</svg>\n")
    return "".join(parts)


def generate_charts(public_dir: Path, captures: Sequence[CaptureSummary]) -> None:
    labels = [str(capture.expected_dispatch_count) for capture in captures]
    (public_dir / "dispatch-attribution.svg").write_text(
        _svg_chart(
            title="Known dispatches match PID-attributed intervals",
            subtitle="Measured counts; equality is verified for every capture.",
            labels=labels,
            series=(
                (
                    "expected dispatches",
                    "#16181a",
                    [capture.expected_dispatch_count for capture in captures],
                ),
                (
                    "PID-attributed intervals",
                    "#c23d16",
                    [capture.attributed_interval_count for capture in captures],
                ),
            ),
            y_label="count",
        ),
        encoding="utf-8",
    )
    (public_dir / "unrelated-interval-share.svg").write_text(
        _svg_chart(
            title="Trace-wide interval share includes unrelated GPU work",
            subtitle="Derived shares of measured counts; each capture totals 100%.",
            labels=labels,
            series=(
                (
                    "target PID share",
                    "#c23d16",
                    [
                        round(
                            100.0
                            - capture.known_unrelated_interval_share_percent
                            - capture.unattributed_interval_share_percent,
                            1,
                        )
                        for capture in captures
                    ],
                ),
                (
                    "known unrelated share",
                    "#4a5157",
                    [
                        capture.known_unrelated_interval_share_percent
                        for capture in captures
                    ],
                ),
                (
                    "unattributed share",
                    "#6f6230",
                    [
                        capture.unattributed_interval_share_percent
                        for capture in captures
                    ],
                ),
            ),
            y_label="percent of trace intervals",
        ),
        encoding="utf-8",
    )


def _verify_public_contents(public_dir: Path) -> None:
    entries = tuple(public_dir.iterdir())
    unexpected = sorted(
        entry.name for entry in entries if entry.name not in PUBLIC_FILES
    )
    if unexpected:
        raise ValueError(
            f"public bundle contains unexpected entries: {', '.join(unexpected)}"
        )
    symlinks = sorted(entry.name for entry in entries if entry.is_symlink())
    if symlinks:
        raise ValueError(f"public bundle contains symlinks: {', '.join(symlinks)}")
    missing = [
        name for name in PUBLIC_CONTENT_FILES if not (public_dir / name).is_file()
    ]
    if missing:
        raise ValueError(f"public bundle is missing files: {', '.join(missing)}")

    summary = json.loads((public_dir / "capture-summary.json").read_text())
    manifest = json.loads((public_dir / "experiment-manifest.json").read_text())
    source_name = manifest.get("workload", {}).get("source")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise ValueError("manifest workload source must be a local basename")
    source_path = Path(__file__).with_name(source_name)
    if manifest["workload"].get("source_sha256") != _sha256(source_path):
        raise ValueError("manifest workload source hash does not match")
    captures = summary["captures"]
    provenance = summary.get("provenance", {})
    if not str(provenance.get("attributed_interval_count", "")).startswith(
        "measured_native"
    ):
        raise ValueError("attributed interval provenance is not measured_native")
    if not str(provenance.get("unattributed_interval_count", "")).startswith(
        "measured_native"
    ):
        raise ValueError("unattributed interval provenance is not measured_native")
    for name in (
        "known_unrelated_interval_count",
        "known_unrelated_interval_share_percent",
        "unattributed_interval_share_percent",
        "window_server_interval_share_percent",
    ):
        if not str(provenance.get(name, "")).startswith("derived:"):
            raise ValueError(f"{name} provenance is not derived")
    if not captures or not all(item["dispatch_count_matches"] for item in captures):
        raise ValueError(
            "not every dispatch count matches its PID-attributed interval count"
        )
    for item in captures:
        if (
            item["attributed_interval_count"]
            + item["known_unrelated_interval_count"]
            + item["unattributed_interval_count"]
            != item["all_process_interval_count"]
        ):
            raise ValueError(
                "target, known unrelated, and unattributed counts do not "
                "reconstruct trace total"
            )
        if item["reference_cells"] > item["exported_cells"]:
            raise ValueError("reference cell count exceeds exported cell count")

    for name in PUBLIC_CONTENT_FILES:
        text = (public_dir / name).read_text(encoding="utf-8")
        for pattern, description in FORBIDDEN_PUBLIC_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"{name} contains forbidden {description}")


def verify_public_bundle(public_dir: Path) -> None:
    _verify_public_contents(public_dir)
    sums_path = public_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise ValueError("public bundle is missing files: SHA256SUMS")
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        expected[name] = digest
    actual = {name: _sha256(public_dir / name) for name in PUBLIC_CONTENT_FILES}
    if expected != actual:
        raise ValueError("SHA256SUMS does not match the public evidence files")


def _summary_document(captures: Sequence[CaptureSummary]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "claim": (
            "Trace-wide Metal interval totals include unrelated processes; "
            "workload claims require target-PID attribution."
        ),
        "rounding": "shares are rounded to one decimal place",
        "provenance": {
            "expected_dispatch_count": "controlled workload input",
            "attributed_interval_count": "measured_native, grouped by TOC target PID",
            "all_process_interval_count": "measured_native",
            "known_unrelated_interval_count": (
                "derived: all_process_interval_count - attributed_interval_count "
                "- unattributed_interval_count"
            ),
            "known_unrelated_interval_share_percent": (
                "derived: known_unrelated_interval_count / all_process_interval_count"
            ),
            "unattributed_interval_count": (
                "measured_native rows with no parseable process PID"
            ),
            "unattributed_interval_share_percent": (
                "derived: unattributed_interval_count / all_process_interval_count"
            ),
            "window_server_interval_count": (
                "measured_native, grouped by the standard macOS service label"
            ),
            "window_server_interval_share_percent": (
                "derived: window_server_interval_count / all_process_interval_count"
            ),
            "schema_count": "measured_native from sanitized TOC",
            "exported_rows": "measured_native table shape",
            "exported_columns": "measured_native table shape",
            "exported_cells": "derived: exported_rows * exported_columns",
            "reference_cells": "measured_native XML row-cell attributes",
            "dispatch_count_matches": (
                "derived comparison: expected_dispatch_count == "
                "attributed_interval_count"
            ),
        },
        "captures": [asdict(capture) for capture in captures],
    }


def _write_public_bundle(
    public_dir: Path,
    captures: Sequence[CaptureSummary],
    *,
    source: Path,
    time_limit: str,
) -> None:
    host = _safe_host_metadata()
    capture_command = shlex.join(
        (
            "uv",
            "run",
            "python",
            "examples/metal_evidence/evidence_demo.py",
            "capture",
            "--output-dir",
            "<OUTPUT_DIR>",
            "--dispatches",
            *(str(capture.expected_dispatch_count) for capture in captures),
            "--time-limit",
            time_limit,
        )
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "environment": host,
        "collection": {
            "template": METAL_SYSTEM_TRACE_TEMPLATE,
            "table_schema": TABLE_SCHEMA,
            "time_limit": time_limit,
            "capture_boundary": (
                "xctrace launched one local workload process; each trace ended when "
                "that process exited, with the time limit as a hard upper bound"
            ),
            "raw_artifacts_committed": False,
        },
        "workload": {
            "name": WORKLOAD_NAME,
            "source": "metal_workload.swift",
            "source_sha256": _sha256(source),
            "dispatch_counts": [
                capture.expected_dispatch_count for capture in captures
            ],
            "threads_per_grid": 262144,
            "command_buffers_per_dispatch": 1,
        },
        "commands": [
            "uv run python examples/metal_evidence/evidence_demo.py capability",
            capture_command,
            "uv run python examples/metal_evidence/evidence_demo.py verify --public-dir <OUTPUT_DIR>/public",
        ],
        "approved_metrics": {
            "measured_native": [
                "metal_gpu_interval_count",
                "metal_gpu_interval_count_all_processes",
                "unattributed_interval_count",
                "window_server_interval_count",
            ],
            "derived": [
                "known_unrelated_interval_count",
                "known_unrelated_interval_share_percent",
                "unattributed_interval_share_percent",
                "window_server_interval_share_percent",
            ],
        },
        "unsupported_metrics": [
            "GPU utilization",
            "GPU busy percentage",
            "kernel time",
            "memory bandwidth",
            "occupancy",
            "GPU power",
            "GPU energy",
            "GPU memory footprint",
        ],
    }
    summary = _summary_document(captures)
    _write_json(public_dir / "experiment-manifest.json", manifest)
    _write_json(public_dir / "capture-summary.json", summary)
    _write_csv(public_dir / "capture-summary.csv", captures)
    generate_charts(public_dir, captures)
    _verify_public_contents(public_dir)
    sums = "".join(
        f"{_sha256(public_dir / name)}  {name}\n"
        for name in sorted(PUBLIC_CONTENT_FILES)
    )
    (public_dir / "SHA256SUMS").write_text(sums, encoding="utf-8")
    verify_public_bundle(public_dir)


def capture(
    output_dir: Path,
    *,
    dispatch_counts: Sequence[int],
    time_limit: str,
    retain_private: bool,
) -> Path:
    runner = SubprocessCommandRunner()
    capability = detect_xctrace_capability(
        runner=runner, template=METAL_SYSTEM_TRACE_TEMPLATE
    )
    if not capability.supported:
        raise RuntimeError(
            f"unsupported host ({capability.capability.value}): {capability.reason}"
        )
    print(f"capability={capability.capability.value}")
    print(f"template={METAL_SYSTEM_TRACE_TEMPLATE}")
    private_dir, public_dir = _prepare_output(output_dir)
    source = Path(__file__).with_name("metal_workload.swift")
    binary = private_dir / WORKLOAD_NAME
    try:
        _compile_workload(source, binary)
        launcher = SubprocessProcessLauncher()
        captures = []
        for count in dispatch_counts:
            capture_summary = _capture_one(
                runner=runner,
                launcher=launcher,
                private_dir=private_dir,
                binary=binary,
                dispatch_count=count,
                time_limit=time_limit,
            )
            captures.append(capture_summary)
            print(f"capture_import dispatches={count} status=completed")
        _write_public_bundle(public_dir, captures, source=source, time_limit=time_limit)
    finally:
        if not retain_private:
            _remove_private_artifacts(private_dir)
    return public_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capability", help="check the local xctrace prerequisites")
    capture_parser = commands.add_parser(
        "capture", help="record, import, summarize, chart, and verify fresh traces"
    )
    capture_parser.add_argument("--output-dir", type=Path, required=True)
    capture_parser.add_argument(
        "--dispatches", type=int, nargs="+", default=list(DEFAULT_DISPATCH_COUNTS)
    )
    capture_parser.add_argument("--time-limit", default=DEFAULT_TIME_LIMIT)
    capture_parser.add_argument(
        "--retain-private",
        action="store_true",
        help="retain raw traces/XML locally; never commit that directory",
    )
    verify_parser = commands.add_parser(
        "verify", help="verify hashes, privacy rules, and count invariants"
    )
    verify_parser.add_argument("--public-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capability":
        report = detect_xctrace_capability(
            runner=SubprocessCommandRunner(), template=METAL_SYSTEM_TRACE_TEMPLATE
        )
        print(f"capability={report.capability.value}")
        print(f"template={METAL_SYSTEM_TRACE_TEMPLATE}")
        print(f"xctrace={report.xctrace_version or 'unknown'}")
        return 0 if report.supported else 3
    if args.command == "verify":
        verify_public_bundle(args.public_dir)
        print("verification=passed")
        return 0
    if any(count <= 0 or count > 10_000 for count in args.dispatches):
        raise ValueError("dispatch counts must be between 1 and 10000")
    public_dir = capture(
        args.output_dir,
        dispatch_counts=tuple(args.dispatches),
        time_limit=args.time_limit,
        retain_private=args.retain_private,
    )
    summary = json.loads((public_dir / "capture-summary.json").read_text())
    print("capture_import=completed")
    for item in summary["captures"]:
        print(
            "dispatches={expected_dispatch_count} attributed={attributed_interval_count} "
            "all_processes={all_process_interval_count} "
            "known_unrelated={known_unrelated_interval_count} "
            "unattributed={unattributed_interval_count} "
            "match={dispatch_count_matches}".format(**item)
        )
    print("verification=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
