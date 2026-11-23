"""Import benchmark results saved by other tools.

``import_benchmark_file`` reads a ``vllm bench serve --save-result`` JSON file
or an AIPerf ``profile_export_aiperf.json`` file and returns an
:class:`ImportedBenchmark`. ``import_aiperf_records_file`` reads AIPerf's
per-request ``profile_export.jsonl`` with its summary and computes the
statistics itself. The perf gate accepts the written artifact. Only the
standard library is imported here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_bytes,
)
from .aiperf import import_aiperf, looks_like_aiperf_profile
from .aiperf_records import import_aiperf_records, looks_like_aiperf_records
from .artifact import (
    AIPERF_RECORDS_TOOL,
    AIPERF_TOOL,
    DERIVED,
    IMPORTED_BENCHMARK_KIND,
    IMPORTED_DERIVED,
    IMPORTED_REPORTED,
    MAX_IMPORTED_ARTIFACT_BYTES,
    METRIC_DIRECTIONS,
    SUPPORTED_TOOLS,
    VLLM_BENCH_TOOL,
    BenchmarkImportError,
    ImportedArtifactError,
    ImportedBenchmark,
    ImportedInterval,
    ImportedMetric,
    ImportedSource,
    ImportedSpeculative,
    ImportedTotal,
    ImportedValue,
    is_imported_benchmark,
    parse_json_document,
    read_imported_benchmark,
)
from .vllm_bench import import_vllm_bench, looks_like_vllm_bench

__all__ = [
    "AIPERF_RECORDS_TOOL",
    "AIPERF_TOOL",
    "DERIVED",
    "IMPORTED_BENCHMARK_KIND",
    "IMPORTED_DERIVED",
    "IMPORTED_REPORTED",
    "MAX_IMPORTED_ARTIFACT_BYTES",
    "METRIC_DIRECTIONS",
    "SUPPORTED_TOOLS",
    "VLLM_BENCH_TOOL",
    "BenchmarkImportError",
    "ImportedArtifactError",
    "ImportedBenchmark",
    "ImportedInterval",
    "ImportedMetric",
    "ImportedSource",
    "ImportedSpeculative",
    "ImportedTotal",
    "ImportedValue",
    "import_aiperf_records_file",
    "import_benchmark_file",
    "importer_version",
    "is_imported_benchmark",
    "read_imported_benchmark",
]

_TOOL_LABELS = {
    VLLM_BENCH_TOOL: "a `vllm bench serve --save-result` file",
    AIPERF_TOOL: "an AIPerf profile_export_aiperf.json file",
}


def importer_version() -> str | None:
    try:
        return version("llmtracefx")
    except PackageNotFoundError:
        return None  # Running from a source tree, as the GitHub Action does.


def _detected_tool(data: object) -> str | None:
    if is_imported_benchmark(data):
        return "llmtracefx"
    if looks_like_vllm_bench(data):
        return VLLM_BENCH_TOOL
    if looks_like_aiperf_profile(data):
        return AIPERF_TOOL
    return None


def _read_source(path: Path) -> tuple[bytes, str]:
    try:
        raw = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
    except (OSError, ArtifactReadError) as exc:
        raise BenchmarkImportError(f"{path}: {exc}") from exc
    try:
        return raw, raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkImportError(f"{path} is not UTF-8 text: {exc}") from exc


def _records_hint(path: Path) -> str:
    return (
        f"{path} holds AIPerf per-request records (profile_export.jsonl); pass "
        f"it with `import aiperf <profile_export_aiperf.json> --records {path}`"
    )


def import_benchmark_file(path: str | Path, tool: str) -> ImportedBenchmark:
    """Read one source file written by ``tool`` and map it.

    Raises :class:`BenchmarkImportError` when the file is unreadable,
    malformed, from another tool, of an unsupported schema, or mixes runs.
    """
    if tool not in (VLLM_BENCH_TOOL, AIPERF_TOOL):
        raise BenchmarkImportError(f"unsupported tool {tool!r}")
    path = Path(path)
    raw, text = _read_source(path)
    if looks_like_aiperf_records(text):
        raise BenchmarkImportError(_records_hint(path))
    data = parse_json_document(text, str(path))
    detected = _detected_tool(data)
    if detected == "llmtracefx":
        raise BenchmarkImportError(
            f"{path} is already an LLMTraceFX imported benchmark artifact"
        )
    if detected is not None and detected != tool:
        raise BenchmarkImportError(
            f"{path} looks like {_TOOL_LABELS[detected]}, not "
            f"{_TOOL_LABELS[tool]}; use `import {detected_command(detected)}`"
        )
    importer = import_vllm_bench if tool == VLLM_BENCH_TOOL else import_aiperf
    try:
        return importer(
            data,
            sha256=hashlib.sha256(raw).hexdigest(),
            filename=path.name,
            importer_version=importer_version(),
        )
    except BenchmarkImportError as exc:
        raise BenchmarkImportError(f"{path}: {exc}") from exc


def import_aiperf_records_file(
    summary_path: str | Path,
    records_path: str | Path,
    *,
    slo: Mapping[str, float] | None = None,
) -> ImportedBenchmark:
    """Read AIPerf's ``profile_export.jsonl`` with the summary from the same run.

    The summary supplies the model, endpoint, load and AIPerf version and is
    checked as ``import_benchmark_file`` checks it. Raises
    :class:`BenchmarkImportError` when either file is refused or the two
    files disagree on request counts or token totals.
    """
    summary_path, records_path = Path(summary_path), Path(records_path)
    if summary_path.resolve() == records_path.resolve():
        raise BenchmarkImportError(
            "--records must be a different file from the summary"
        )
    summary = import_benchmark_file(summary_path, AIPERF_TOOL)
    raw, text = _read_source(records_path)
    if not looks_like_aiperf_records(text):
        try:
            data = parse_json_document(text, str(records_path))
        except BenchmarkImportError:
            data = None
        if looks_like_aiperf_profile(data):
            raise BenchmarkImportError(
                f"{records_path} is an AIPerf summary (profile_export_aiperf.json), "
                "not the per-request profile_export.jsonl"
            )
    try:
        return import_aiperf_records(
            text,
            summary=summary,
            sha256=hashlib.sha256(raw).hexdigest(),
            filename=records_path.name,
            slo=slo,
            importer_version=importer_version(),
        )
    except BenchmarkImportError as exc:
        raise BenchmarkImportError(f"{records_path}: {exc}") from exc


def detected_command(tool: str) -> str:
    return "vllm-bench" if tool == VLLM_BENCH_TOOL else "aiperf"
