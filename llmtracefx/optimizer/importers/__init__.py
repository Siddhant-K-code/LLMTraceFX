"""Import benchmark results saved by other tools.

``import_benchmark_file`` reads a ``vllm bench serve --save-result`` JSON file
or an AIPerf ``profile_export_aiperf.json`` file and returns an
:class:`ImportedBenchmark`. The perf gate accepts the written artifact. Only
the standard library is imported here.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_bytes,
)
from .aiperf import import_aiperf, looks_like_aiperf_profile
from .artifact import (
    AIPERF_TOOL,
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
    ImportedMetric,
    ImportedSource,
    ImportedTotal,
    ImportedValue,
    is_imported_benchmark,
    parse_json_document,
    read_imported_benchmark,
)
from .vllm_bench import import_vllm_bench, looks_like_vllm_bench

__all__ = [
    "AIPERF_TOOL",
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
    "ImportedMetric",
    "ImportedSource",
    "ImportedTotal",
    "ImportedValue",
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


def import_benchmark_file(path: str | Path, tool: str) -> ImportedBenchmark:
    """Read one source file written by ``tool`` and map it.

    Raises :class:`BenchmarkImportError` when the file is unreadable,
    malformed, from another tool, of an unsupported schema, or mixes runs.
    """
    if tool not in SUPPORTED_TOOLS:
        raise BenchmarkImportError(f"unsupported tool {tool!r}")
    path = Path(path)
    try:
        raw = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
    except (OSError, ArtifactReadError) as exc:
        raise BenchmarkImportError(f"{path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BenchmarkImportError(f"{path} is not UTF-8 text: {exc}") from exc
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


def detected_command(tool: str) -> str:
    return "vllm-bench" if tool == VLLM_BENCH_TOOL else "aiperf"
