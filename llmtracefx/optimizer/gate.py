"""Performance regression gate for CI.

Compares candidate runs against committed baseline runs of the same
workload on the same platform and decides ``pass``, ``regression`` or
``inconclusive``. It refuses to call a direction when the groups are not
comparable, when a group has too few repetitions, when the metric was
measured with different provenance or units, when the delta is smaller
than the run-to-run noise, or, before passing, when the combined noise is
larger than the allowed regression. That keeps a noisy CI runner from failing, or
passing, a pull request by chance.

The runtime version and configuration hash are allowed to differ between
the groups, because they are usually what the pull request changed.

The gate also accepts imported benchmark artifacts written by
``llmtracefx-optimizer import``. Their values are summary statistics that
vLLM bench or AIPerf computed, so the metric names one statistic
(``ttft_ms.p99``), each imported file counts as one run, and imported
artifacts are never mixed with records or with another tool's imports.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
)
from .collectors._shared import atomic_write_text
from .importers.aiperf import looks_like_aiperf_profile
from .importers.artifact import (
    HIGHER_IS_BETTER,
    IMPORTED_BENCHMARK_KIND,
    MAX_IMPORTED_ARTIFACT_BYTES,
    METRIC_DIRECTIONS,
    BenchmarkImportError,
    ImportedArtifactError,
    ImportedBenchmark,
    is_imported_benchmark,
    parse_json_document,
)
from .importers.vllm_bench import looks_like_vllm_bench
from .load.artifact import find_sweep_artifact, sweep_refusal
from .schema import ExperimentRecord, Measurement, SchemaValidationError

GATE_REPORT_KIND = "llmtracefx.gate.report"
GATE_REPORT_SCHEMA_VERSION = "1"
_PHASE_METRICS = ("total", "model_load", "tokenize", "prefill", "decode")
DECODE_TPS = "decode_tokens_per_second"
METRIC_CHOICES = (*_PHASE_METRICS, DECODE_TPS)
RECORDS_INPUT = "records"
IMPORTED_INPUT = "imported_benchmark"

GateInput = ExperimentRecord | ImportedBenchmark


class GateVerdict(StrEnum):
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


class GateInputError(ValueError):
    """An input could not be read, was invalid, or the inputs cannot be mixed."""


@dataclass(frozen=True)
class _Sample:
    run_id: str
    value: float
    unit: str
    provenance: str


@dataclass(frozen=True)
class GateReport:
    verdict: GateVerdict
    reason: str
    metric: str
    higher_is_better: bool
    max_regression: float
    min_repetitions: int
    baseline_run_ids: tuple[str, ...] = ()
    candidate_run_ids: tuple[str, ...] = ()
    unit: str | None = None
    provenance: str | None = None
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    baseline_stdev: float | None = None
    candidate_stdev: float | None = None
    delta: float | None = None
    delta_pct: float | None = None
    """Signed relative change; positive always means worse."""
    excluded: tuple[str, ...] = ()
    input_kind: str = RECORDS_INPUT
    source_tool: str | None = None
    sources: tuple[dict[str, Any], ...] = ()

    @property
    def exit_code(self) -> int:
        return {
            GateVerdict.PASS: 0,
            GateVerdict.REGRESSION: 1,
            GateVerdict.INCONCLUSIVE: 2,
        }[self.verdict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": GATE_REPORT_KIND,
            "schema_version": GATE_REPORT_SCHEMA_VERSION,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "max_regression": self.max_regression,
            "min_repetitions": self.min_repetitions,
            "unit": self.unit,
            "provenance": self.provenance,
            "baseline": {
                "run_ids": list(self.baseline_run_ids),
                "mean": self.baseline_mean,
                "stdev": self.baseline_stdev,
            },
            "candidate": {
                "run_ids": list(self.candidate_run_ids),
                "mean": self.candidate_mean,
                "stdev": self.candidate_stdev,
            },
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "excluded": list(self.excluded),
            "input_kind": self.input_kind,
            "source_tool": self.source_tool,
            "sources": [dict(source) for source in self.sources],
        }

    def write_json(self, path: Path) -> None:
        atomic_write_text(
            path, json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        )

    def to_markdown(self) -> str:
        icon = {
            GateVerdict.PASS: "✅",
            GateVerdict.REGRESSION: "❌",
            GateVerdict.INCONCLUSIVE: "⚠️",
        }[self.verdict]
        lines = [
            f"### {icon} LLMTraceFX perf gate: {self.verdict.value}",
            "",
            self.reason,
            "",
        ]
        if self.baseline_mean is not None and self.candidate_mean is not None:
            unit = self.unit or ""
            lines += [
                "| | runs | mean | stdev |",
                "|---|---:|---:|---:|",
                f"| baseline | {len(self.baseline_run_ids)} | "
                f"{self.baseline_mean:.6g} {unit} | {self.baseline_stdev or 0:.3g} |",
                f"| candidate | {len(self.candidate_run_ids)} | "
                f"{self.candidate_mean:.6g} {unit} | {self.candidate_stdev or 0:.3g} |",
                "",
            ]
        direction = "higher" if self.higher_is_better else "lower"
        lines.append(
            f"Metric `{self.metric}` ({direction} is better, provenance "
            f"`{self.provenance or 'n/a'}`); allowed regression "
            f"{self.max_regression:.1%}; at least {self.min_repetitions} runs "
            "per side."
        )
        if self.input_kind == IMPORTED_INPUT:
            lines += [
                "",
                f"Values were computed by `{self.source_tool}` and imported; "
                "LLMTraceFX did not measure them. Each imported file counts "
                "as one run.",
            ]
        if self.excluded:
            lines += ["", "Excluded runs:", ""]
            lines += [f"- {item}" for item in self.excluded]
        return "\n".join(lines) + "\n"


def load_records(paths: Sequence[str | Path]) -> list[ExperimentRecord]:
    """Read records from files, expanding directories to ``**/record.json``."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.rglob("record.json"))
            if not found:
                raise GateInputError(f"no record.json found under {path}")
            files.extend(found)
        else:
            files.append(path)
    records = []
    for file in files:
        try:
            records.append(ExperimentRecord.read_json(file))
        except (OSError, SchemaValidationError) as exc:
            raise GateInputError(f"{file}: {exc}") from exc
    return records


def _imported_artifact(file: Path, text: str) -> ImportedBenchmark:
    if len(text.encode("utf-8")) > MAX_IMPORTED_ARTIFACT_BYTES:
        raise GateInputError(f"{file}: imported benchmark artifact is too large")
    try:
        return ImportedBenchmark.from_dict(parse_json_document(text, str(file)))
    except (BenchmarkImportError, ImportedArtifactError) as exc:
        raise GateInputError(f"{file}: {exc}") from exc


def _read_input(file: Path) -> GateInput:
    try:
        text = read_bounded_regular_text(file, MAX_EVIDENCE_ARTIFACT_BYTES)
    except (OSError, ArtifactReadError) as exc:
        raise GateInputError(f"{file}: {exc}") from exc
    try:
        data: Any = parse_json_document(text, str(file))
    except BenchmarkImportError:
        data = None  # Let the record parser report the error.
    if IMPORTED_BENCHMARK_KIND in text and is_imported_benchmark(data):
        return _imported_artifact(file, text)
    for looks_like, command in (
        (looks_like_vllm_bench, "vllm-bench"),
        (looks_like_aiperf_profile, "aiperf"),
    ):
        if looks_like(data):
            raise GateInputError(
                f"{file} is a raw benchmark result; convert it first with "
                f"`llmtracefx-optimizer import {command} {file} --output ...`"
            )
    try:
        return ExperimentRecord.from_json(text)
    except SchemaValidationError as exc:
        raise GateInputError(
            f"{file}: {exc} (raw vllm bench or AIPerf results must be converted "
            "with `llmtracefx-optimizer import` first)"
        ) from exc


def _imported_in(directory: Path) -> list[Path]:
    found = []
    for file in sorted(directory.rglob("*.json")):
        if file.name == "record.json" or not file.is_file() or file.is_symlink():
            continue
        try:
            if file.stat().st_size > MAX_IMPORTED_ARTIFACT_BYTES:
                continue
            text = read_bounded_regular_text(file, MAX_IMPORTED_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError):
            continue
        if IMPORTED_BENCHMARK_KIND not in text:
            continue
        try:
            data = parse_json_document(text, str(file))
        except BenchmarkImportError:
            continue
        if is_imported_benchmark(data):
            found.append(file)
    return found


def load_gate_inputs(paths: Sequence[str | Path]) -> list[GateInput]:
    """Read records and imported benchmark artifacts.

    Files are read as given. A directory expands to every ``record.json`` and
    every other ``*.json`` holding an imported benchmark artifact below it;
    other JSON files there are skipped. A load sweep artifact, given directly
    or found below a directory, is refused rather than skipped.
    """
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        sweep = find_sweep_artifact(path)
        if sweep is not None:
            raise GateInputError(sweep_refusal(sweep))
        if path.is_dir():
            found = sorted({*path.rglob("record.json"), *_imported_in(path)})
            if not found:
                raise GateInputError(
                    f"no record.json or imported benchmark artifact found under "
                    f"{path}"
                )
            files.extend(found)
        else:
            files.append(path)
    return [_read_input(file) for file in files]


def _comparability(record: ExperimentRecord, metric: str) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        record.model.model_id,
        record.model.model_revision,
        record.model.quantization,
        record.runtime.name,
        record.runtime.backend,
        record.platform.os_name,
        record.platform.architecture,
        record.platform.accelerator,
        record.command.workload_hash,
        record.tokens.input_tokens,
    )
    if metric != DECODE_TPS:
        # Latency of a longer generation is not comparable; a rate is.
        key += (record.tokens.generated_tokens,)
    return key


def _sample(record: ExperimentRecord, metric: str) -> _Sample | str:
    """The metric value, or the reason this record cannot supply one."""
    if not record.outcome.success or record.error is not None:
        return "run did not succeed"
    if metric == DECODE_TPS:
        decode = record.timing.decode
        tokens = record.tokens.generated_tokens
        if decode is None or tokens is None:
            return "no decode time or generated token count"
        if decode.unit != "ms" or not math.isfinite(decode.value) or decode.value <= 0:
            return "decode time is not a positive duration in ms"
        return _Sample(
            record.run_id,
            tokens / (decode.value / 1000.0),
            "tokens/s",
            f"derived_from_{decode.provenance.value}",
        )
    measurement: Measurement | None = getattr(record.timing, metric)
    if measurement is None:
        return f"no timing.{metric}"
    if not math.isfinite(measurement.value) or measurement.value < 0:
        return f"timing.{metric} is not a finite, non-negative value"
    return _Sample(
        record.run_id,
        measurement.value,
        measurement.unit,
        measurement.provenance.value,
    )


@dataclass(frozen=True)
class _Adapter:
    higher_is_better: bool
    sample: Callable[[Any], _Sample | str]
    key: Callable[[Any], tuple[Any, ...]]
    run_id: Callable[[Any], str]
    differs: str
    source_tool: str | None = None


def _record_adapter(metric: str) -> _Adapter:
    if metric not in METRIC_CHOICES:
        raise ValueError(f"metric must be one of {', '.join(METRIC_CHOICES)}")
    return _Adapter(
        higher_is_better=metric == DECODE_TPS,
        sample=lambda record: _sample(record, metric),
        key=lambda record: _comparability(record, metric),
        run_id=lambda record: record.run_id,
        differs="model, runtime backend, platform, workload or token counts differ",
    )


_EXAMPLE_METRIC = {
    "vllm-bench-serve": "ttft_ms.p99 or request_throughput.value",
    "aiperf": "time_to_first_token.p99 or request_throughput.avg",
}


def _imported_metric(tool: str, metric: str) -> tuple[str, str]:
    choices = sorted(METRIC_DIRECTIONS[tool])
    # Metric names have no dots; statistics can (vLLM writes p99.9).
    name, dot, stat = metric.partition(".")
    if not dot or not stat or name not in METRIC_DIRECTIONS[tool]:
        raise ValueError(
            f"for {tool} imports the metric is <metric>.<statistic>, for example "
            f"{_EXAMPLE_METRIC[tool]}; metrics: {', '.join(choices)}"
        )
    return name, stat


def _imported_sample(item: ImportedBenchmark, name: str, stat: str) -> _Sample | str:
    failed = item.totals.get("failed_requests")
    if failed is None:
        return "the source does not state how many requests failed"
    if failed.value > 0:
        return (
            f"{failed.value:g} request(s) failed; the source tool summarises "
            "successful requests only, so the run is not used"
        )
    if item.load_unresolved:
        return "load settings unresolved: " + ", ".join(item.load_unresolved)
    metric = item.metrics.get(name)
    if metric is None:
        return f"no {name} in the source file"
    value = metric.stats.get(stat)
    if value is None:
        return f"no {name}.{stat} in the source file (has {', '.join(metric.stats)})"
    if not math.isfinite(value.value) or value.value < 0:
        return f"{name}.{stat} is not a finite, non-negative value"
    return _Sample(item.run_id, value.value, metric.unit, metric.provenance)


def _imported_key(item: ImportedBenchmark, higher_is_better: bool) -> tuple[Any, ...]:
    def total(name: str) -> float | None:
        found = item.totals.get(name)
        return None if found is None else found.value

    key: tuple[Any, ...] = (
        item.source.tool,
        item.source.tool_version,
        item.source.schema_version,
        item.model_id,
        tuple(sorted(item.endpoint.items())),
        tuple(sorted(item.load.items())),
        total("total_input_tokens"),
    )
    if not higher_is_better:
        key += (total("total_output_tokens"),)
    return key


def _imported_adapter(items: Sequence[ImportedBenchmark], metric: str) -> _Adapter:
    tools = sorted({item.source.tool for item in items})
    if len(tools) != 1:
        raise GateInputError(
            f"imported artifacts come from different tools ({', '.join(tools)}); "
            "their metrics are defined differently and are not compared"
        )
    tool = tools[0]
    hashes = [item.source.sha256 for item in items]
    repeated = sorted({sha for sha in hashes if hashes.count(sha) > 1})
    if repeated:
        raise GateInputError(
            f"the same source file (sha256 {repeated[0][:12]}) is imported more "
            "than once; each run must be a distinct file"
        )
    name, stat = _imported_metric(tool, metric)
    higher = METRIC_DIRECTIONS[tool][name] == HIGHER_IS_BETTER
    return _Adapter(
        higher_is_better=higher,
        sample=lambda item: _imported_sample(item, name, stat),
        key=lambda item: _imported_key(item, higher),
        run_id=lambda item: item.run_id,
        differs=("tool version, model, endpoint, load settings or token totals differ"),
        source_tool=tool,
    )


def _source_summary(item: ImportedBenchmark) -> dict[str, Any]:
    return {
        "run_id": item.run_id,
        "filename": item.source.filename,
        "sha256": item.source.sha256,
        "tool_version": item.source.tool_version,
        "schema_version": item.source.schema_version,
    }


def evaluate_gate(
    baseline: Sequence[GateInput],
    candidate: Sequence[GateInput],
    *,
    metric: str = "total",
    max_regression: float = 0.05,
    min_repetitions: int = 3,
) -> GateReport:
    """Compare two groups of records, or two groups of imported artifacts."""
    if not 0 <= max_regression < 1:
        raise ValueError("max_regression must be in [0, 1)")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least 2 to estimate noise")
    everything = [*baseline, *candidate]
    imported = [item for item in everything if isinstance(item, ImportedBenchmark)]
    if imported and len(imported) != len(everything):
        raise GateInputError(
            "records and imported benchmark artifacts cannot be mixed; LLMTraceFX "
            "measurements and statistics computed by another tool are not compared"
        )
    adapter = (
        _imported_adapter(imported, metric) if imported else _record_adapter(metric)
    )
    higher_is_better = adapter.higher_is_better
    base: dict[str, Any] = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "max_regression": max_regression,
        "min_repetitions": min_repetitions,
        "input_kind": IMPORTED_INPUT if imported else RECORDS_INPUT,
        "source_tool": adapter.source_tool,
        "sources": tuple(_source_summary(item) for item in imported),
    }

    excluded: list[str] = []
    groups: dict[str, list[_Sample]] = {"baseline": [], "candidate": []}
    keys: dict[str, set[tuple[Any, ...]]] = {"baseline": set(), "candidate": set()}
    for side, items in (("baseline", baseline), ("candidate", candidate)):
        for item in items:
            sample = adapter.sample(item)
            if isinstance(sample, str):
                excluded.append(f"{side} {adapter.run_id(item)}: {sample}")
                continue
            groups[side].append(sample)
            keys[side].add(adapter.key(item))
    ids = {
        side: tuple(sample.run_id for sample in samples)
        for side, samples in groups.items()
    }
    context: dict[str, Any] = {
        **base,
        "baseline_run_ids": ids["baseline"],
        "candidate_run_ids": ids["candidate"],
        "excluded": tuple(excluded),
    }

    def inconclusive(reason: str, **extra: Any) -> GateReport:
        return GateReport(
            verdict=GateVerdict.INCONCLUSIVE, reason=reason, **context, **extra
        )

    for side, samples in groups.items():
        if len(samples) < min_repetitions:
            return inconclusive(
                f"{side} has {len(samples)} usable run(s); at least "
                f"{min_repetitions} are needed"
            )
        if len(keys[side]) > 1:
            return inconclusive(
                f"{side} runs are not comparable with each other "
                f"({adapter.differs})"
            )
    if keys["baseline"] != keys["candidate"]:
        return inconclusive(
            f"baseline and candidate are not comparable ({adapter.differs})"
        )
    all_samples = groups["baseline"] + groups["candidate"]
    units = {sample.unit for sample in all_samples}
    provenances = {sample.provenance for sample in all_samples}
    if len(units) > 1 or len(provenances) > 1:
        return inconclusive(
            f"{metric} mixes units {sorted(units)} or provenance "
            f"{sorted(provenances)}; values measured differently are not compared"
        )

    b_values = [sample.value for sample in groups["baseline"]]
    c_values = [sample.value for sample in groups["candidate"]]
    b_mean, c_mean = statistics.mean(b_values), statistics.mean(c_values)
    b_sd, c_sd = statistics.pstdev(b_values), statistics.pstdev(c_values)
    stats: dict[str, Any] = {
        "unit": units.pop(),
        "provenance": provenances.pop(),
        "baseline_mean": b_mean,
        "candidate_mean": c_mean,
        "baseline_stdev": b_sd,
        "candidate_stdev": c_sd,
    }
    if not math.isfinite(b_mean) or b_mean <= 0:
        return inconclusive("baseline mean is not positive", **stats)
    worse_by = (b_mean - c_mean) if higher_is_better else (c_mean - b_mean)
    delta_pct = worse_by / b_mean
    stats.update(delta=c_mean - b_mean, delta_pct=delta_pct)
    unit = stats["unit"]
    noise = b_sd + c_sd

    if delta_pct <= max_regression:
        allowance = max_regression * b_mean
        if noise > allowance:
            return inconclusive(
                f"candidate looks {abs(delta_pct):.1%} "
                f"{'better' if worse_by < 0 else 'worse'}, but the combined "
                f"run-to-run noise ({noise:.6g} {unit}) is larger than the "
                f"allowed regression ({allowance:.6g} {unit}), so a pass could "
                "be chance; collect more runs or reduce noise",
                **stats,
            )
        change = "better" if worse_by < 0 else "worse"
        return GateReport(
            verdict=GateVerdict.PASS,
            reason=(
                f"candidate mean {metric} is {abs(delta_pct):.1%} {change} than "
                f"baseline ({c_mean:.6g} vs {b_mean:.6g} {unit}); the allowed "
                f"regression is {max_regression:.1%}"
            ),
            **context,
            **stats,
        )
    if worse_by <= noise:
        return inconclusive(
            f"candidate looks {delta_pct:.1%} worse, but the difference "
            f"({worse_by:.6g} {unit}) is within the combined run-to-run noise "
            f"({noise:.6g} {unit}); collect more runs or reduce noise",
            **stats,
        )
    return GateReport(
        verdict=GateVerdict.REGRESSION,
        reason=(
            f"candidate mean {metric} is {delta_pct:.1%} worse than baseline "
            f"({c_mean:.6g} vs {b_mean:.6g} {unit}), beyond the allowed "
            f"{max_regression:.1%} and the combined noise ({noise:.6g} {unit})"
        ),
        **context,
        **stats,
    )
