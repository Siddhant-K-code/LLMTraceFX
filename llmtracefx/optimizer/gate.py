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
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .collectors._shared import atomic_write_text
from .schema import ExperimentRecord, Measurement, SchemaValidationError

GATE_REPORT_KIND = "llmtracefx.gate.report"
GATE_REPORT_SCHEMA_VERSION = "1"
_PHASE_METRICS = ("total", "model_load", "tokenize", "prefill", "decode")
DECODE_TPS = "decode_tokens_per_second"
METRIC_CHOICES = (*_PHASE_METRICS, DECODE_TPS)


class GateVerdict(StrEnum):
    PASS = "pass"
    REGRESSION = "regression"
    INCONCLUSIVE = "inconclusive"


class GateInputError(ValueError):
    """A record path could not be read or did not hold a valid record."""


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


def evaluate_gate(
    baseline: Sequence[ExperimentRecord],
    candidate: Sequence[ExperimentRecord],
    *,
    metric: str = "total",
    max_regression: float = 0.05,
    min_repetitions: int = 3,
) -> GateReport:
    if metric not in METRIC_CHOICES:
        raise ValueError(f"metric must be one of {', '.join(METRIC_CHOICES)}")
    if not 0 <= max_regression < 1:
        raise ValueError("max_regression must be in [0, 1)")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be at least 2 to estimate noise")
    higher_is_better = metric == DECODE_TPS
    base = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "max_regression": max_regression,
        "min_repetitions": min_repetitions,
    }

    excluded: list[str] = []
    groups: dict[str, list[_Sample]] = {"baseline": [], "candidate": []}
    keys: dict[str, set[tuple[Any, ...]]] = {"baseline": set(), "candidate": set()}
    for side, records in (("baseline", baseline), ("candidate", candidate)):
        for record in records:
            sample = _sample(record, metric)
            if isinstance(sample, str):
                excluded.append(f"{side} {record.run_id}: {sample}")
                continue
            groups[side].append(sample)
            keys[side].add(_comparability(record, metric))
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
                f"{side} runs are not comparable with each other (model, "
                "runtime backend, platform, workload or token counts differ)"
            )
    if keys["baseline"] != keys["candidate"]:
        return inconclusive(
            "baseline and candidate are not comparable (model, runtime backend, "
            "platform, workload or token counts differ)"
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
