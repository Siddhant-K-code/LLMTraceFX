"""Report/evidence tooling for the Qwen3-8B M5 Pro control.

Reuses ``lab.core``'s report machinery almost entirely unmodified: the
per-row manifest-binding checks (``_record_matches_manifest``), the
environment-drift checks (``_environment_errors``), state scoping
(``_current_state_scope``), tier aggregation (``_tier_summary``),
sanitized-report enforcement (``assert_shareable``), self-contained HTML
rendering (``render_lab_report_html``), and the tune/compare layers are
all the exact functions the packaged 27B lab uses.

Only the top-level report assembly differs, and only where it must:
this control's "model" section describes a self-converted checkpoint
(so it never repeats the 27B report's disclaimer about not binding a
source revision -- this control *does* bind one), and its "limitations"
explicitly says the results are not comparable to the 27B lab's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...collectors._shared import atomic_write_text
from ...compare.compare import compare
from ...compare.policy import CompareConstraints, CompareObjective, ComparePolicy
from ...compare.report_html import render_compare_report_html
from ...schema import ExperimentRecord, MetricProvenance
from ...tune.policy import TuneConstraints, TuneObjective, TunePolicy
from ...tune.report_html import render_tune_report_html
from ...tune.tuner import tune
from ...workloads.verify import RowStatus, RowVerification
from ..core import (
    LabError,
    _current_state_scope,
    _environment_errors,
    _materialize_current_results,
    _measurement,
    _record_matches_manifest,
    _stale_result_run_ids,
    _tier_summary,
    assert_shareable,
    render_lab_report_html,
    verify_catalog,
    verify_evidence,
)
from ..manifest import LabManifest

CONTROL_REPORT_SCHEMA_VERSION = "1"

#: ``verify_evidence`` is fully manifest-driven already (no 27B-specific
#: literal anywhere in it); reused verbatim.
verify_control_evidence = verify_evidence


def _tier_with_tokens(
    manifest: LabManifest,
    name: str,
    rows: list[tuple[RowVerification, ExperimentRecord]],
) -> dict[str, Any]:
    summary = _tier_summary(name, rows)
    evaluated_actual = [
        record.tokens.input_tokens
        for verification, record in rows
        if record.tokens.input_tokens is not None
        and verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
    ]
    summary["requested_tokens"] = manifest.tier(name).target_tokens
    summary["mean_actual_input_tokens"] = (
        sum(evaluated_actual) / len(evaluated_actual) if evaluated_actual else None
    )
    return summary


def build_control_report(manifest: LabManifest, *, workspace: Path) -> dict[str, Any]:
    state_path = workspace / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("status") == "running":
        raise LabError(
            "current benchmark run is incomplete; report generation is refused"
        )
    allowed_tiers, measured_run_ids, warmup_run_ids = _current_state_scope(
        manifest, state
    )
    environment: dict[str, Any] | None = None
    rows_by_tier: dict[str, list[tuple[RowVerification, ExperimentRecord]]] = {}
    runs_dir = workspace / "results" / "runs"
    if runs_dir.is_dir():
        for verification_path in sorted(runs_dir.glob("*/verification.json")):
            verification = RowVerification.read_json(verification_path)
            if (
                verification.final_record_path is None
                or verification.run_id not in measured_run_ids
            ):
                continue
            record = ExperimentRecord.read_json(
                verification_path.parent / "final_record.json"
            )
            if not _record_matches_manifest(
                manifest,
                verification,
                record,
                warmup=False,
                allowed_tiers=allowed_tiers,
            ):
                continue
            environment_path = (
                verification_path.parent / "collection" / "environment.json"
            )
            if _environment_errors(manifest, environment_path):
                continue
            if environment is None and environment_path.is_file():
                raw_environment = json.loads(
                    environment_path.read_text(encoding="utf-8")
                )
                environment = {
                    "os_name": raw_environment.get("os_name"),
                    "os_release": raw_environment.get("os_release"),
                    "architecture": raw_environment.get("architecture"),
                    "python_implementation": raw_environment.get(
                        "python_implementation"
                    ),
                    "python_version": raw_environment.get("python_version"),
                    "cpu_count": raw_environment.get("cpu_count"),
                    "total_memory_gb": raw_environment.get("total_memory_gb"),
                    "accelerator": record.platform.accelerator,
                    "package_versions": raw_environment.get("package_versions", {}),
                }
            rows_by_tier.setdefault(verification.context_tier, []).append(
                (verification, record)
            )
    tier_order = {tier.name: tier.order for tier in manifest.context_tiers}
    tiers = [
        _tier_with_tokens(manifest, name, rows)
        for name, rows in sorted(
            rows_by_tier.items(), key=lambda item: tier_order[item[0]]
        )
    ]
    failed_attempts: list[dict[str, Any]] = []
    warmups_dir = workspace / "warmups"
    if warmups_dir.is_dir():
        for verification_path in sorted(warmups_dir.glob("*/runs/*/verification.json")):
            verification = RowVerification.read_json(verification_path)
            if verification.run_id not in warmup_run_ids:
                continue
            record_path = verification_path.parent / "final_record.json"
            failed_record = (
                ExperimentRecord.read_json(record_path)
                if verification.final_record_path is not None and record_path.is_file()
                else None
            )
            if failed_record is not None and not _record_matches_manifest(
                manifest,
                verification,
                failed_record,
                warmup=True,
                allowed_tiers=allowed_tiers,
            ):
                continue
            if failed_record is not None and _environment_errors(
                manifest, verification_path.parent / "collection" / "environment.json"
            ):
                continue
            if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED):
                continue
            failed_attempts.append(
                {
                    "run_id": verification.run_id,
                    "tier": verification.context_tier,
                    "phase": "warmup",
                    "status": verification.status.value,
                    "reason": verification.reason,
                    "input_tokens": (
                        failed_record.tokens.input_tokens
                        if failed_record is not None
                        else None
                    ),
                    "total_ms": (
                        _measurement(failed_record, "total")
                        if failed_record is not None
                        else None
                    ),
                    "peak_memory_bytes": (
                        failed_record.memory.peak.value
                        if failed_record is not None
                        and failed_record.memory.peak is not None
                        else None
                    ),
                }
            )
    for tier_rows in rows_by_tier.values():
        for verification, record in tier_rows:
            if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED):
                continue
            failed_attempts.append(
                {
                    "run_id": verification.run_id,
                    "tier": verification.context_tier,
                    "phase": "measured",
                    "status": verification.status.value,
                    "reason": verification.reason,
                    "input_tokens": record.tokens.input_tokens,
                    "total_ms": _measurement(record, "total"),
                    "peak_memory_bytes": (
                        record.memory.peak.value
                        if record.memory.peak is not None
                        else None
                    ),
                }
            )
    stop_reasons = state.get("stop_reasons", [])
    if failed_attempts:
        stop_reasons = [
            attempt["reason"]
            for attempt in failed_attempts
            if attempt["reason"] is not None
        ]
    stopped_during = state.get("stopped_during")
    if stopped_during is None and failed_attempts:
        stopped_during = {
            "tier": failed_attempts[0]["tier"],
            "phase": failed_attempts[0]["phase"],
        }
    unattempted_tiers = state.get("unattempted_tiers")
    if not isinstance(unattempted_tiers, list) and stopped_during is not None:
        stopped_tier = manifest.tier(stopped_during["tier"])
        unattempted_tiers = [
            tier.name
            for tier in manifest.context_tiers
            if tier.order > stopped_tier.order
        ]
    stopped_before_tier = (
        state.get("stopped_before_tier") if stopped_during is None else None
    )
    requested_max_tier = state.get("requested_max_tier")
    not_executed_tiers = [
        tier.name for tier in manifest.context_tiers if tier.name not in allowed_tiers
    ]
    preflight_snapshot = state.get("preflight", {}).get("snapshot", {})
    tier_states = state.get("tiers", [])
    last_postflight = (
        tier_states[-1].get("postflight", {}).get("snapshot", {})
        if tier_states and isinstance(tier_states[-1], dict)
        else {}
    )
    report = {
        "schema_version": CONTROL_REPORT_SCHEMA_VERSION,
        "lab_id": manifest.lab_id,
        "generated_at": state.get("ended_at"),
        "run_mode": state.get("run_mode"),
        "clean_boot_confirmed": state.get("clean_boot_confirmed"),
        "model": {
            "official_id": manifest.model.official_id,
            "official_revision": manifest.model.official_revision,
            "repository_id": manifest.model.repository_id,
            "revision": manifest.model.revision,
            "license": manifest.model.license,
            "quantization": manifest.model.quantization,
            "converter": manifest.model.converter,
            "converter_revision": manifest.model.converter_revision,
            "output_total_bytes": manifest.model.expected_download_bytes,
            "sources": list(manifest.model.sources),
        },
        "self_conversion": {
            "provenance": (
                "self-converted locally with this repository's pinned "
                "mlx-lm from the official upstream Qwen3-8B revision; "
                "never claims byte-equivalence with any third-party "
                "conversion (e.g. mlx-community)"
            ),
            "official_revision": manifest.model.official_revision,
            "converter": manifest.model.converter,
            "converter_revision": manifest.model.converter_revision,
            "quantization": manifest.model.quantization,
            "binding_revision": manifest.model.revision,
            "binding_revision_provenance": (
                "a deterministic fingerprint of the conversion identity "
                "(source revision, converter revision, quantization "
                "parameters, output file hashes), not an upstream git "
                "commit"
            ),
        },
        "environment": environment,
        "runtime_contract": {
            "name": manifest.runtime.name,
            "version": manifest.runtime.version,
            "mlx_version": manifest.runtime.mlx_version,
            "mlx_lm_version": manifest.runtime.mlx_lm_version,
            "transformers_version": manifest.runtime.transformers_version,
            "prefill_step_size": manifest.runtime.prefill_step_size,
        },
        "generation": {
            "max_output_tokens": manifest.generation.max_output_tokens,
            "seed": manifest.generation.seed,
            "temperature": manifest.generation.temperature,
            "top_p": manifest.generation.top_p,
            "enable_thinking": manifest.generation.enable_thinking,
        },
        "repetitions": {
            "warmup_per_tier": manifest.repetitions.warmup_per_tier,
            "measured_per_workload": manifest.repetitions.measured_per_workload,
        },
        "cooperative_timeout_seconds": manifest.cooperative_timeout_seconds,
        "safety": {
            "required_chip": manifest.safety.required_chip,
            "required_total_memory_bytes": manifest.safety.required_total_memory_bytes,
            "minimum_memory_free_percent": manifest.safety.minimum_memory_free_percent,
            "maximum_peak_memory_bytes": manifest.safety.maximum_peak_memory_bytes,
            "maximum_swap_used_bytes": manifest.safety.maximum_swap_used_bytes,
            "stop_on_any_failed_row": manifest.safety.stop_on_any_failed_row,
        },
        "safety_observations": {
            "provenance": "macOS memory_pressure and sysctl",
            "preflight_memory_free_percent": preflight_snapshot.get(
                "memory_free_percent"
            ),
            "preflight_swap_used_bytes": preflight_snapshot.get("swap_used_bytes"),
            "postflight_memory_free_percent": last_postflight.get(
                "memory_free_percent"
            ),
            "postflight_swap_used_bytes": last_postflight.get("swap_used_bytes"),
        },
        "tiers": tiers,
        "failed_attempts": failed_attempts,
        "stopped_before_tier": stopped_before_tier,
        "stopped_during": stopped_during,
        "requested_max_tier": requested_max_tier,
        "unattempted_tiers": unattempted_tiers or [],
        "not_executed_tiers": not_executed_tiers,
        "stop_reasons": stop_reasons,
        "limitations": [
            "This is a self-converted Qwen3-8B positive control, not the "
            "packaged Qwen3.8-27B lab; the two are different models, "
            "different quantized checkpoints, and different memory/timing "
            "envelopes, and are never directly comparable.",
            "Results apply only to the pinned official source revision, "
            "this repository's pinned mlx-lm converter revision, the "
            "exact recorded quantization parameters, and this M5 Pro "
            "configuration; they say nothing about any other conversion "
            "of Qwen3-8B, including third-party ones.",
            "This report only shows completion under the host state "
            "actually observed during the run; it is not a claim about "
            "peak achievable throughput or about any other machine.",
            "The workload catalog's context tiers target an approximate, "
            "model-independent token count; the checkpoint's own mlx-lm "
            "chat template and tokenizer generally produce a different "
            "*actual* input token count than that *requested* target -- "
            "each tier reports both, never conflated.",
            "Total, prefill, and decode durations are host wall-clock "
            "boundaries measured by a fresh, isolated subprocess per row; "
            "they are not kernel timings.",
            "Peak memory is MLX allocator peak memory when available, "
            "not whole-system resident memory.",
            "Decode token rate is derived from measured generated tokens "
            "and decode wall time.",
            "The per-row timeout is a parent-enforced subprocess wall-"
            "clock bound with TERM->KILL cleanup; it is not a claim about "
            "any cooperative in-process timeout being sufficient alone.",
            "No GPU utilization, bandwidth, power, energy, or kernel "
            "timing is measured or inferred.",
        ],
    }
    assert_shareable(report)
    return report


def write_control_reports(
    manifest: LabManifest, *, workspace: Path, shareable_dir: Path | None = None
) -> dict[str, Any]:
    verify_catalog(manifest)
    evidence_verification = verify_control_evidence(manifest, workspace=workspace)
    if not evidence_verification["verified"]:
        raise LabError(
            "report generation refused unverified current-run evidence: "
            + "; ".join(evidence_verification["failures"])
        )
    stale = _stale_result_run_ids(manifest, workspace=workspace)
    if stale:
        raise LabError(
            "report generation refused stale or mismatched run evidence: "
            + ", ".join(stale)
        )
    report = build_control_report(manifest, workspace=workspace)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        reports_dir / "control-report.json",
        json.dumps(report, indent=2, sort_keys=False) + "\n",
    )
    atomic_write_text(
        reports_dir / "control-report.html", render_lab_report_html(report)
    )

    results_dir = _materialize_current_results(manifest, workspace=workspace)
    tune_policy = TunePolicy(
        name="Qwen3-8B M5 Pro self-converted control",
        description=(
            "Rank only repetitions that pass deterministic evaluation, remain "
            "under the configured MLX peak-memory ceiling, and share identity."
        ),
        objective=TuneObjective.MAX_CORRECT_CASES_PER_MINUTE,
        constraints=TuneConstraints(
            min_pass_rate=1.0,
            max_peak_memory_bytes=float(manifest.safety.maximum_peak_memory_bytes),
            allowed_provenances=frozenset(
                {MetricProvenance.MEASURED_WALL_CLOCK, MetricProvenance.MEASURED_NATIVE}
            ),
            min_measured_repetitions=manifest.repetitions.measured_per_workload,
        ),
    )
    tune_report = tune(results_dirs=(results_dir,), policy=tune_policy)
    atomic_write_text(reports_dir / "tune-report.json", tune_report.to_json() + "\n")
    atomic_write_text(
        reports_dir / "tune-report.html",
        render_tune_report_html(tune_report, redact_paths=True),
    )

    compare_policy = ComparePolicy(
        name="Pinned self-converted local-system evidence",
        description=(
            "Apply the same deterministic pass-rate and repetition bar "
            "through the merged comparison layer; no hosted systems are "
            "included."
        ),
        objective=CompareObjective.MAX_CORRECT_CASES_PER_MINUTE,
        constraints=CompareConstraints(
            min_pass_rate=1.0,
            allowed_provenances=frozenset({MetricProvenance.MEASURED_WALL_CLOCK}),
            min_measured_repetitions=manifest.repetitions.measured_per_workload,
        ),
    )
    compare_report = compare(results_dirs=(results_dir,), policy=compare_policy)
    atomic_write_text(
        reports_dir / "compare-report.json", compare_report.to_json() + "\n"
    )
    atomic_write_text(
        reports_dir / "compare-report.html",
        render_compare_report_html(compare_report, redact_paths=True),
    )

    if shareable_dir is not None:
        shareable_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            shareable_dir / "evidence-summary.json",
            json.dumps(report, indent=2, sort_keys=False) + "\n",
        )
        atomic_write_text(shareable_dir / "report.html", render_lab_report_html(report))
    return report


__all__ = [
    "CONTROL_REPORT_SCHEMA_VERSION",
    "build_control_report",
    "verify_control_evidence",
    "write_control_reports",
]
