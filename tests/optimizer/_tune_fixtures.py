"""Shared full-fake-artifact-tree builder for `tune` tests.

Not a test module itself (no `test_` prefix, so pytest never collects it);
imported by the various ``test_tune_*.py`` modules that need to build a
`workloads run --output-dir`-shaped results directory (verification.json +
final_record.json under `<results_dir>/runs/<run_id>/`) without actually
running the verify pipeline or loading a model.
"""

from __future__ import annotations

from pathlib import Path

from llmtracefx.optimizer.schema import (
    CommandInfo,
    ExperimentRecord,
    Measurement,
    MemoryMetrics,
    MetricProvenance,
    ModelInfo,
    OutcomeInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SpeculativeDecodingInfo,
    TimingMetrics,
    TokenCounts,
    utc_now_iso,
)
from llmtracefx.optimizer.workloads.verify import RowStatus, RowVerification

DEFAULT_PROMPT_HASH = "sha256:promptabc"


def write_run(
    results_dir: Path,
    run_id: str,
    *,
    status: RowStatus = RowStatus.COMPLETED,
    workload_id: str = "structured-json-profile-extraction",
    workload_version: str = "1",
    category: str = "structured_json",
    context_tier: str = "2k",
    decode_mode: str = "autoregressive",
    model_id: str = "local/qwen3-8b",
    model_family: str | None = "qwen3_next",
    model_revision: str | None = None,
    tokenizer_revision: str | None = None,
    accelerator: str | None = "Apple M5 Pro",
    runtime_name: str = "mlx-lm",
    runtime_version: str | None = "0.31.3",
    runtime_backend: str | None = "Metal",
    quantization: str | None = "Q4",
    speculative_enabled: bool = False,
    speculative_method: str | None = None,
    speculative_depth: int | None = None,
    seed: int | None = 0,
    config_hash: str | None = "cfg-hash",
    prompt_hash: str | None = DEFAULT_PROMPT_HASH,
    recorded_prompt_hash: str | None = None,
    workload_hash_override: str | None = None,
    total_ms: float | None = 1000.0,
    total_provenance: MetricProvenance = MetricProvenance.MEASURED_WALL_CLOCK,
    peak_bytes: float | None = 4 * 1024**3,
    peak_provenance: MetricProvenance = MetricProvenance.MEASURED_NATIVE,
    success: bool = True,
    quality_score: float | None = 1.0,
    quality_metric: str | None = "structured_json_exact_field_match",
    write_final_record: bool = True,
    corrupt_final_record: bool = False,
    reason: str | None = None,
) -> Path:
    """Write one fake `<run_id>` entry under `results_dir/runs/`.

    Returns the run's directory. By default builds a fully passing,
    fully-evidenced ``COMPLETED`` autoregressive row; every axis can be
    overridden to exercise a specific rejection/edge case.
    """
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    final_record_path: Path | None = None
    if write_final_record:
        final_record_path = run_dir / "final_record.json"
        record = ExperimentRecord(
            run_id=run_id,
            started_at=utc_now_iso(),
            platform=PlatformInfo(
                os_name="Darwin",
                os_version="24.0",
                architecture="arm64",
                accelerator=accelerator,
            ),
            model=ModelInfo(
                model_id=model_id,
                model_family=model_family,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                quantization=quantization,
            ),
            runtime=RuntimeInfo(
                name=runtime_name, version=runtime_version, backend=runtime_backend
            ),
            command=CommandInfo(
                argv=("llmtracefx-optimizer", "collect-mlx"),
                config_hash=config_hash,
                workload_hash=(
                    workload_hash_override
                    if workload_hash_override is not None
                    else prompt_hash
                ),
            ),
            repetition=RepetitionInfo(
                warmup_repetitions=0,
                measured_repetitions=1,
                repetition_index=0,
                seed=seed,
            ),
            tokens=TokenCounts(input_tokens=10, context_tokens=10, generated_tokens=20),
            timing=TimingMetrics(
                total=(
                    None
                    if total_ms is None
                    else Measurement(
                        value=total_ms, provenance=total_provenance, unit="ms"
                    )
                )
            ),
            speculative=SpeculativeDecodingInfo(
                enabled=speculative_enabled,
                method=speculative_method,
                configured_depth=speculative_depth,
            ),
            memory=MemoryMetrics(
                peak=(
                    None
                    if peak_bytes is None
                    else Measurement(
                        value=peak_bytes, provenance=peak_provenance, unit="bytes"
                    )
                )
            ),
            outcome=OutcomeInfo(
                success=success,
                quality_score=quality_score,
                quality_metric=quality_metric,
            ),
        )
        if corrupt_final_record:
            final_record_path.parent.mkdir(parents=True, exist_ok=True)
            final_record_path.write_text("not json", encoding="utf-8")
        else:
            record.write_json(final_record_path)

    verification = RowVerification(
        schema_version="1",
        run_id=run_id,
        workload_id=workload_id,
        workload_version=workload_version,
        category=category,
        context_tier=context_tier,
        decode_mode=decode_mode,
        status=status,
        reason=reason,
        recorded_prompt_hash=(
            recorded_prompt_hash if recorded_prompt_hash is not None else prompt_hash
        ),
        verified_prompt_hash=prompt_hash,
        run_binding_hash="sha256:bindingabc",
        resumed=(status == RowStatus.SKIPPED),
        outcome_success=success if write_final_record else None,
        quality_score=quality_score if write_final_record else None,
        total_ms=total_ms if write_final_record else None,
        started_at=utc_now_iso(),
        ended_at=utc_now_iso(),
        final_record_path=(
            str(final_record_path) if final_record_path is not None else None
        ),
        collection_dir=str(run_dir / "collection") if write_final_record else None,
    )
    (run_dir / "verification.json").write_text(verification.to_json(), encoding="utf-8")
    return run_dir
