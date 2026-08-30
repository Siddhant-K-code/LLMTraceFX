"""Shared synthetic-artifact builder for ``compare`` tests.

Not a test module (no ``test_`` prefix, so pytest never collects it). It
builds a ``workloads run --output-dir``-shaped tree -- ``verification.json``
plus ``final_record.json`` under ``<results_dir>/runs/<run_id>/``, and
optionally an ``api_evidence.json`` under that run's collection directory --
without running the verify pipeline, loading a model, or calling an API.

Everything produced here is synthetic. The numbers are chosen to exercise
specific code paths and are not measurements of any real system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.collectors._shared import sha256_bytes
from llmtracefx.optimizer.collectors.openai_api import (
    API_EVIDENCE_SCHEMA_VERSION,
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
)
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
from llmtracefx.optimizer.workloads.api_verify import RUN_MANIFEST_SCHEMA_VERSION
from llmtracefx.optimizer.workloads.verify import RowStatus, RowVerification

DEFAULT_PROMPT_HASH = "sha256:synthetic-prompt-abc"
DEFAULT_WORKLOAD_ID = "structured-json-profile-extraction"
DEFAULT_QUALITY_METRIC = "structured_json_exact_field_match"


def api_evidence_payload(
    *,
    run_id: str,
    provider: str = "z-ai",
    model_id: str = "glm-5.3",
    model_revision: str | None = None,
    reasoning_effort: str | None = "high",
    thinking_type: str | None = None,
    max_output_tokens: int | None = 512,
    temperature: float | None = 0.0,
    top_p: float | None = 1.0,
    usage_reported: bool = True,
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 400,
    cached_prompt_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    first_content_token_offset_ms: float | None = 220.0,
    malformed_fields: tuple[str, ...] = (),
    workload_hash: str | None = DEFAULT_PROMPT_HASH,
    config_hash: str | None = "api-cfg-hash",
    endpoint_origin: str | None = "https://example.invalid",
    endpoint_path: str | None = "/v1/chat/completions",
    system_prompt_hash: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the subset of ``api_evidence.json`` the compare loader reads."""
    request_parameters: dict[str, Any] = {}
    if max_output_tokens is not None:
        request_parameters["max_tokens"] = max_output_tokens
    if temperature is not None:
        request_parameters["temperature"] = temperature
    if top_p is not None:
        request_parameters["top_p"] = top_p

    extensions: dict[str, Any] = {}
    if reasoning_effort is not None:
        extensions["reasoning_effort"] = reasoning_effort
    if thinking_type is not None:
        extensions["thinking"] = {"type": thinking_type}

    if messages is None:
        messages = []
        if system_prompt_hash is not None:
            messages.append(
                {
                    "role": "system",
                    "characters": 32,
                    "content_sha256": system_prompt_hash,
                }
            )
        messages.append(
            {
                "role": "user",
                "characters": 64,
                "content_sha256": workload_hash or DEFAULT_PROMPT_HASH,
            }
        )

    return {
        "schema_version": API_EVIDENCE_SCHEMA_VERSION,
        "run_id": run_id,
        "collected_at": utc_now_iso(),
        "plan": {
            "schema_version": API_EVIDENCE_SCHEMA_VERSION,
            "provider": provider,
            "model_id": model_id,
            "model_revision": model_revision,
            "endpoint_origin": endpoint_origin,
            "endpoint_path": endpoint_path,
            "messages": messages,
            "request_parameters": request_parameters,
            "provider_extensions": extensions,
            "workload_hash": workload_hash,
            "config_hash": config_hash,
        },
        "success": True,
        "usage": {
            "reported": usage_reported,
            "provenance": MetricProvenance.PROVIDER_REPORTED.value,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                None
                if prompt_tokens is None or completion_tokens is None
                else prompt_tokens + completion_tokens
            ),
            "cached_prompt_tokens": cached_prompt_tokens,
            "reasoning_tokens": reasoning_tokens,
            "malformed_fields": list(malformed_fields),
        },
        "timeline": {
            "clock": "monotonic_client_perf_counter",
            "provenance": MetricProvenance.MEASURED_WALL_CLOCK.value,
            "first_content_token_offset_ms": first_content_token_offset_ms,
        },
        "statistics": {},
        "reasoning_content_returned": reasoning_tokens is not None,
        "reasoning_text_persisted": False,
    }


def write_run(
    results_dir: Path,
    run_id: str,
    *,
    status: RowStatus = RowStatus.COMPLETED,
    workload_id: str = DEFAULT_WORKLOAD_ID,
    workload_version: str = "1",
    category: str = "structured_json",
    context_tier: str = "2k",
    decode_mode: str = "autoregressive",
    model_id: str = "local/qwen3-8b",
    model_family: str | None = "qwen3_next",
    model_revision: str | None = None,
    provider: str | None = None,
    accelerator: str | None = "Apple M5 Pro",
    runtime_name: str = "mlx-lm",
    runtime_version: str | None = "0.32.2",
    runtime_backend: str | None = "Metal",
    quantization: str | None = "Q4",
    seed: int | None = 0,
    config_hash: str | None = "cfg-hash",
    prompt_hash: str | None = DEFAULT_PROMPT_HASH,
    total_ms: float | None = 4200.0,
    total_provenance: MetricProvenance = MetricProvenance.MEASURED_WALL_CLOCK,
    prefill_ms: float | None = 310.0,
    peak_bytes: float | None = 9 * 1024**3,
    success: bool = True,
    quality_score: float | None = 1.0,
    quality_metric: str | None = DEFAULT_QUALITY_METRIC,
    max_tokens_argv: int | None = 512,
    temperature_argv: float | None = 0.0,
    top_p_argv: float | None = 1.0,
    api_evidence: dict[str, Any] | None = None,
    api_evidence_text: str | None = None,
    write_artifact_marker: bool = True,
    write_final_record: bool = True,
    corrupt_final_record: bool = False,
    reason: str | None = None,
) -> Path:
    """Write one synthetic run under ``results_dir/runs/<run_id>/``.

    By default this is a fully passing local MLX row. Pass ``provider`` and
    ``api_evidence`` to build a hosted-API row instead. Every axis can be
    overridden to exercise a specific edge case.
    """
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    collection_dir = run_dir / "collection"

    argv: list[str] = ["llmtracefx-optimizer", "collect-mlx", "--run-id", run_id]
    if max_tokens_argv is not None:
        argv.extend(("--max-tokens", str(max_tokens_argv)))
    if temperature_argv is not None:
        argv.extend(("--temperature", str(temperature_argv)))
    if top_p_argv is not None:
        argv.extend(("--top-p", str(top_p_argv)))

    final_record_path: Path | None = None
    if write_final_record:
        final_record_path = run_dir / "final_record.json"
        record = ExperimentRecord(
            run_id=run_id,
            started_at=utc_now_iso(),
            platform=PlatformInfo(
                os_name="Darwin",
                os_version="26.0",
                architecture="arm64",
                accelerator=accelerator,
            ),
            model=ModelInfo(
                model_id=model_id,
                model_family=model_family,
                model_revision=model_revision,
                quantization=quantization,
            ),
            runtime=RuntimeInfo(
                name=runtime_name,
                version=runtime_version,
                backend=runtime_backend,
                provider=provider,
            ),
            command=CommandInfo(
                argv=tuple(argv),
                config_hash=config_hash,
                workload_hash=prompt_hash,
            ),
            repetition=RepetitionInfo(
                warmup_repetitions=0,
                measured_repetitions=1,
                repetition_index=0,
                seed=seed,
            ),
            tokens=TokenCounts(input_tokens=1000, generated_tokens=400),
            timing=TimingMetrics(
                prefill=(
                    None
                    if prefill_ms is None
                    else Measurement(
                        value=prefill_ms,
                        provenance=MetricProvenance.MEASURED_NATIVE,
                        unit="ms",
                    )
                ),
                total=(
                    None
                    if total_ms is None
                    else Measurement(
                        value=total_ms, provenance=total_provenance, unit="ms"
                    )
                ),
            ),
            speculative=SpeculativeDecodingInfo(enabled=False),
            memory=MemoryMetrics(
                peak=(
                    None
                    if peak_bytes is None
                    else Measurement(
                        value=peak_bytes,
                        provenance=MetricProvenance.MEASURED_NATIVE,
                        unit="bytes",
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
            final_record_path.write_text("not json", encoding="utf-8")
        else:
            record.write_json(final_record_path)

    if api_evidence is not None or api_evidence_text is not None:
        collection_dir.mkdir(parents=True, exist_ok=True)
        payload = (
            api_evidence_text
            if api_evidence_text is not None
            else json.dumps(api_evidence, indent=2)
        )
        artifacts: list[tuple[str, str]] = [("api_evidence.json", payload)]
        record_text = (
            final_record_path.read_text(encoding="utf-8")
            if final_record_path is not None and not corrupt_final_record
            else "{}\n"
        )
        # The collector's completeness marker names the whole canonical set,
        # so a faithful fixture writes all four artifacts. Anything less is
        # an incomplete set, which is exactly what the marker exists to catch.
        artifacts.append(("record.json", record_text))
        artifacts.append(("response.txt", "synthetic response\n"))
        artifacts.append(("environment.json", '{"synthetic": true}\n'))
        for name, text in artifacts:
            (collection_dir / name).write_text(text, encoding="utf-8")
        if write_artifact_marker:
            marker = {
                "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
                "run_id": run_id,
                "artifacts": [
                    {"name": name, "sha256": sha256_bytes(text.encode("utf-8"))}
                    for name, text in artifacts
                ],
            }
            (collection_dir / ARTIFACT_MANIFEST_NAME).write_text(
                json.dumps(marker, indent=2) + "\n", encoding="utf-8"
            )

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
        recorded_prompt_hash=prompt_hash,
        verified_prompt_hash=prompt_hash,
        run_binding_hash="sha256:synthetic-binding",
        resumed=(status == RowStatus.SKIPPED),
        outcome_success=success if write_final_record else None,
        quality_score=quality_score if write_final_record else None,
        total_ms=total_ms if write_final_record else None,
        started_at=utc_now_iso(),
        ended_at=utc_now_iso(),
        final_record_path=(
            str(final_record_path) if final_record_path is not None else None
        ),
        collection_dir=str(collection_dir) if write_final_record else None,
    )
    (run_dir / "verification.json").write_text(verification.to_json(), encoding="utf-8")
    return run_dir


def write_api_run(
    results_dir: Path,
    run_id: str,
    *,
    provider: str = "z-ai",
    model_id: str = "glm-5.3",
    model_revision: str | None = None,
    reasoning_effort: str | None = "high",
    total_ms: float = 2600.0,
    prompt_tokens: int | None = 1000,
    completion_tokens: int | None = 400,
    cached_prompt_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    usage_reported: bool = True,
    first_content_token_offset_ms: float | None = 220.0,
    max_output_tokens: int | None = 512,
    temperature: float | None = 0.0,
    top_p: float | None = 1.0,
    config_hash: str | None = "api-cfg-hash",
    thinking_type: str | None = None,
    system_prompt_hash: str | None = None,
    prompt_hash: str | None = DEFAULT_PROMPT_HASH,
    endpoint_origin: str | None = "https://example.invalid",
    success: bool = True,
    quality_score: float | None = 1.0,
    write_run_seal: bool = True,
    **kwargs: Any,
) -> Path:
    """Write one synthetic hosted-API run, sidecar and run seal included."""
    run_dir = write_run(
        results_dir,
        run_id,
        provider=provider,
        model_id=model_id,
        model_revision=model_revision,
        accelerator=None,
        runtime_name="openai-compatible-stream",
        runtime_version="1",
        runtime_backend=None,
        quantization=None,
        model_family=None,
        # A hosted API reports no local prefill and no local peak memory.
        prefill_ms=None,
        peak_bytes=None,
        prompt_hash=prompt_hash,
        total_ms=total_ms,
        success=success,
        quality_score=quality_score,
        max_tokens_argv=None,
        temperature_argv=None,
        top_p_argv=None,
        config_hash=config_hash,
        api_evidence=api_evidence_payload(
            run_id=run_id,
            provider=provider,
            model_id=model_id,
            model_revision=model_revision,
            reasoning_effort=reasoning_effort,
            thinking_type=thinking_type,
            config_hash=config_hash,
            workload_hash=prompt_hash,
            system_prompt_hash=system_prompt_hash,
            endpoint_origin=endpoint_origin,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            usage_reported=usage_reported,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            first_content_token_offset_ms=first_content_token_offset_ms,
        ),
        **kwargs,
    )
    if write_run_seal:
        write_run_manifest(results_dir, run_id)
    return run_dir


def write_run_manifest(results_dir: Path, run_id: str) -> None:
    """Seal a run directory the way ``workloads run-api`` does.

    ``run-api`` writes this last, hashing the collector's own marker together
    with ``final_record.json`` and ``verification.json``. A hosted run
    without it is not something the pipeline produces, so fixtures that omit
    it were not representative of the input this layer actually receives.
    """
    run_dir = results_dir / "runs" / run_id
    collection = run_dir / "collection"
    if not (collection / "artifacts.json").is_file():
        # ``run-api`` seals only a row that produced a full artifact set; a
        # row whose collector marker is missing has nothing to seal.
        return
    payload = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "artifacts": [
            {"name": name, "sha256": sha256_bytes(path.read_bytes())}
            for name, path in (
                ("collection/artifacts.json", collection / "artifacts.json"),
                ("final_record.json", run_dir / "final_record.json"),
                ("verification.json", run_dir / "verification.json"),
            )
        ],
    }
    (run_dir / "run.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


PRICING_MANIFEST: dict[str, Any] = {
    "schema_version": "1",
    "name": "synthetic illustrative rates",
    "currency": "USD",
    "entries": [
        {
            "entry_id": "glm-5.3-illustrative",
            "provider": "z-ai",
            "model_id": "glm-5.3",
            "currency": "USD",
            "effective_at": "2026-01-01",
            "source": "illustrative example, not a published price list",
            "rates_are_illustrative": True,
            "input_per_million": 0.6,
            "output_per_million": 2.2,
        },
        {
            "entry_id": "glm-5.3-flash-illustrative",
            "provider": "z-ai",
            "model_id": "glm-5.3-flash",
            "currency": "USD",
            "effective_at": "2026-01-01",
            "source": "illustrative example, not a published price list",
            "rates_are_illustrative": True,
            "input_per_million": 0.1,
            "output_per_million": 0.3,
        },
    ],
}


COMPARE_POLICY: dict[str, Any] = {
    "schema_version": "1",
    "name": "synthetic latency comparison",
    "objective": "min_mean_total_latency_ms",
    "constraints": {"min_pass_rate": 0.5},
}


def reseal_run(results_dir: Path, run_id: str) -> None:
    """Recompute ``run.json`` if this run has one.

    A test that edits a sealed artifact to exercise one specific check would
    otherwise trip the run seal first and be excluded for the wrong reason.
    Tests that mean to break the seal do so explicitly.
    """
    if (results_dir / "runs" / run_id / "run.json").is_file():
        write_run_manifest(results_dir, run_id)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def collection_dir_for(results_dir: Path, run_id: str) -> Path:
    return results_dir / "runs" / run_id / "collection"


def refresh_artifact_marker(collection_dir: Path) -> None:
    """Recompute ``artifacts.json`` over whatever is currently on disk.

    A test that edits ``api_evidence.json`` to exercise a specific identity
    check would otherwise trip the completeness marker first and be excluded
    for the wrong reason. Refreshing the marker keeps each test aimed at the
    one check it is about.
    """
    names = sorted(
        name
        for name in (
            "record.json",
            "response.txt",
            "api_evidence.json",
            "environment.json",
        )
        if (collection_dir / name).is_file()
    )
    marker = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": collection_dir.parent.name,
        "artifacts": [
            {"name": name, "sha256": sha256_bytes((collection_dir / name).read_bytes())}
            for name in names
        ],
    }
    (collection_dir / ARTIFACT_MANIFEST_NAME).write_text(
        json.dumps(marker, indent=2) + "\n", encoding="utf-8"
    )
    run_dir = collection_dir.parent
    if (run_dir / "run.json").is_file():
        write_run_manifest(run_dir.parent.parent, run_dir.name)


def edit_sidecar(results_dir: Path, run_id: str, mutate: Any) -> Path:
    """Mutate a run's ``api_evidence.json`` and refresh the marker over it."""
    collection = collection_dir_for(results_dir, run_id)
    sidecar = collection / "api_evidence.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    mutate(payload)
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    refresh_artifact_marker(collection)
    return sidecar
