"""CLI entrypoint for the inference-optimizer foundation primitives.

Subcommands:
    manifest         Collect a CPU-only, non-sensitive environment manifest.
    run              Execute a configured experiment (warmups + measured reps).
    collect-mlx      Run one local MLX-LM inference and record normalized evidence.
    collect-api      Stream one OpenAI-compatible chat completion (e.g. Z.ai GLM)
                     and record normalized, credential-free evidence.
    native-mtp       Native Qwen MTP capability report / evidence collection.
    parse-llama-cpp  Convert llama.cpp text output into a canonical ExperimentRecord.
    doctor speculative  Diagnose whether speculative decoding/MTP is a net regression.
    workloads        Generate a deterministic code/JSON/reasoning workload matrix,
                     execute selected runnable rows (``workloads run``), and
                     aggregate results (``workloads summarize``).
    tune             Offline, evidence-constrained recommendation of the best
                     verified configuration for a workload/hardware target.
    tune-report      Render a `tune` JSON report as a self-contained, portable
                     HTML file for offline inspection (no Streamlit needed).
    optimize         End-to-end: execute selected matrix rows, tune the
                     resulting evidence, and render the JSON/HTML report,
                     composing the above primitives without duplicating any
                     of their logic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import NoReturn, Protocol, TypeVar, cast, overload

from .collectors._shared import atomic_write_text
from .collectors.mlx import (
    MLXCollectionConfig,
    MLXCollectorError,
    MLXLMRuntime,
    collect_mlx,
)
from .collectors.native_mtp import (
    NativeMTPCollectionConfig,
    NativeMTPCollectorError,
    capability_report_for_target,
    collect_native_mtp,
)
from .collectors.openai_api import (
    DEFAULT_RETAINED_EVENT_LIMIT,
    GLM_REASONING_EFFORT_LEVELS,
    THINKING_TYPES,
    APICollectionConfig,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    UrllibStreamingTransport,
    assert_credential_not_embedded,
    build_request_plan,
    collect_openai_stream,
    redact_text_for_dry_run,
)
from .doctor.speculative import diagnose_speculative_regression
from .manifest import collect_environment_manifest
from .optimize_summary import (
    OPTIMIZE_SUMMARY_SCHEMA_VERSION,
    OptimizeSummary,
    OptimizeSummaryValidationError,
    OverallStatus,
    PhaseName,
    PhaseReport,
    PhaseStatus,
    RecommendedCandidate,
    RowStatusCounts,
)
from .parsers.llama_cpp import LlamaCppParseError, build_experiment_record
from .runner import ExperimentRunner, RunnerConfig, RunnerConfigError
from .schema import (
    CommandInfo,
    ExperimentRecord,
    ModelInfo,
    PlatformInfo,
    RepetitionInfo,
    SchemaValidationError,
    utc_now_iso,
)
from .tune.explain import format_report_text
from .tune.loader import TuneInputError
from .tune.policy import TunePolicy, TunePolicyError
from .tune.report import GroupOutcome, TuneReport, TuneReportValidationError
from .tune.report_html import render_tune_report_html
from .tune.tuner import tune
from .workloads.aggregate import summarize_results, write_summary
from .workloads.catalog import WORKLOADS, workload_by_id
from .workloads.evaluators import evaluate_workload
from .workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    MatrixManifest,
    MatrixSchemaError,
    generate_matrix,
    write_matrix,
)
from .workloads.schema import ContextTier, WorkloadCategory, WorkloadSchemaError
from .workloads.verify import (
    RowPlan,
    RowSelection,
    RowStatus,
    RunBinding,
    VerifyError,
    plan_selected_rows,
    run_selected_rows,
)


def _platform_from_manifest(*, accelerator: str | None = None) -> PlatformInfo:
    manifest = collect_environment_manifest()
    return PlatformInfo(
        os_name=manifest.os_name,
        os_version=manifest.os_release,
        architecture=manifest.architecture,
        cpu_cores=manifest.cpu_count,
        total_memory_gb=manifest.total_memory_gb,
        accelerator=accelerator,
    )


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = collect_environment_manifest()
    text = manifest.to_json()
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Environment manifest written to {args.output}")
    else:
        print(text)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        config = RunnerConfig.from_file(args.config)
    except RunnerConfigError as exc:
        print(f"Invalid runner config: {exc}", file=sys.stderr)
        return 1

    runner = ExperimentRunner(config)
    results = runner.run(resume=not args.no_resume)
    succeeded = sum(1 for result in results if result.succeeded)
    print(
        f"Completed {succeeded}/{len(results)} measured repetitions for '{config.run_id}'"
    )
    print(f"Artifacts written to {config.results_dir}")
    return 0 if succeeded == len(results) else 1


def _collect_mlx_argv(args: argparse.Namespace) -> tuple[str, ...]:
    invocation = getattr(args, "_invocation", None)
    if invocation is not None:
        return tuple(invocation)

    reconstructed = [
        "llmtracefx-optimizer",
        "collect-mlx",
        "--run-id",
        args.run_id,
        "--model-path",
        args.model_path,
        "--model-id",
        args.model_id,
        "--prompt-file",
        args.prompt_file,
        "--output-dir",
        args.output_dir,
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--num-draft-tokens",
        str(args.num_draft_tokens),
    ]
    for flag, value in (
        ("--model-revision", args.model_revision),
        ("--tokenizer-revision", args.tokenizer_revision),
        ("--quantization", args.quantization),
        ("--accelerator", args.accelerator),
        ("--draft-model-path", args.draft_model_path),
    ):
        if value is not None:
            reconstructed.extend((flag, value))
    return tuple(reconstructed)


def _cmd_collect_mlx(args: argparse.Namespace) -> int:
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        config = MLXCollectionConfig(
            run_id=args.run_id,
            model_path=Path(args.model_path),
            model_id=args.model_id,
            prompt=prompt,
            output_dir=Path(args.output_dir),
            command_argv=_collect_mlx_argv(args),
            max_tokens=args.max_tokens,
            seed=args.seed,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            quantization=args.quantization,
            accelerator=args.accelerator,
            draft_model_path=(
                Path(args.draft_model_path) if args.draft_model_path else None
            ),
            num_draft_tokens=args.num_draft_tokens,
        )
        result = collect_mlx(config, runtime=MLXLMRuntime())
    except (OSError, UnicodeError, MLXCollectorError) as exc:
        print(f"Failed to collect MLX evidence: {exc}", file=sys.stderr)
        return 1

    record_path = config.output_dir / "record.json"
    print(f"MLX experiment record written to {record_path}")
    if result.record.outcome.success:
        return 0
    error_message = (
        result.record.error.message
        if result.record.error is not None
        else "unknown runtime failure"
    )
    print(f"MLX inference failed: {error_message}", file=sys.stderr)
    return 1


def _collect_api_argv(args: argparse.Namespace) -> tuple[str, ...]:
    """Rebuild a credential-free, fully explicit invocation for the record.

    The real ``sys.argv`` is deliberately not reused here. Reconstructing
    the command resolves every default (so the record states which
    environment variable held the credential instead of relying on the
    default at replay time) and guarantees that nothing the caller typed
    can reach an artifact.
    """
    reconstructed = [
        "llmtracefx-optimizer",
        "collect-api",
        "--run-id",
        args.run_id,
        "--provider",
        args.provider,
        "--endpoint",
        args.endpoint,
        "--model-id",
        args.model_id,
        "--prompt-file",
        args.prompt_file,
        "--output-dir",
        args.output_dir,
        "--api-key-env",
        args.api_key_env,
        "--request-timeout",
        str(args.request_timeout),
        "--retained-event-limit",
        str(args.retained_event_limit),
    ]
    for flag, value in (
        ("--model-revision", args.model_revision),
        ("--system-prompt-file", args.system_prompt_file),
        ("--max-output-tokens", args.max_output_tokens),
        ("--temperature", args.temperature),
        ("--top-p", args.top_p),
        ("--seed", args.seed),
        ("--reasoning-effort", args.reasoning_effort),
        ("--thinking", args.thinking),
        ("--clear-thinking", args.clear_thinking),
        ("--provider-request-id", args.provider_request_id),
    ):
        if value is not None:
            reconstructed.extend((flag, str(value)))
    if args.dry_run:
        reconstructed.append("--dry-run")
    return tuple(reconstructed)


def _api_collection_config(args: argparse.Namespace) -> APICollectionConfig:
    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )
    return APICollectionConfig(
        run_id=args.run_id,
        provider=args.provider,
        endpoint=args.endpoint,
        model_id=args.model_id,
        prompt=Path(args.prompt_file).read_text(encoding="utf-8"),
        output_dir=Path(args.output_dir),
        command_argv=_collect_api_argv(args),
        credential_env_var=args.api_key_env,
        system_prompt=system_prompt,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        request_timeout_seconds=args.request_timeout,
        retained_event_limit=args.retained_event_limit,
        extensions=ProviderExtensions(
            reasoning_effort=args.reasoning_effort,
            thinking_type=args.thinking,
            clear_thinking=(
                None if args.clear_thinking is None else args.clear_thinking == "true"
            ),
            provider_request_id=args.provider_request_id,
        ),
        model_revision=args.model_revision,
    )


def _api_detail(args: argparse.Namespace, exc: BaseException) -> str:
    """Render a collector diagnostic with no caller-supplied value in it.

    Two independent scrubs, because they cover different gaps. The
    credential scrub catches a key that reached the message through the
    endpoint, and it is keyed on the resolved environment value, so it does
    nothing when the caller put the credential in the ``--api-key-env``
    name slot and no such variable exists. The argv scrub covers exactly
    that case, because it replaces every token the caller supplied
    regardless of whether it names anything.
    """
    credential = os.environ.get(args.api_key_env, "").strip() or None
    return _scrub_argv_values(redact_text_for_dry_run(str(exc), credential))


def _cmd_collect_api(args: argparse.Namespace) -> int:
    try:
        config = _api_collection_config(args)
    except (OSError, UnicodeError, OpenAIStreamCollectorError) as exc:
        # Config failure is the most likely moment for a key pasted into the
        # endpoint to surface, so the diagnostic is scrubbed before it is
        # printed even though no request was attempted.
        print(
            f"Failed to configure API collection: {_api_detail(args, exc)}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        credential = os.environ.get(config.credential_env_var, "")
        try:
            assert_credential_not_embedded(config, os.environ)
        except OpenAIStreamCollectorError as exc:
            print(
                f"Failed to configure API collection: {_api_detail(args, exc)}",
                file=sys.stderr,
            )
            return 1
        plan = build_request_plan(config, environ=os.environ)
        payload = {
            "dry_run": True,
            "network_request_performed": False,
            # The plan decides what may be written down. A name the
            # environment does not define was never proven to be a name and
            # may be the credential, so the plan masks it and the payload
            # follows rather than reaching around it to the raw config.
            "credential_env_var": plan.credential_env_var,
            "credential_env_var_present": bool(credential.strip()),
            "plan": plan.to_dict(),
        }
        text = json.dumps(payload, indent=2, allow_nan=False)
        # Defence in depth behind the check above: the rendered document is
        # scrubbed rather than trusting every field to have been sanitized
        # individually.
        text = redact_text_for_dry_run(text, credential.strip() or None)
        try:
            atomic_write_text(config.output_dir / "request_plan.json", text + "\n")
        except OSError as exc:
            print(
                f"Failed to write request plan: {_api_detail(args, exc)}",
                file=sys.stderr,
            )
            return 1
        print(text)
        return 0

    try:
        result = collect_openai_stream(config, transport=UrllibStreamingTransport())
    except (OSError, OpenAIStreamCollectorError) as exc:
        print(
            f"Failed to collect API evidence: {_api_detail(args, exc)}", file=sys.stderr
        )
        return 1

    if result.record.outcome.success:
        print(f"API experiment record written to {config.output_dir / 'record.json'}")
        return 0
    failure = result.evidence.failure
    detail = (
        "unknown failure"
        if failure is None
        else f"{failure.category}: {failure.message}"
    )
    print(f"API collection failed: {_scrub_argv_values(detail)}", file=sys.stderr)
    return 1


def _cmd_native_mtp_capability_report(args: argparse.Namespace) -> int:
    try:
        report = capability_report_for_target(Path(args.target_model_path))
    except NativeMTPCollectorError as exc:
        print(f"Failed to determine native-MTP capability: {exc}", file=sys.stderr)
        return 1

    if args.output:
        report.write_json(args.output)
        print(f"Native-MTP capability report written to {args.output}")
    else:
        print(report.to_json())
    return 0 if report.supported else 3


def _native_mtp_collect_argv(args: argparse.Namespace) -> tuple[str, ...]:
    invocation = getattr(args, "_invocation", None)
    if invocation is not None:
        return tuple(invocation)

    reconstructed = [
        "llmtracefx-optimizer",
        "native-mtp",
        "collect",
        "--run-id",
        args.run_id,
        "--target-model-path",
        args.target_model_path,
        "--mtp-sidecar-path",
        args.mtp_sidecar_path,
        "--model-id",
        args.model_id,
        "--prompt-file",
        args.prompt_file,
        "--output-dir",
        args.output_dir,
        "--max-tokens",
        str(args.max_tokens),
        "--seed",
        str(args.seed),
        "--configured-depth",
        str(args.configured_depth),
    ]
    for flag, value in (
        ("--model-revision", args.model_revision),
        ("--tokenizer-revision", args.tokenizer_revision),
        ("--quantization", args.quantization),
        ("--accelerator", args.accelerator),
    ):
        if value is not None:
            reconstructed.extend((flag, value))
    return tuple(reconstructed)


def _cmd_native_mtp_collect(args: argparse.Namespace) -> int:
    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        config = NativeMTPCollectionConfig(
            run_id=args.run_id,
            target_model_path=Path(args.target_model_path),
            mtp_sidecar_path=Path(args.mtp_sidecar_path),
            model_id=args.model_id,
            prompt=prompt,
            output_dir=Path(args.output_dir),
            command_argv=_native_mtp_collect_argv(args),
            max_tokens=args.max_tokens,
            seed=args.seed,
            configured_depth=args.configured_depth,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            quantization=args.quantization,
            accelerator=args.accelerator,
        )
        result = collect_native_mtp(config, runtime=None)
    except (OSError, UnicodeError, NativeMTPCollectorError) as exc:
        print(f"Failed to collect native-MTP evidence: {exc}", file=sys.stderr)
        return 1

    record_path = config.output_dir / "record.json"
    print(f"Native-MTP experiment record written to {record_path}")
    if result.record.outcome.success:
        return 0
    error_message = (
        result.record.error.message
        if result.record.error is not None
        else "unknown failure"
    )
    print(
        f"Native-MTP collection did not run (capability unsupported): {error_message}",
        file=sys.stderr,
    )
    print(
        f"See {config.output_dir / 'capability_report.json'} for details",
        file=sys.stderr,
    )
    return 1


def _cmd_parse_llama_cpp(args: argparse.Namespace) -> int:
    stdout_text = (
        Path(args.stdout_file).read_text(encoding="utf-8") if args.stdout_file else ""
    )
    stderr_text = (
        Path(args.stderr_file).read_text(encoding="utf-8") if args.stderr_file else ""
    )
    if not stdout_text and not stderr_text:
        print("Provide at least one of --stdout-file/--stderr-file", file=sys.stderr)
        return 1

    command_argv = args.llama_command
    if command_argv and command_argv[0] == "--":
        command_argv = command_argv[1:]
    if not command_argv:
        print(
            "Provide the executed command after a literal '--', e.g. "
            "`... --run-id x -- llama-cli -m model.gguf`",
            file=sys.stderr,
        )
        return 1
    command_argv = _redact_credential_flag_values(command_argv)

    try:
        record = build_experiment_record(
            run_id=args.run_id,
            started_at=utc_now_iso(),
            platform=_platform_from_manifest(accelerator=args.accelerator),
            model=ModelInfo(
                model_id=args.model_id,
                model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision,
                quantization=args.quantization,
            ),
            command=CommandInfo(argv=tuple(command_argv)),
            repetition=RepetitionInfo(
                warmup_repetitions=args.warmup_repetitions,
                measured_repetitions=args.measured_repetitions,
                repetition_index=args.repetition_index,
            ),
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            runtime_version=args.runtime_version,
            runtime_git_revision=args.runtime_git_revision,
            speculative_method=args.speculative_method,
        )
    except (LlamaCppParseError, SchemaValidationError) as exc:
        print(f"Failed to parse llama.cpp output: {exc}", file=sys.stderr)
        return 1

    if args.output:
        record.write_json(args.output)
        print(f"Experiment record written to {args.output}")
    else:
        print(record.to_json())
    return 0


def _cmd_doctor_speculative(args: argparse.Namespace) -> int:
    try:
        baseline_records = [ExperimentRecord.read_json(path) for path in args.baseline]
        speculative_records = [
            ExperimentRecord.read_json(path) for path in args.speculative
        ]
    except SchemaValidationError as exc:
        print(f"Invalid ExperimentRecord input: {exc}", file=sys.stderr)
        return 1

    report = diagnose_speculative_regression(
        baseline_records,
        speculative_records,
        min_repetitions=args.min_repetitions,
        relative_threshold=args.relative_threshold,
    )
    payload = asdict(report)
    payload["verdict"] = report.verdict.value
    print(json.dumps(payload, indent=2))
    return 0 if report.verdict.value != "inconclusive" else 2


def _cmd_workloads_list(args: argparse.Namespace) -> int:
    for workload in WORKLOADS:
        print(f"{workload.workload_id}\t{workload.category.value}\tv{workload.version}")
    return 0


def _cmd_workloads_generate_matrix(args: argparse.Namespace) -> int:
    context_tiers = (
        tuple(ContextTier(value) for value in args.context_tiers)
        if args.context_tiers
        else tuple(ContextTier)
    )
    manifest = generate_matrix(
        model_id=args.model_id,
        model_family=args.model_family,
        output_dir=args.output_dir,
        target_model_path=args.target_model_path,
        mtp_sidecar_path=args.mtp_sidecar_path,
        context_tiers=context_tiers,
        max_tokens=args.max_tokens,
    )
    write_matrix(manifest)

    runnable = sum(1 for entry in manifest.entries if entry.runnable)
    print(
        f"Wrote {len(manifest.entries)} planned matrix entries "
        f"({runnable} runnable) to {args.output_dir}/manifest.json. "
        "No model was loaded or downloaded."
    )
    return 0


def _cmd_workloads_evaluate(args: argparse.Namespace) -> int:
    try:
        workload = workload_by_id(args.workload_id)
    except KeyError as exc:
        print(f"Unknown workload: {exc}", file=sys.stderr)
        return 1

    response_text = Path(args.response_file).read_text(encoding="utf-8")
    try:
        outcome = evaluate_workload(workload, response_text)
    except WorkloadSchemaError as exc:
        print(f"Failed to evaluate workload: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "success": outcome.success,
                "quality_score": outcome.quality_score,
                "quality_metric": outcome.quality_metric,
                "notes": outcome.notes,
            },
            indent=2,
        )
    )
    return 0 if outcome.success else 1


def _row_selection_from_args(args: argparse.Namespace) -> RowSelection:
    return RowSelection(
        run_ids=frozenset(args.run_id) if args.run_id else None,
        categories=frozenset(args.category) if args.category else None,
        context_tiers=frozenset(args.context_tier) if args.context_tier else None,
        decode_modes=frozenset(args.mode) if args.mode else None,
    )


def _optional_run_binding(args: argparse.Namespace) -> RunBinding | None:
    """Best-effort binding for dry-run reporting: ``None`` on any problem.

    Invalid/missing paths are surfaced as per-row blockers by
    ``plan_selected_rows`` rather than aborting the dry run outright.
    """
    if not args.model_path:
        return None
    try:
        return RunBinding(
            target_model_path=Path(args.model_path),
            draft_model_path=(
                Path(args.draft_model_path) if args.draft_model_path else None
            ),
            seed=args.seed,
            num_draft_tokens=args.num_draft_tokens,
        )
    except VerifyError:
        return None


def _cmd_workloads_run(args: argparse.Namespace) -> int:
    matrix_path = Path(args.matrix)
    try:
        manifest = MatrixManifest.read_json(matrix_path)
    except (OSError, MatrixSchemaError) as exc:
        print(f"Failed to load matrix manifest: {exc}", file=sys.stderr)
        return 1

    selection = _row_selection_from_args(args)
    output_dir = Path(args.output_dir)
    manifest_dir = matrix_path.parent

    if args.dry_run:
        plans = plan_selected_rows(
            manifest,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            selection=selection,
            binding=_optional_run_binding(args),
        )
        if not plans:
            print("No matrix rows matched the selection filters", file=sys.stderr)
            return 1
        for plan in plans:
            label = (
                "UNSUPPORTED"
                if plan.unsupported
                else ("READY" if plan.ready else "BLOCKED")
            )
            print(
                f"[{label}] {plan.entry.run_id} "
                f"({plan.entry.decode_mode}, {plan.entry.context_tier})"
            )
            print(f"    prompt file: {plan.prompt_path}")
            print(f"    collection dir: {plan.collection_dir}")
            print(f"    final record: {plan.final_record_path}")
            if plan.unsupported:
                print(f"    unsupported reason: {plan.unsupported_reason}")
            for blocker in plan.blockers:
                print(f"    blocker: {blocker}")

        blocked = sum(1 for plan in plans if not plan.unsupported and not plan.ready)
        unsupported = sum(1 for plan in plans if plan.unsupported)
        print(
            f"{len(plans)} row(s) selected, {blocked} blocked, "
            f"{unsupported} unsupported; no model was loaded or downloaded"
        )
        return 0 if blocked == 0 else 2

    if not args.model_path:
        print("--model-path is required unless --dry-run is set", file=sys.stderr)
        return 1

    try:
        binding = RunBinding(
            target_model_path=Path(args.model_path),
            draft_model_path=(
                Path(args.draft_model_path) if args.draft_model_path else None
            ),
            seed=args.seed,
            num_draft_tokens=args.num_draft_tokens,
        )
    except VerifyError as exc:
        print(f"Invalid model path binding: {exc}", file=sys.stderr)
        return 1

    results = run_selected_rows(
        manifest,
        manifest_dir=manifest_dir,
        output_dir=output_dir,
        selection=selection,
        binding=binding,
        resume=not args.no_resume,
        runtime_factory=MLXLMRuntime,
    )
    if not results:
        print("No matrix rows matched the selection filters", file=sys.stderr)
        return 1

    for result in results:
        status = result.verification.status.value.upper()
        suffix = f": {result.verification.reason}" if result.verification.reason else ""
        print(f"[{status}] {result.entry.run_id}{suffix}")

    failed = sum(1 for r in results if r.verification.status == RowStatus.FAILED)
    inconclusive = sum(
        1 for r in results if r.verification.status == RowStatus.INCONCLUSIVE
    )
    print(f"Artifacts written to {output_dir}")
    if failed:
        return 1
    if inconclusive:
        return 2
    return 0


def _cmd_workloads_summarize(args: argparse.Namespace) -> int:
    results_dir = Path(args.results)
    summary = summarize_results(results_dir)
    if args.output:
        write_summary(summary, Path(args.output))
        print(f"Verification summary written to {args.output}")
    else:
        print(summary.to_json())
    return 0 if summary.overall.total > 0 else 1


def _cmd_tune(args: argparse.Namespace) -> int:
    try:
        policy = TunePolicy.from_file(args.policy)
    except (OSError, UnicodeError, TunePolicyError) as exc:
        print(f"Invalid tune policy: {exc}", file=sys.stderr)
        return 1

    results_dirs = tuple(Path(path) for path in args.results)
    try:
        report = tune(results_dirs=results_dirs, policy=policy)
    except TuneInputError as exc:
        print(f"Invalid tuning input: {exc}", file=sys.stderr)
        return 1

    print(format_report_text(report, verbose=args.explain))

    if args.output:
        atomic_write_text(Path(args.output), report.to_json() + "\n")
        print(f"\nTune report written to {args.output}")

    if not report.groups:
        print(
            "\nNo comparable groups were found across the given results "
            "directories.",
            file=sys.stderr,
        )
        return 1
    return 0 if report.has_recommendation else 2


def _cmd_tune_report(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    try:
        report = TuneReport.read_json(input_path)
    except (OSError, UnicodeError) as exc:
        print(f"Could not read tune report input {input_path}: {exc}", file=sys.stderr)
        return 1
    except TuneReportValidationError as exc:
        print(f"Invalid tune report in {input_path}: {exc}", file=sys.stderr)
        return 1

    html_document = render_tune_report_html(report, redact_paths=not args.include_paths)

    output_path = Path(args.output)
    try:
        atomic_write_text(output_path, html_document)
    except OSError as exc:
        print(
            f"Could not write tune report HTML to {output_path}: {exc}", file=sys.stderr
        )
        return 1

    print(f"Tune report HTML written to {output_path}")
    if args.include_paths:
        print("Full local artifact paths are included (--include-paths was set).")
    else:
        print(
            "Local artifact paths were redacted to basenames/stable labels; "
            "rerun with --include-paths to include full paths."
        )
    return 0


def _exit_code_severity(code: int) -> int:
    """Rank exit codes so a hard failure always outranks an inconclusive one.

    Mirrors the precedence already implied by every other subcommand in
    this CLI: ``1`` (invalid input / hard failure) is worse than ``2``
    (evidence collected but inconclusive), which is worse than ``0``.
    """
    if code == 1:
        return 2
    if code == 2:
        return 1
    return 0


def _combine_exit_codes(*codes: int) -> int:
    return max(codes, key=_exit_code_severity)


def _resolve_optimize_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    output_dir = Path(args.results).expanduser().resolve(strict=False)
    return {
        "matrix": Path(args.matrix).expanduser().resolve(strict=False),
        "policy": Path(args.policy).expanduser().resolve(strict=False),
        "report-json": (
            Path(args.report_json).expanduser().resolve(strict=False)
            if args.report_json
            else None
        ),
        "report-html": (
            Path(args.report_html).expanduser().resolve(strict=False)
            if args.report_html
            else None
        ),
        "summary-json": (
            Path(args.summary_json).expanduser().resolve(strict=False)
            if args.summary_json
            else output_dir / "optimize_summary.json"
        ),
        "results": output_dir,
    }


def _filesystem_is_case_insensitive(path: Path) -> bool:
    """Detect case folding read-only using the nearest existing ancestor."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    while probe != probe.parent:
        name = probe.name
        for index, character in enumerate(name):
            if not character.isalpha():
                continue
            swapped = character.swapcase()
            if swapped == character:
                continue
            alias = probe.with_name(name[:index] + swapped + name[index + 1 :])
            try:
                return alias.exists() and alias.samefile(probe)
            except OSError:
                return False
        probe = probe.parent
    return False


def _paths_alias(left: Path, right: Path) -> bool:
    left = left.expanduser().resolve(strict=False)
    right = right.expanduser().resolve(strict=False)
    if left == right:
        return True
    if left.exists() and right.exists():
        try:
            if left.samefile(right):
                return True
        except OSError:
            pass
    return str(left).casefold() == str(
        right
    ).casefold() and _filesystem_is_case_insensitive(left)


def _validate_optimize_paths(paths: dict[str, Path | None]) -> str | None:
    """Reject aliases that could overwrite optimize inputs or artifacts."""
    configured: list[tuple[str, Path]] = []
    for label in ("matrix", "policy", "report-json", "report-html", "summary-json"):
        path = paths[label]
        if path is not None:
            configured.append((label, path))
    for (left_label, left), (right_label, right) in combinations(configured, 2):
        if _paths_alias(left, right):
            return (
                "optimize paths must be distinct; "
                f"{left_label} and {right_label} alias {left}"
            )
    return None


def _validate_optimize_plan_paths(
    paths: dict[str, Path | None], plans: tuple[RowPlan, ...]
) -> str | None:
    """Prevent optimize inputs/outputs from aliasing selected-row artifacts."""
    configured = [
        (label, path)
        for label in (
            "matrix",
            "policy",
            "report-json",
            "report-html",
            "summary-json",
        )
        if (path := paths[label]) is not None
    ]
    for plan in plans:
        artifacts = (
            ("prompt", plan.prompt_path),
            ("verification", plan.verification_path),
            ("final record", plan.final_record_path),
            ("collector record", plan.collection_dir / "record.json"),
            ("response", plan.collection_dir / "response.txt"),
            ("environment", plan.collection_dir / "environment.json"),
        )
        for label, path in configured:
            for artifact_label, artifact_path in artifacts:
                if _paths_alias(path, artifact_path):
                    return (
                        f"optimize {label} path aliases selected run "
                        f"{plan.entry.run_id!r} {artifact_label} artifact: {path}"
                    )
    return None


def _cmd_optimize(args: argparse.Namespace) -> int:
    """Compose row execution, offline tuning, and report rendering.

    Reuses ``workloads.verify.run_selected_rows``/``plan_selected_rows``,
    ``tune.tuner.tune``, and ``tune.report_html.render_tune_report_html``
    exactly as the standalone ``workloads run``/``tune``/``tune-report``
    subcommands do; nothing here duplicates their collection, evaluation,
    tuning, or rendering logic.
    """
    paths = _resolve_optimize_paths(args)
    path_error = _validate_optimize_paths(paths)
    if path_error is not None:
        print(path_error, file=sys.stderr)
        return 1

    matrix_path = paths["matrix"]
    policy_path = paths["policy"]
    output_dir = paths["results"]
    report_json_path = paths["report-json"]
    report_html_path = paths["report-html"]
    summary_path = paths["summary-json"]
    assert matrix_path is not None
    assert policy_path is not None
    assert output_dir is not None
    assert summary_path is not None
    try:
        manifest = MatrixManifest.read_json(matrix_path)
    except (OSError, MatrixSchemaError) as exc:
        print(f"Failed to load matrix manifest: {exc}", file=sys.stderr)
        return 1

    try:
        policy = TunePolicy.from_file(policy_path)
    except (OSError, UnicodeError, TunePolicyError) as exc:
        print(f"Invalid tune policy: {exc}", file=sys.stderr)
        return 1

    selection = _row_selection_from_args(args)
    manifest_dir = matrix_path.parent

    if args.dry_run:
        plans = plan_selected_rows(
            manifest,
            manifest_dir=manifest_dir,
            output_dir=output_dir,
            selection=selection,
            binding=_optional_run_binding(args),
        )
        if not plans:
            print("No matrix rows matched the selection filters", file=sys.stderr)
            return 1
        plan_path_error = _validate_optimize_plan_paths(paths, plans)
        if plan_path_error is not None:
            print(plan_path_error, file=sys.stderr)
            return 1

        for plan in plans:
            label = (
                "UNSUPPORTED"
                if plan.unsupported
                else ("READY" if plan.ready else "BLOCKED")
            )
            print(
                f"[{label}] {plan.entry.run_id} "
                f"({plan.entry.decode_mode}, {plan.entry.context_tier})"
            )
            print(f"    prompt file: {plan.prompt_path}")
            print(f"    collection dir: {plan.collection_dir}")
            print(f"    final record: {plan.final_record_path}")
            if plan.unsupported:
                print(f"    unsupported reason: {plan.unsupported_reason}")
            for blocker in plan.blockers:
                print(f"    blocker: {blocker}")

        blocked = sum(1 for plan in plans if not plan.unsupported and not plan.ready)
        unsupported = sum(1 for plan in plans if plan.unsupported)
        print(
            f"{len(plans)} row(s) selected, {blocked} blocked, "
            f"{unsupported} unsupported"
        )
        print(
            f"Planned tune policy objective: {policy.objective.value} ({policy_path})"
        )
        if report_json_path is not None:
            print(f"Planned tune report JSON: {report_json_path}")
        if report_html_path is not None:
            print(f"Planned tune report HTML: {report_html_path}")
        exit_code = 0 if blocked == 0 else 2
        overall_status = (
            OverallStatus.SUCCESS if exit_code == 0 else OverallStatus.INCONCLUSIVE
        )
        planned_detail = (
            f"{len(plans)} row(s) selected, {blocked} blocked, "
            f"{unsupported} unsupported"
        )
        summary = OptimizeSummary(
            schema_version=OPTIMIZE_SUMMARY_SCHEMA_VERSION,
            generated_at=utc_now_iso(),
            dry_run=True,
            matrix_path=str(matrix_path),
            results_dir=str(output_dir),
            extra_results_dirs=tuple(args.extra_results or ()),
            policy_path=str(policy_path),
            report_json_path=(
                str(report_json_path) if report_json_path is not None else None
            ),
            report_html_path=(
                str(report_html_path) if report_html_path is not None else None
            ),
            phases=(
                PhaseReport(
                    name=PhaseName.PLANNED,
                    status=PhaseStatus.OK,
                    detail=planned_detail,
                ),
                PhaseReport(name=PhaseName.EXECUTED, status=PhaseStatus.NOT_RUN),
                PhaseReport(name=PhaseName.VERIFIED, status=PhaseStatus.NOT_RUN),
                PhaseReport(name=PhaseName.TUNED, status=PhaseStatus.NOT_RUN),
                PhaseReport(name=PhaseName.RENDERED, status=PhaseStatus.NOT_RUN),
            ),
            row_counts=RowStatusCounts(
                total=len(plans),
                ready=sum(1 for plan in plans if plan.ready and not plan.unsupported),
                blocked=blocked,
                unsupported=unsupported,
            ),
            recommendations=(),
            overall_status=overall_status,
            exit_code=exit_code,
        )
        try:
            atomic_write_text(summary_path, summary.to_json() + "\n")
            OptimizeSummary.read_json(summary_path)
        except (OSError, OptimizeSummaryValidationError) as exc:
            print(
                f"Failed to write/validate orchestration summary: {exc}",
                file=sys.stderr,
            )
            return 1
        print(
            "No model was loaded or downloaded, no results were tuned, and "
            "no tune report was written (--dry-run)"
        )
        print(f"Dry-run orchestration summary written to {summary_path}")
        return exit_code

    if report_json_path is None:
        print("--report-json is required unless --dry-run is set", file=sys.stderr)
        return 1
    if not args.model_path:
        print("--model-path is required unless --dry-run is set", file=sys.stderr)
        return 1

    try:
        binding = RunBinding(
            target_model_path=Path(args.model_path),
            draft_model_path=(
                Path(args.draft_model_path) if args.draft_model_path else None
            ),
            seed=args.seed,
            num_draft_tokens=args.num_draft_tokens,
        )
    except VerifyError as exc:
        print(f"Invalid model path binding: {exc}", file=sys.stderr)
        return 1

    plans = plan_selected_rows(
        manifest,
        manifest_dir=manifest_dir,
        output_dir=output_dir,
        selection=selection,
        binding=binding,
    )
    if not plans:
        print("No matrix rows matched the selection filters", file=sys.stderr)
        return 1
    plan_path_error = _validate_optimize_plan_paths(paths, plans)
    if plan_path_error is not None:
        print(plan_path_error, file=sys.stderr)
        return 1

    results = run_selected_rows(
        manifest,
        manifest_dir=manifest_dir,
        output_dir=output_dir,
        selection=selection,
        binding=binding,
        resume=not args.no_resume,
        runtime_factory=MLXLMRuntime,
    )
    if not results:
        print("No matrix rows matched the selection filters", file=sys.stderr)
        return 1

    for result in results:
        status = result.verification.status.value.upper()
        suffix = f": {result.verification.reason}" if result.verification.reason else ""
        print(f"[{status}] {result.entry.run_id}{suffix}")
    print(f"Artifacts written to {output_dir}")

    counts = RowStatusCounts(
        total=len(results),
        ready=sum(
            1
            for result in results
            if result.verification.status != RowStatus.UNSUPPORTED
        ),
        completed=sum(
            1 for r in results if r.verification.status == RowStatus.COMPLETED
        ),
        skipped=sum(1 for r in results if r.verification.status == RowStatus.SKIPPED),
        failed=sum(1 for r in results if r.verification.status == RowStatus.FAILED),
        unsupported=sum(
            1 for r in results if r.verification.status == RowStatus.UNSUPPORTED
        ),
        inconclusive=sum(
            1 for r in results if r.verification.status == RowStatus.INCONCLUSIVE
        ),
    )

    if counts.failed:
        executed_status = PhaseStatus.FAILED
    elif counts.inconclusive:
        executed_status = PhaseStatus.INCONCLUSIVE
    else:
        executed_status = PhaseStatus.OK

    trustable = counts.completed + counts.skipped
    if counts.failed:
        verified_status = PhaseStatus.FAILED
    elif counts.inconclusive:
        verified_status = PhaseStatus.INCONCLUSIVE
    elif trustable == 0:
        verified_status = PhaseStatus.UNSUPPORTED
    else:
        verified_status = PhaseStatus.OK

    extra_results = tuple(Path(path) for path in (args.extra_results or ()))
    results_dirs = (output_dir, *extra_results)

    tune_report: TuneReport | None = None
    tune_detail: str | None = None
    try:
        tune_report = tune(
            results_dirs=results_dirs,
            policy=policy,
            primary_run_ids=frozenset(result.entry.run_id for result in results),
        )
    except TuneInputError as exc:
        tune_detail = str(exc)
        print(f"Invalid tuning input: {tune_detail}", file=sys.stderr)

    if tune_report is not None:
        print(format_report_text(tune_report, verbose=args.explain))
        if not tune_report.groups:
            tuned_status = PhaseStatus.FAILED
            tune_detail = (
                "no comparable groups were found across the given results "
                "directories"
            )
            print(f"\n{tune_detail}", file=sys.stderr)
        elif not tune_report.has_recommendation:
            tuned_status = PhaseStatus.INCONCLUSIVE
        else:
            tuned_status = PhaseStatus.OK
    else:
        tuned_status = PhaseStatus.FAILED

    recommendations: list[RecommendedCandidate] = []
    if tune_report is not None:
        for group in tune_report.groups:
            if (
                group.outcome == GroupOutcome.RECOMMENDED
                and group.recommended is not None
            ):
                recommendations.append(
                    RecommendedCandidate(
                        group_label=group.group_key.label(),
                        run_ids=group.recommended.run_ids,
                        objective_name=group.recommended.objective_name,
                        objective_value=group.recommended.objective_value,
                    )
                )

    rendered_status = PhaseStatus.SKIPPED
    rendered_detail: str | None = None
    if tune_report is not None:
        try:
            atomic_write_text(report_json_path, tune_report.to_json() + "\n")
            TuneReport.read_json(report_json_path)
        except (OSError, TuneReportValidationError) as exc:
            rendered_status = PhaseStatus.FAILED
            rendered_detail = f"failed to write/validate tune report JSON: {exc}"
            print(rendered_detail, file=sys.stderr)
        else:
            print(f"Tune report JSON written to {report_json_path}")
            rendered_status = PhaseStatus.OK

        if rendered_status == PhaseStatus.OK and report_html_path is not None:
            try:
                html_document = render_tune_report_html(
                    tune_report, redact_paths=not args.include_paths
                )
                atomic_write_text(report_html_path, html_document)
            except OSError as exc:
                rendered_status = PhaseStatus.FAILED
                rendered_detail = f"failed to write tune report HTML: {exc}"
                print(rendered_detail, file=sys.stderr)
            else:
                print(f"Tune report HTML written to {report_html_path}")
    else:
        rendered_detail = "tuning failed; nothing to render"

    exec_code = 1 if counts.failed else (2 if counts.inconclusive else 0)
    tune_code = (
        1
        if tuned_status == PhaseStatus.FAILED
        else 2 if tuned_status == PhaseStatus.INCONCLUSIVE else 0
    )
    rendered_code = 1 if rendered_status == PhaseStatus.FAILED else 0
    exit_code = _combine_exit_codes(exec_code, tune_code, rendered_code)

    if exit_code == 0:
        overall_status = OverallStatus.SUCCESS
    elif exit_code == 2:
        overall_status = OverallStatus.INCONCLUSIVE
    else:
        overall_status = OverallStatus.FAILED

    summary = OptimizeSummary(
        schema_version=OPTIMIZE_SUMMARY_SCHEMA_VERSION,
        generated_at=utc_now_iso(),
        dry_run=False,
        matrix_path=str(matrix_path),
        results_dir=str(output_dir),
        extra_results_dirs=tuple(str(path) for path in extra_results),
        policy_path=str(policy_path),
        report_json_path=str(report_json_path),
        report_html_path=str(report_html_path) if report_html_path else None,
        phases=(
            PhaseReport(name=PhaseName.PLANNED, status=PhaseStatus.OK),
            PhaseReport(name=PhaseName.EXECUTED, status=executed_status),
            PhaseReport(name=PhaseName.VERIFIED, status=verified_status),
            PhaseReport(name=PhaseName.TUNED, status=tuned_status, detail=tune_detail),
            PhaseReport(
                name=PhaseName.RENDERED,
                status=rendered_status,
                detail=rendered_detail,
            ),
        ),
        row_counts=counts,
        recommendations=tuple(recommendations),
        overall_status=overall_status,
        exit_code=exit_code,
    )

    try:
        atomic_write_text(summary_path, summary.to_json() + "\n")
        OptimizeSummary.read_json(summary_path)
    except (OSError, OptimizeSummaryValidationError) as exc:
        print(f"Failed to write/validate orchestration summary: {exc}", file=sys.stderr)
        # The summary itself is the source of truth for what happened; if it
        # cannot be trusted on disk, the whole invocation must be reported
        # as a hard failure regardless of how the individual phases went.
        return 1

    print(f"Orchestration summary written to {summary_path}")
    return exit_code


_MIN_SCRUBBED_ARGUMENT_CHARS = 4

_CREDENTIAL_ARGUMENT_STEMS = frozenset(
    {
        "apikey",
        "apikeys",
        "apisecret",
        "apitoken",
        "accesskey",
        "accesstoken",
        "auth",
        "authorization",
        "authtoken",
        "bearer",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "key",
        "password",
        "passwd",
        "pwd",
        "secret",
        "secretkey",
        "token",
    }
)

_GLUED_CREDENTIAL_FLAGS = tuple(
    sorted(
        {
            "--access-key",
            "--access-token",
            "--api-key",
            "--api-secret",
            "--api-token",
            "--api_key",
            "--apikey",
            "--auth",
            "--auth-token",
            "--authorization",
            "--bearer",
            "--bearer-token",
            "--client-secret",
            "--credential",
            "--key",
            "--passwd",
            "--password",
            "--pwd",
            "--secret",
            "--secret-key",
            "--token",
            "-key",
            "-k",
            "-p",
        },
        key=len,
        reverse=True,
    )
)

#: Scrub state for one parse, held per context rather than in module
#: globals. ``main`` used to set globals and never restore them, so a
#: later ``build_parser().parse_args(...)`` in the same process inherited
#: a populated state that did not describe its own argv and echoed the
#: value it was given. A ``ContextVar`` is restored by its token on every
#: exit path and is not shared between threads.
_scrub_state: ContextVar[tuple[tuple[str, ...], tuple[str, ...]] | None] = ContextVar(
    "llmtracefx_argv_scrub_state", default=None
)


def _option_stem(token: str) -> str:
    """Normalize ``--Api_Key=value`` to ``apikey`` for comparison.

    Spelling is not evidence of intent. A caller reaching for a credential
    flag may type it with dashes, underscores or neither, so the stem is
    compared with all of that removed and the value dropped.
    """
    name = token.split("=", 1)[0].lstrip("-")
    return "".join(character for character in name.lower() if character.isalnum())


def _redact_credential_flag_values(argv: Sequence[str]) -> tuple[str, ...]:
    """Replace the value of any credential-shaped flag in a recorded command.

    ``parse-llama-cpp`` records the command the operator actually ran, and
    ``llama-server`` has its own ``--api-key``. The flag itself is evidence
    worth keeping, the value is a credential that would otherwise be
    written verbatim into ``record.command.argv``. Both the separate and
    the attached form are covered.
    """
    redacted: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            redacted.append("[REDACTED]")
            continue
        if not _is_credential_flag(token):
            glued = _glued_credential_prefix(token)
            redacted.append(f"{glued}[REDACTED]" if glued else token)
            continue
        name, separator, _ = token.partition("=")
        if separator:
            redacted.append(f"{name}=[REDACTED]")
            continue
        following = argv[index + 1] if index + 1 < len(argv) else None
        # A credential flag always takes a value, so whatever follows is
        # redacted whatever it looks like: the base64url alphabet includes
        # ``-``, so skipping values that start like a flag would leak
        # roughly one credential in sixty four. The one token that is not a
        # value is another credential flag, which has its own value after
        # it. Consuming that would leave the real credential untouched, so
        # it is left for the next iteration to handle.
        skip_next = following is not None and not _is_credential_flag(following)
        redacted.append(token)
    return tuple(redacted)


def _glued_credential_prefix(token: str) -> str:
    """The credential flag ``token`` begins with, when it swallowed a value.

    Dropping the space is an ordinary typing slip and an ordinary shell
    habit, so ``--api-keySECRET`` reaches a recorded command as one token.
    Nothing later in the pipeline splits it, so without this the value is
    written into ``record.command.argv`` in full.

    Prefixes are explicit rather than inferred from every credential-like
    word. That keeps legitimate options such as ``--authentication-method``,
    ``--authorization-policy`` and ``--tokenizer-model`` reproducible.
    Because a delimiter is missing, the tail must also have credential
    evidence: a case boundary, a digit, a credential prefix, or a
    non-option alphabet character. Leading ``-`` and ``_`` are retained as
    evidence because both belong to common credential alphabets.
    """
    option_name = token.split("=", 1)[0]
    folded_name = option_name.casefold()
    for prefix in _GLUED_CREDENTIAL_FLAGS:
        if not folded_name.startswith(prefix.casefold()) or len(option_name) == len(
            prefix
        ):
            continue
        tail = option_name[len(prefix) :]
        body = tail.lstrip("-_")
        lowered = body.lower()
        if len(body) < 6:
            continue
        if (
            any(character.isupper() or character.isdigit() for character in body)
            or any(character in "+/=" for character in body)
            or lowered.startswith(
                ("sk-", "sk_", "ghp_", "github_pat_", "xoxb-", "xoxp-", "eyj")
            )
        ):
            return option_name[: len(prefix)]
    return ""


def _is_credential_flag(token: str) -> bool:
    return token.startswith("-") and _option_stem(token) in _CREDENTIAL_ARGUMENT_STEMS


def _argument_values(
    raw_argv: Sequence[str], literals: Sequence[str] = ()
) -> tuple[str, ...]:
    """Every caller-supplied value in ``raw_argv``, longest first.

    Token syntax is not evidence of what a token is. The property that
    matters is whether this program defined the string, and ``literals``
    answers that, so only a name this program chose is kept out of the
    scrub set. Everything else is a value that argparse would otherwise
    quote back into stderr. Longest first so a value that contains
    another is replaced whole.

    A short cluster such as ``-p<secret>`` is the attached form, so only
    the flag letter is a name and the rest is a value. That is the shape
    ``mysql -p<password>`` teaches, and it is contributed regardless of
    any ``=`` inside it, because a base64 credential ends in ``=`` and
    splitting on it first would leave nothing to redact. A long option
    with a dropped space, ``--api-key<secret>``, is not a name either;
    when a defined option is a prefix of it the tail alone is the value,
    which keeps the diagnostic actionable, and otherwise the whole token
    is treated as a value.
    """
    values: set[str] = set()

    def contribute(candidate: str) -> None:
        # Every non-empty value is collected. How much of it can be
        # replaced safely is decided per rendering in
        # ``_value_renderings``, which drops the bare form for values too
        # short to replace without mangling the surrounding message.
        if candidate:
            values.add(candidate)

    known = frozenset(literals)
    for token in raw_argv:
        if not token.startswith("-"):
            contribute(token)
            continue
        name, separator, attached = token.partition("=")
        if separator:
            contribute(attached)
        if not token.startswith("--"):
            contribute(token[2:])
            continue
        if token in known or name in known:
            continue
        prefix = next(
            (
                literal
                for literal in literals
                if len(literal) < len(token) and token.startswith(literal)
            ),
            "",
        )
        contribute(token[len(prefix) :] if prefix else token)
    return tuple(sorted(values, key=len, reverse=True))


def _value_renderings(value: str) -> tuple[str, ...]:
    """Every spelling of ``value`` an argparse message may contain.

    argparse formats the offending token with ``%r`` in several of its
    messages, not ``%s``: ``invalid choice``, ``ignored explicit
    argument`` and every ``type=`` conversion failure among them.
    ``repr`` escapes each character for which ``str.isprintable()`` is
    false, so a value carrying a trailing newline, a zero-width space, a
    non-breaking space or a backslash never appears in the message in the
    form the caller typed. Replacing only the raw form would then match
    nothing and print the value in full. The escaped body is returned
    alongside the raw form so both spellings are covered; the surrounding
    quotes are left in place because they belong to the message.
    """
    quoted = repr(value)
    if len(value) < _MIN_SCRUBBED_ARGUMENT_CHARS:
        # Too short to replace as a bare substring: a two character value
        # occurs inside ordinary words, and blanking every occurrence
        # would destroy the message that makes the error actionable.
        # argparse always quotes the offending token in the messages that
        # repeat one, so the quoted spellings are both distinctive enough
        # to replace safely and sufficient to keep the guarantee.
        return (quoted, f'"{value}"')
    return (value, quoted[1:-1])


def _scrub_short_bare_tokens(message: str, values: Sequence[str]) -> str:
    """Replace short values only when they occupy a whole diagnostic token."""
    scrubbed = message
    for value in values:
        if not value or len(value) >= _MIN_SCRUBBED_ARGUMENT_CHARS:
            continue
        scrubbed = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])",
            "[REDACTED]",
            scrubbed,
        )
    return scrubbed


def _scrub_argv_values(message: str) -> str:
    """Replace every caller-supplied value in ``message``.

    This program's own vocabulary is put beyond reach first, because a
    caller value can be a prefix of one: mistyping ``collect-ap`` would
    otherwise rewrite the valid ``collect-api`` in the list of choices and
    leave the caller without a correct spelling to copy. A literal is only
    protected when no value contains it, so a secret that happens to
    embed an option name is still replaced whole.
    """
    values, literals = _scrub_state.get() or ((), ())
    if not values:
        return message
    renderings = sorted(
        {rendering for value in values for rendering in _value_renderings(value)},
        key=len,
        reverse=True,
    )
    protected = {
        f"\x00PROTECTED{index}VALUE\x00": literal
        for index, literal in enumerate(literals)
        if not any(literal in rendering for rendering in renderings)
    }
    scrubbed = message
    for placeholder, literal in protected.items():
        scrubbed = scrubbed.replace(literal, placeholder)
    for rendering in renderings:
        scrubbed = scrubbed.replace(rendering, "[REDACTED]")
    scrubbed = _scrub_short_bare_tokens(scrubbed, values)
    for placeholder, literal in protected.items():
        scrubbed = scrubbed.replace(placeholder, literal)
    return scrubbed


def _materialize_argv(args: Iterable[str] | None) -> list[str] | None:
    """Read an argument iterable once, so two consumers both see it.

    ``ArgumentParser.parse_args`` accepts any iterable of strings, and the
    scrub scope has to read the tokens to know which values to blank. A
    one-shot iterator would be drained by whichever ran first, leaving the
    other with an empty command line: either the parse silently sees no
    arguments, or the scrub has nothing to blank and the diagnostics echo
    the values verbatim.
    """
    return None if args is None else list(args)


@contextmanager
def _argument_scrub_scope(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> Iterator[None]:
    """Install the scrub state for the duration of one parse.

    ``main`` is not the only way in: ``build_parser().parse_args(...)`` is
    public, is what a test or an embedding caller reaches for, and used to
    run with the scrub state still at its module defaults. An empty state
    makes ``_scrub_argv_values`` a no-op, so argparse echoed invalid
    choices and unrecognized arguments verbatim.

    An enclosing scope wins. Subparsers are instances of this class too and
    parse a suffix of the command line, so letting the inner parse reinstall
    the state would narrow it to the tokens that subparser happens to see.
    Nesting is detected from the context variable actually being set, not
    from it merely being non-empty, so a scope that has genuinely exited
    can never be mistaken for one still in force.
    """
    if _scrub_state.get() is not None:
        yield
        return
    literals = _parser_literals(parser)
    token = _scrub_state.set(
        (
            _argument_values(list(sys.argv[1:] if argv is None else argv), literals),
            literals,
        )
    )
    try:
        yield
    finally:
        _scrub_state.reset(token)


_N = TypeVar("_N")


class _ParsedCommand(Protocol):
    func: Callable[[argparse.Namespace], int]


class SecureArgumentParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose diagnostics never repeat a value.

    argparse quotes the offending token straight back to the caller, so a
    mistyped ``--api-key <secret>`` lands in stderr verbatim and from there
    in shell history, CI logs and screenshots. Option names and the usage
    block are kept, since they carry no caller input and are what make the
    error actionable; every value the caller supplied is replaced.
    """

    @overload
    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> argparse.Namespace: ...

    @overload
    def parse_args(self, args: Iterable[str] | None, namespace: _N) -> _N: ...

    @overload
    def parse_args(self, *, namespace: _N) -> _N: ...

    def parse_args(
        self,
        args: Iterable[str] | None = None,
        namespace: _N | None = None,
    ) -> _N | argparse.Namespace:
        # ``parse_args`` reports unrecognized arguments itself, after
        # ``parse_known_args`` has returned and its scope has been undone,
        # so the scope has to cover this call too or that one diagnostic
        # echoes the leftover tokens verbatim.
        argv = _materialize_argv(args)
        handler_argv = tuple(sys.argv[1:] if argv is None else argv)
        with _argument_scrub_scope(self, argv):
            parsed = super().parse_args(argv, namespace)
            handler = getattr(parsed, "func", None)
            if callable(handler):
                command = cast(_ParsedCommand, parsed)
                command.func = _scoped_command_handler(self, handler_argv, handler)
            return parsed

    @overload
    def parse_known_args(
        self,
        args: Iterable[str] | None = None,
        namespace: None = None,
    ) -> tuple[argparse.Namespace, list[str]]: ...

    @overload
    def parse_known_args(
        self, args: Iterable[str] | None, namespace: _N
    ) -> tuple[_N, list[str]]: ...

    @overload
    def parse_known_args(self, *, namespace: _N) -> tuple[_N, list[str]]: ...

    def parse_known_args(
        self,
        args: Iterable[str] | None = None,
        namespace: _N | None = None,
    ) -> tuple[_N | argparse.Namespace, list[str]]:
        argv = _materialize_argv(args)
        handler_argv = tuple(sys.argv[1:] if argv is None else argv)
        with _argument_scrub_scope(self, argv):
            parsed, remaining = super().parse_known_args(argv, namespace)
            handler = getattr(parsed, "func", None)
            if callable(handler):
                command = cast(_ParsedCommand, parsed)
                command.func = _scoped_command_handler(self, handler_argv, handler)
            return parsed, remaining

    def error(self, message: str) -> NoReturn:
        super().error(_scrub_argv_values(message))


def _scoped_command_handler(
    parser: argparse.ArgumentParser,
    raw_argv: tuple[str, ...],
    handler: Callable[[argparse.Namespace], int],
) -> Callable[[argparse.Namespace], int]:
    """Keep caller-value scrubbing installed while a parsed handler runs."""

    def run(args: argparse.Namespace) -> int:
        with _argument_scrub_scope(parser, raw_argv):
            return handler(args)

    return run


def _reject_credential_arguments(raw_argv: Sequence[str]) -> None:
    """Refuse a credential-bearing flag before argparse can echo its value.

    The collectors read the credential from the environment and have no
    flag that accepts one, so any such flag is a mistake. Rejecting it here
    rather than letting it fall through to "unrecognized arguments" means
    the value is never formatted into a message in the first place, and the
    caller is told where the credential actually belongs.

    Scanning stops at a bare ``--``. Everything after it belongs to a
    recorded external command, not to this program, and ``llama-server``
    has its own ``--api-key``. Refusing that would leave no way to record
    such a run and would point the caller at a flag that does not exist on
    the subcommand they used. Those values are redacted where they are
    persisted instead.
    """
    for token in raw_argv:
        if token == "--":
            return
        if not token.startswith("-"):
            continue
        if _option_stem(token) not in _CREDENTIAL_ARGUMENT_STEMS:
            continue
        name = token.split("=", 1)[0]
        print(
            f"llmtracefx-optimizer: error: {name} is not a supported option "
            "and a credential must never appear in a command line. Export "
            "the credential to an environment variable and name that "
            "variable with --api-key-env.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _parser_literals(parser: argparse.ArgumentParser) -> tuple[str, ...]:
    """Every option string and subcommand name this program defines.

    These are the parts of a diagnostic worth keeping, so they are
    collected once and put beyond the reach of the value scrub.
    """
    literals: set[str] = set()
    for action in parser._actions:
        literals.update(action.option_strings)
        choices = action.choices
        if isinstance(choices, dict):
            for name, subparser in choices.items():
                literals.add(name)
                if isinstance(subparser, argparse.ArgumentParser):
                    literals.update(_parser_literals(subparser))
        elif choices is not None:
            literals.update(choice for choice in choices if isinstance(choice, str))
    return tuple(sorted(literals, key=len, reverse=True))


def build_parser() -> argparse.ArgumentParser:
    parser = SecureArgumentParser(
        prog="llmtracefx-optimizer",
        description="Inference-optimizer foundation primitives for LLMTraceFX",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Collect a CPU-only environment manifest"
    )
    manifest_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write manifest JSON to this path instead of stdout",
    )
    manifest_parser.set_defaults(func=_cmd_manifest)

    run_parser = subparsers.add_parser(
        "run", help="Run a configured experiment (warmups + measured repetitions)"
    )
    run_parser.add_argument(
        "--config", required=True, help="Path to a JSON (or YAML) runner config"
    )
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all repetitions even if already completed",
    )
    run_parser.set_defaults(func=_cmd_run)

    collect_mlx_parser = subparsers.add_parser(
        "collect-mlx",
        help="Run one local MLX-LM inference and record normalized evidence",
    )
    collect_mlx_parser.add_argument("--run-id", required=True)
    collect_mlx_parser.add_argument(
        "--model-path",
        required=True,
        help="Existing local MLX model directory; models are never downloaded",
    )
    collect_mlx_parser.add_argument("--model-id", required=True)
    collect_mlx_parser.add_argument("--model-revision", default=None)
    collect_mlx_parser.add_argument("--tokenizer-revision", default=None)
    collect_mlx_parser.add_argument("--quantization", default=None)
    collect_mlx_parser.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 prompt file; prompt contents are hashed but not copied into the record",
    )
    collect_mlx_parser.add_argument("--output-dir", required=True)
    collect_mlx_parser.add_argument("--max-tokens", type=int, default=128)
    collect_mlx_parser.add_argument("--seed", type=int, default=0)
    collect_mlx_parser.add_argument(
        "--accelerator",
        default=None,
        help="Explicit accelerator name; otherwise MLX device_info is used",
    )
    collect_mlx_parser.add_argument(
        "--draft-model-path",
        default=None,
        help=(
            "Optional existing local MLX-LM draft model. This enables generic "
            "draft-model speculation, not native Qwen MTP."
        ),
    )
    collect_mlx_parser.add_argument("--num-draft-tokens", type=int, default=2)
    collect_mlx_parser.set_defaults(func=_cmd_collect_mlx)

    collect_api_parser = subparsers.add_parser(
        "collect-api",
        help=(
            "Stream one OpenAI-compatible chat completion and record "
            "normalized, credential-free evidence"
        ),
        # Prefix matching would let "--api-key <secret>" resolve to
        # "--api-key-env", which would then treat the secret as an
        # environment variable name and persist it.
        allow_abbrev=False,
    )
    collect_api_parser.add_argument("--run-id", required=True)
    collect_api_parser.add_argument(
        "--provider",
        default="z.ai",
        help="Short provider label recorded in evidence (never a secret)",
    )
    collect_api_parser.add_argument(
        "--endpoint",
        required=True,
        help=(
            "Full chat-completions URL, e.g. "
            "https://api.z.ai/api/paas/v4/chat/completions. Must be https for "
            "non-local hosts and must not embed credentials."
        ),
    )
    collect_api_parser.add_argument(
        "--model-id",
        required=True,
        help="Provider model ID, e.g. glm-5.3 or glm-5.3-flash",
    )
    collect_api_parser.add_argument(
        "--model-revision",
        default=None,
        help=(
            "Provider-side model build, when the provider exposes one. Hosted "
            "APIs usually do not, in which case leave it unset rather than "
            "guessing."
        ),
    )
    collect_api_parser.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 prompt file; prompt contents are hashed but not copied into artifacts",
    )
    collect_api_parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional UTF-8 system prompt file; also hashed, never copied",
    )
    collect_api_parser.add_argument("--output-dir", required=True)
    collect_api_parser.add_argument(
        "--api-key-env",
        default="ZAI_API_KEY",
        help=(
            "Name of the environment variable holding the API key. Only the "
            "name is recorded; the value is never persisted, echoed or "
            "accepted as a command argument."
        ),
    )
    collect_api_parser.add_argument("--max-output-tokens", type=int, default=None)
    collect_api_parser.add_argument("--temperature", type=float, default=None)
    collect_api_parser.add_argument("--top-p", type=float, default=None)
    collect_api_parser.add_argument("--seed", type=int, default=None)
    collect_api_parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds; no retries are performed",
    )
    collect_api_parser.add_argument(
        "--retained-event-limit",
        type=int,
        default=DEFAULT_RETAINED_EVENT_LIMIT,
        help=(
            "How many per-event timing rows to keep in the timeline. The "
            "counters, the rates and the inter-token distribution stay exact "
            "past this bound; only the individual rows stop being retained, "
            "and the record says so."
        ),
    )
    collect_api_parser.add_argument(
        "--reasoning-effort",
        choices=GLM_REASONING_EFFORT_LEVELS,
        default=None,
        help=(
            "Provider-specific reasoning budget. Z.ai documents low/high/max "
            "for glm-5.3 and glm-5.3-flash, defaulting to max when unset."
        ),
    )
    collect_api_parser.add_argument(
        "--thinking",
        choices=THINKING_TYPES,
        default=None,
        help=(
            "Provider-specific thinking.type. Z.ai documents that glm-5.3 and "
            "glm-5.3-flash accept only 'enabled'."
        ),
    )
    collect_api_parser.add_argument(
        "--clear-thinking",
        choices=("true", "false"),
        default=None,
        help=(
            "Provider-specific thinking.clear_thinking. Controls whether "
            "reasoning_content from previous turns is cleared; it does not "
            "change whether this turn thinks. Unset means the provider default."
        ),
    )
    collect_api_parser.add_argument(
        "--provider-request-id",
        default=None,
        help="Optional caller-supplied provider request ID (Z.ai body request_id)",
    )
    collect_api_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration and print the credential-free request plan "
            "without performing any network request"
        ),
    )
    collect_api_parser.set_defaults(func=_cmd_collect_api)

    native_mtp_parser = subparsers.add_parser(
        "native-mtp",
        help="Native Qwen MTP capability report / evidence collection",
    )
    native_mtp_subparsers = native_mtp_parser.add_subparsers(
        dest="native_mtp_command", required=True
    )

    capability_parser = native_mtp_subparsers.add_parser(
        "capability-report",
        help=(
            "Report whether this environment can produce trustworthy "
            "native-MTP evidence for a local checkpoint's architecture "
            "family (exit 0 if supported, 3 if not, 1 on error)"
        ),
    )
    capability_parser.add_argument(
        "--target-model-path",
        required=True,
        help="Existing local target checkpoint directory containing config.json",
    )
    capability_parser.add_argument(
        "--output",
        default=None,
        help="Write the capability report JSON to this path instead of stdout",
    )
    capability_parser.set_defaults(func=_cmd_native_mtp_capability_report)

    native_mtp_collect_parser = native_mtp_subparsers.add_parser(
        "collect",
        help=(
            "Validate a target/sidecar checkpoint pair and either collect "
            "native-MTP evidence or record an explicit unsupported result"
        ),
    )
    native_mtp_collect_parser.add_argument("--run-id", required=True)
    native_mtp_collect_parser.add_argument(
        "--target-model-path",
        required=True,
        help="Existing local target MLX checkpoint; never downloaded",
    )
    native_mtp_collect_parser.add_argument(
        "--mtp-sidecar-path",
        required=True,
        help="Existing local MTP sidecar/drafter checkpoint; never downloaded",
    )
    native_mtp_collect_parser.add_argument("--model-id", required=True)
    native_mtp_collect_parser.add_argument("--model-revision", default=None)
    native_mtp_collect_parser.add_argument("--tokenizer-revision", default=None)
    native_mtp_collect_parser.add_argument("--quantization", default=None)
    native_mtp_collect_parser.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 prompt file; prompt contents are hashed but not copied into the record",
    )
    native_mtp_collect_parser.add_argument("--output-dir", required=True)
    native_mtp_collect_parser.add_argument("--max-tokens", type=int, default=128)
    native_mtp_collect_parser.add_argument("--seed", type=int, default=0)
    native_mtp_collect_parser.add_argument("--configured-depth", type=int, default=2)
    native_mtp_collect_parser.add_argument(
        "--accelerator",
        default=None,
        help="Explicit accelerator name; otherwise left unset",
    )
    native_mtp_collect_parser.set_defaults(func=_cmd_native_mtp_collect)

    parse_parser = subparsers.add_parser(
        "parse-llama-cpp",
        help="Convert llama.cpp text output into a canonical ExperimentRecord",
    )
    parse_parser.add_argument("--run-id", required=True)
    parse_parser.add_argument("--model-id", required=True)
    parse_parser.add_argument("--model-revision", default=None)
    parse_parser.add_argument("--tokenizer-revision", default=None)
    parse_parser.add_argument("--quantization", default=None)
    parse_parser.add_argument("--runtime-version", default=None)
    parse_parser.add_argument("--runtime-git-revision", default=None)
    parse_parser.add_argument(
        "--accelerator",
        default=None,
        help=(
            "Explicit accelerator/GPU name (e.g. 'Apple M5 Pro', 'NVIDIA RTX 4090'). "
            "Takes precedence over any device name llama.cpp reports in its own "
            "output; if omitted, the parsed device hint (when present) is used "
            "instead so comparability checks can tell different accelerators apart."
        ),
    )
    parse_parser.add_argument(
        "--speculative-method",
        default=None,
        help="e.g. 'mtp' -- only used if speculative counters are present",
    )
    parse_parser.add_argument(
        "--stdout-file", default=None, help="Path to captured stdout text"
    )
    parse_parser.add_argument(
        "--stderr-file", default=None, help="Path to captured stderr text"
    )
    parse_parser.add_argument(
        "llama_command",
        nargs=argparse.REMAINDER,
        metavar="-- COMMAND [ARGS ...]",
        help=(
            "Exact command/args that were run, after a literal '--' separator, "
            "e.g. `... --run-id x -- llama-cli -m model.gguf -p prompt`"
        ),
    )
    parse_parser.add_argument("--warmup-repetitions", type=int, default=0)
    parse_parser.add_argument("--measured-repetitions", type=int, default=1)
    parse_parser.add_argument("--repetition-index", type=int, default=0)
    parse_parser.add_argument(
        "--output",
        default=None,
        help="Write the ExperimentRecord JSON to this path instead of stdout",
    )
    parse_parser.set_defaults(func=_cmd_parse_llama_cpp)

    doctor_parser = subparsers.add_parser("doctor", help="Evidence-based diagnostics")
    doctor_subparsers = doctor_parser.add_subparsers(
        dest="doctor_command", required=True
    )
    speculative_parser = doctor_subparsers.add_parser(
        "speculative",
        help="Diagnose whether speculative decoding/MTP is a net regression",
    )
    speculative_parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="ExperimentRecord JSON files for autoregressive baseline runs",
    )
    speculative_parser.add_argument(
        "--speculative",
        nargs="+",
        required=True,
        help="ExperimentRecord JSON files for speculative-decoding runs",
    )
    speculative_parser.add_argument("--min-repetitions", type=int, default=2)
    speculative_parser.add_argument("--relative-threshold", type=float, default=0.03)
    speculative_parser.set_defaults(func=_cmd_doctor_speculative)

    workloads_parser = subparsers.add_parser(
        "workloads",
        help="Deterministic code/JSON/reasoning workload matrix and evaluators",
    )
    workloads_subparsers = workloads_parser.add_subparsers(
        dest="workloads_command", required=True
    )

    list_parser = workloads_subparsers.add_parser(
        "list", help="List the pinned workload catalog"
    )
    list_parser.set_defaults(func=_cmd_workloads_list)

    matrix_parser = workloads_subparsers.add_parser(
        "generate-matrix",
        help=(
            "Materialize the deterministic (workload, context tier, decode "
            "mode) matrix, prompts, and planned commands. Dry-run only: "
            "never loads a model or downloads weights."
        ),
    )
    matrix_parser.add_argument("--model-id", required=True)
    matrix_parser.add_argument(
        "--model-family",
        required=True,
        help=(
            "Architecture family (e.g. 'qwen3_next'), used to determine "
            "which native-MTP rows are runnable via capability detection"
        ),
    )
    matrix_parser.add_argument("--output-dir", required=True)
    matrix_parser.add_argument(
        "--target-model-path",
        default=None,
        help="Existing local target checkpoint path for the planned commands",
    )
    matrix_parser.add_argument(
        "--mtp-sidecar-path",
        default=None,
        help="Existing local MTP sidecar checkpoint path for the planned commands",
    )
    matrix_parser.add_argument(
        "--context-tiers",
        nargs="+",
        choices=[tier.value for tier in ContextTier],
        default=None,
        help="Subset of context tiers to plan for (default: all)",
    )
    matrix_parser.add_argument("--max-tokens", type=int, default=128)
    matrix_parser.set_defaults(func=_cmd_workloads_generate_matrix)

    evaluate_parser = workloads_subparsers.add_parser(
        "evaluate",
        help="Run the deterministic evaluator for one workload against a response",
    )
    evaluate_parser.add_argument("--workload-id", required=True)
    evaluate_parser.add_argument(
        "--response-file", required=True, help="Path to the model's response text"
    )
    evaluate_parser.set_defaults(func=_cmd_workloads_evaluate)

    run_parser = workloads_subparsers.add_parser(
        "run",
        help=(
            "Execute selected runnable matrix rows through the MLX-LM "
            "collector and evaluate them deterministically"
        ),
    )
    run_parser.add_argument(
        "--matrix",
        required=True,
        help="Path to a `workloads generate-matrix` manifest.json",
    )
    run_parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Existing local target MLX model directory; required unless "
            "--dry-run (models are never downloaded)"
        ),
    )
    run_parser.add_argument(
        "--draft-model-path",
        default=None,
        help=(
            "Optional existing local MLX draft model; enables generic "
            "draft-model speculation on selected autoregressive rows, "
            "never on native-mtp rows"
        ),
    )
    run_parser.add_argument("--num-draft-tokens", type=int, default=2)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument(
        "--run-id", nargs="+", default=None, help="Select specific run_id(s)"
    )
    run_parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        choices=[category.value for category in WorkloadCategory],
        help="Filter selected rows by workload category",
    )
    run_parser.add_argument(
        "--context-tier",
        nargs="+",
        default=None,
        choices=[tier.value for tier in ContextTier],
        help="Filter selected rows by context tier",
    )
    run_parser.add_argument(
        "--mode",
        nargs="+",
        default=None,
        choices=[DECODE_MODE_AUTOREGRESSIVE, DECODE_MODE_NATIVE_MTP],
        help=(
            "Filter selected rows by decode mode; native-mtp rows are "
            "always rejected as unsupported, never executed"
        ),
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all selected rows even if a hash-matching completed artifact exists",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print selected rows, required local model paths, expected "
            "artifacts, and blockers without loading any model"
        ),
    )
    run_parser.set_defaults(func=_cmd_workloads_run)

    summarize_parser = workloads_subparsers.add_parser(
        "summarize",
        help="Aggregate pass rate/throughput/status counts across a `workloads run` results directory",
    )
    summarize_parser.add_argument(
        "--results", required=True, help="The --output-dir passed to `workloads run`"
    )
    summarize_parser.add_argument(
        "--output",
        default=None,
        help="Write the summary JSON to this path instead of stdout",
    )
    summarize_parser.set_defaults(func=_cmd_workloads_summarize)

    tune_parser = subparsers.add_parser(
        "tune",
        help=(
            "Offline, evidence-constrained recommendation of the fastest "
            "verified configuration for a workload/hardware target. Reads "
            "already-collected `workloads run` verification.json + "
            "final_record.json artifacts; never loads a model, requires a "
            "GPU, or executes a benchmark."
        ),
    )
    tune_parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more `workloads run --output-dir` results directories",
    )
    tune_parser.add_argument(
        "--policy",
        required=True,
        help="Path to a tune policy JSON/YAML file (objective + constraints)",
    )
    tune_parser.add_argument(
        "--output",
        default=None,
        help="Atomically write the full tune report JSON to this path",
    )
    tune_parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Print every violated constraint for every rejected candidate "
            "and the full accepted-candidate ranking (default: a concise "
            "summary with only the top rejection reason per candidate)"
        ),
    )
    tune_parser.set_defaults(func=_cmd_tune)

    tune_report_parser = subparsers.add_parser(
        "tune-report",
        help=(
            "Render a `tune` JSON report as a single, self-contained, "
            "portable HTML file (inline CSS, no JavaScript, no CDN, works "
            "offline). Never re-scores or re-computes anything; purely a "
            "read-only view over an already-produced tune report."
        ),
    )
    tune_report_parser.add_argument(
        "--input",
        required=True,
        help="Path to a `tune --output` JSON report",
    )
    tune_report_parser.add_argument(
        "--output",
        required=True,
        help="Path to atomically write the rendered HTML report to",
    )
    tune_report_parser.add_argument(
        "--include-paths",
        action="store_true",
        help=(
            "Include full local artifact paths (results directories, "
            "verification.json/final_record.json paths) as plain text. "
            "Default: redact every path to a basename/stable "
            "`runs/<run_id>/<file>` label so the report is safe to share "
            "without leaking a user's home directory layout."
        ),
    )
    tune_report_parser.set_defaults(func=_cmd_tune_report)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help=(
            "End-to-end: execute selected `workloads generate-matrix` rows, "
            "offline-tune the resulting evidence, and render the JSON/HTML "
            "report in one invocation. Composes `workloads run`, `tune`, "
            "and `tune-report` exactly as they already behave; never loads "
            "a model, downloads weights, or writes a report until its own "
            "phase's evidence exists and validates."
        ),
    )
    optimize_parser.add_argument(
        "--matrix",
        required=True,
        help="Path to a `workloads generate-matrix` manifest.json",
    )
    optimize_parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Existing local target MLX model directory; required unless "
            "--dry-run (models are never downloaded)"
        ),
    )
    optimize_parser.add_argument(
        "--draft-model-path",
        default=None,
        help=(
            "Optional existing local MLX draft model; enables generic "
            "draft-model speculation on selected autoregressive rows, "
            "never on native-mtp rows"
        ),
    )
    optimize_parser.add_argument("--num-draft-tokens", type=int, default=2)
    optimize_parser.add_argument(
        "--results",
        required=True,
        help=(
            "The `workloads run` results directory for this invocation; "
            "this is also the only results directory tuned unless "
            "--extra-results explicitly opts in additional ones"
        ),
    )
    optimize_parser.add_argument(
        "--extra-results",
        nargs="+",
        default=None,
        help=(
            "Opt-in: additional already-collected `workloads run` results "
            "directories to include when tuning, alongside --results. "
            "Omit to tune only the evidence produced by this invocation"
        ),
    )
    optimize_parser.add_argument(
        "--run-id", nargs="+", default=None, help="Select specific run_id(s)"
    )
    optimize_parser.add_argument(
        "--category",
        nargs="+",
        default=None,
        choices=[category.value for category in WorkloadCategory],
        help="Filter selected rows by workload category",
    )
    optimize_parser.add_argument(
        "--context-tier",
        nargs="+",
        default=None,
        choices=[tier.value for tier in ContextTier],
        help="Filter selected rows by context tier",
    )
    optimize_parser.add_argument(
        "--mode",
        nargs="+",
        default=None,
        choices=[DECODE_MODE_AUTOREGRESSIVE, DECODE_MODE_NATIVE_MTP],
        help=(
            "Filter selected rows by decode mode; native-mtp rows are "
            "always rejected as unsupported, never executed"
        ),
    )
    optimize_parser.add_argument("--seed", type=int, default=0)
    optimize_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all selected rows even if a hash-matching completed artifact exists",
    )
    optimize_parser.add_argument(
        "--policy",
        required=True,
        help="Path to a tune policy JSON/YAML file (objective + constraints)",
    )
    optimize_parser.add_argument(
        "--report-json",
        default=None,
        help=(
            "Atomically write the full tune report JSON to this path; "
            "required unless --dry-run"
        ),
    )
    optimize_parser.add_argument(
        "--report-html",
        default=None,
        help=(
            "Atomically write a self-contained HTML tune report to this "
            "path. Optional: omit to skip HTML rendering entirely"
        ),
    )
    optimize_parser.add_argument(
        "--include-paths",
        action="store_true",
        help=(
            "Include full local artifact paths in the HTML report. "
            "Default: redact every path, same as `tune-report`"
        ),
    )
    optimize_parser.add_argument(
        "--summary-json",
        default=None,
        help=(
            "Path to atomically write the machine-readable orchestration "
            "summary (phase statuses, run counts, recommendations, exit "
            "status) to. Default: <--results>/optimize_summary.json"
        ),
    )
    optimize_parser.add_argument(
        "--explain",
        action="store_true",
        help=(
            "Print every violated constraint for every rejected candidate "
            "(same as `tune --explain`)"
        ),
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print selected rows, required local model paths, expected "
            "run/tune/report artifacts, unsupported rows, and blockers "
            "without loading any model, tuning, or writing any report"
        ),
    )
    optimize_parser.set_defaults(func=_cmd_optimize)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # The scope, rather than a bare assignment, so the state is undone on
    # every exit path including the ``SystemExit`` argparse raises. Left
    # standing it would suppress installation for the next parse in this
    # process, which would then be scrubbed against the wrong argv.
    #
    # The command runs inside the scope too. ``_api_detail`` scrubs argv
    # values out of its diagnostics, and a value the caller supplied can
    # surface long after parsing succeeds: a path that does not open
    # reaches the error text verbatim. Closing the scope at the end of
    # parsing left that scrub with nothing installed and made it a no-op
    # for exactly the values it exists to remove.
    with _argument_scrub_scope(parser, raw_argv):
        _reject_credential_arguments(raw_argv)
        args = parser.parse_args(raw_argv)
        args._invocation = (parser.prog, *raw_argv)
        status = args.func(args)
    sys.exit(status)


if __name__ == "__main__":
    main()
