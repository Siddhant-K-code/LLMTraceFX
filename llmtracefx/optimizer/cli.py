"""CLI entrypoint for the inference-optimizer foundation primitives.

Subcommands:
    manifest         Collect a CPU-only, non-sensitive environment manifest.
    run              Execute a configured experiment (warmups + measured reps).
    collect-mlx      Run one local MLX-LM inference and record normalized evidence.
    native-mtp       Native Qwen MTP capability report / evidence collection.
    parse-llama-cpp  Convert llama.cpp text output into a canonical ExperimentRecord.
    doctor speculative  Diagnose whether speculative decoding/MTP is a net regression.
    workloads        Generate a deterministic code/JSON/reasoning workload matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

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
from .doctor.speculative import diagnose_speculative_regression
from .manifest import collect_environment_manifest
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
from .workloads.catalog import WORKLOADS, workload_by_id
from .workloads.evaluators import evaluate_workload
from .workloads.matrix import generate_matrix, write_matrix
from .workloads.schema import ContextTier, WorkloadSchemaError


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._invocation = (parser.prog, *raw_argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
