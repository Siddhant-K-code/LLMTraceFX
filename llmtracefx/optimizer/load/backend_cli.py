"""``sweep-llama-cpp``, ``sweep-mlx`` and ``sweep-ollama``: load sweeps for
backends driven through their native API instead of an OpenAI-compatible one.

All reuse the ``sweep-api`` load options, dispatcher, goodput and artifact.
Only the worker, the request plan and the attribution ``backend`` block
differ. See ``docs/load-sweeps.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..collectors.mlx import MLXCollectorError, MLXLMRuntime
from ..collectors.ollama import (
    API_CHAT,
    API_PATHS,
    DEFAULT_BASE_URL,
    OllamaSettings,
    probe_version,
)
from ..collectors.ollama import request_plan as ollama_request_plan
from ..collectors.ollama import resolve_credential as resolve_ollama_credential
from ..collectors.openai_api import UrllibStreamingTransport
from ..server_metrics import ServerMetricsError, validate_metrics_url
from . import llama_cpp_worker, mlx_worker, ollama_worker
from .api_worker import ApiRequestSpec
from .cli import (
    Fail,
    SweepOutput,
    add_dry_run_argument,
    add_load_arguments,
    add_remote_load_argument,
    add_server_metrics_arguments,
    ensure_new_output_dir,
    load_argv,
    load_settings,
    metrics_pairing,
    points_from_args,
    read_prompts,
    refuse_remote_endpoint,
    run_sweep,
    slo_from_args,
    sweep_mode,
    validate_load_args,
    workload_summary,
    write_sweep_plan,
)
from .dispatcher import DispatchConfigError
from .llama_cpp_worker import (
    LLAMA_CPP_FIELD_PROVENANCE,
    LlamaCppConfig,
    check_slots,
    llama_cpp_attribution,
    make_llama_cpp_sender,
    probe_props,
    resolve_credential,
)
from .mlx_worker import (
    IN_PROCESS_CPU_REASON,
    MLX_FIELD_PROVENANCE,
    MLXLoadConfig,
    MLXSession,
    check_points,
    make_mlx_sender,
    mlx_attribution,
    request_settings,
)
from .ollama_worker import (
    OLLAMA_FIELD_PROVENANCE,
    make_ollama_sender,
    ollama_attribution,
)
from .sweep import SweepPoint, point_specs

_PROG = "llmtracefx-optimizer"


Detail = Callable[[BaseException], str]


def _fail_with(env_var: str | None) -> tuple[Detail, Fail]:
    from ..cli import _scrubbed_detail

    def detail(exc: BaseException) -> str:
        return _scrubbed_detail(env_var, exc)

    def fail(prefix: str, exc: BaseException) -> int:
        print(f"{prefix}: {detail(exc)}", file=sys.stderr)
        return 1

    return detail, fail


def _argv(
    command: str, pairs: list[tuple[str, Any]], args: argparse.Namespace
) -> list[str]:
    argv = [_PROG, command]
    for flag, value in pairs:
        if value is True:
            argv.append(flag)
        elif value is not None and value is not False:
            argv.extend((flag, str(value)))
    argv.extend(load_argv(args))
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    return argv


def _specs_for(
    args: argparse.Namespace, prompts: list[str]
) -> Callable[[SweepPoint], list[ApiRequestSpec]]:
    def specs_for(point: SweepPoint) -> list[ApiRequestSpec]:
        return point_specs(prompts, args.requests_per_point, point_index=point.index)

    return specs_for


# sweep-llama-cpp


def add_sweep_llama_cpp_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "sweep-llama-cpp",
        help=(
            "Run a load sweep against llama-server's native /completion "
            "endpoint, refusing load above its slot count"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--endpoint",
        required=True,
        help=(
            "llama-server root URL, with any --api-prefix, for example "
            "http://127.0.0.1:8080. /completion and /props are added."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "Name of the environment variable holding llama-server's --api-key, "
            "if it has one. Only the name is recorded."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        required=True,
        help="n_predict for every request; llama.cpp's own default is no limit",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        help=(
            "Let the server reuse a slot's KV cache across requests. Off by "
            "default so each request pays its own prefill."
        ),
    )
    parser.add_argument(
        "--allow-server-queue",
        action="store_true",
        help=(
            "Allow more requests in flight than the server has slots; the extra "
            "requests wait in llama-server's queue"
        ),
    )
    add_load_arguments(parser)
    add_server_metrics_arguments(parser)
    add_remote_load_argument(parser)
    add_dry_run_argument(parser)
    parser.set_defaults(func=cmd_sweep_llama_cpp)
    return parser


def _llama_cpp_argv(args: argparse.Namespace) -> list[str]:
    return _argv(
        "sweep-llama-cpp",
        [
            ("--run-id", args.run_id),
            ("--endpoint", args.endpoint),
            ("--output-dir", args.output_dir),
            ("--api-key-env", args.api_key_env),
            ("--max-output-tokens", args.max_output_tokens),
            ("--temperature", args.temperature),
            ("--top-p", args.top_p),
            ("--top-k", args.top_k),
            ("--seed", args.seed),
            ("--request-timeout", args.request_timeout),
            ("--cache-prompt", args.cache_prompt),
            ("--allow-server-queue", args.allow_server_queue),
            ("--server-metrics-url", args.server_metrics_url),
            (
                "--server-metrics-runtime",
                (
                    args.server_metrics_runtime
                    if args.server_metrics_url is not None
                    else None
                ),
            ),
            (
                "--server-metrics-assume-isolated",
                args.server_metrics_assume_isolated,
            ),
            (
                "--server-metrics-settle-timeout",
                (
                    args.server_metrics_settle_timeout
                    if args.server_metrics_url is not None
                    else None
                ),
            ),
            ("--allow-remote-load", args.allow_remote_load),
        ],
        args,
    )


def cmd_sweep_llama_cpp(args: argparse.Namespace) -> int:
    detail, fail = _fail_with(args.api_key_env)
    output_dir = Path(args.output_dir)
    try:
        validate_load_args(args)
        config = LlamaCppConfig(
            run_id=args.run_id,
            base_url=args.endpoint,
            n_predict=args.max_output_tokens,
            cache_prompt=args.cache_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            request_timeout_seconds=args.request_timeout,
            credential_env_var=args.api_key_env,
        )
        refuse_remote_endpoint(args.endpoint, allowed=args.allow_remote_load)
        slo = slo_from_args(args)
        prompts = read_prompts(args)
        metrics_url = (
            None
            if args.server_metrics_url is None
            else validate_metrics_url(args.server_metrics_url)
        )
        points = points_from_args(args)
        plan = llama_cpp_worker.request_plan(config)
        plan["command_argv"] = _llama_cpp_argv(args)
    except (OSError, UnicodeError, ValueError, ServerMetricsError) as exc:
        return fail("Failed to configure sweep", exc)

    credential = (
        os.environ.get(args.api_key_env, "").strip() or None
        if args.api_key_env
        else None
    )

    def output(props: Any) -> SweepOutput:
        backend = llama_cpp_worker.backend_block(
            config, props, allow_server_queue=args.allow_server_queue
        )
        return SweepOutput(
            output_dir=output_dir,
            mode=sweep_mode(args),
            load=load_settings(args),
            slo=slo,
            request_plan=plan,
            workload=workload_summary(args, prompts, None),
            points=points,
            requests_per_point=args.requests_per_point,
            credential=credential,
            attribution=llama_cpp_attribution(backend),
            field_provenance=LLAMA_CPP_FIELD_PROVENANCE,
        )

    if args.dry_run:
        extra: dict[str, Any] = {
            "credential_env_var_present": (
                None if args.api_key_env is None else credential is not None
            )
        }
        if metrics_url is not None:
            extra["server_metrics_url"] = metrics_url
        return write_sweep_plan(output(None), extra, fail)

    transport = UrllibStreamingTransport()
    try:
        resolve_credential(config, os.environ)
        props = probe_props(config, transport=transport, environ=os.environ)
        check_slots(
            points, props.total_slots, allow_server_queue=args.allow_server_queue
        )
        ensure_new_output_dir(output_dir)
        send = make_llama_cpp_sender(config, transport=transport, environ=os.environ)
    except (OSError, ValueError, DispatchConfigError) as exc:
        return fail("Failed to configure sweep", exc)

    return run_sweep(
        output(props),
        send=send,
        specs_for=_specs_for(args, prompts),
        cooldown_seconds=args.cooldown_seconds,
        max_dispatch_lag_ms=args.max_dispatch_lag_ms,
        server_metrics=metrics_pairing(args, metrics_url),
        describe_error=detail,
        fail=fail,
    )


# sweep-mlx


def add_sweep_mlx_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "sweep-mlx",
        help=(
            "Run a load sweep against one mlx-lm model loaded in this process; "
            "one request at a time"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local MLX model directory; required unless --dry-run is set",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300.0,
        help="Per-request limit in seconds, checked between generated tokens",
    )
    add_load_arguments(parser)
    add_dry_run_argument(parser)
    parser.set_defaults(func=cmd_sweep_mlx)
    return parser


def _mlx_argv(args: argparse.Namespace) -> list[str]:
    return _argv(
        "sweep-mlx",
        [
            ("--run-id", args.run_id),
            ("--model-path", args.model_path),
            ("--model-id", args.model_id),
            ("--output-dir", args.output_dir),
            ("--max-output-tokens", args.max_output_tokens),
            ("--seed", args.seed),
            ("--temperature", args.temperature),
            ("--top-p", args.top_p),
            ("--request-timeout", args.request_timeout),
        ],
        args,
    )


def cmd_sweep_mlx(args: argparse.Namespace) -> int:
    detail, fail = _fail_with(None)
    output_dir = Path(args.output_dir)
    try:
        validate_load_args(args)
        if args.model_path is None and not args.dry_run:
            raise ValueError("--model-path is required unless --dry-run is set")
        model_path = Path(args.model_path) if args.model_path else Path()
        if args.model_path is not None and not model_path.exists():
            raise ValueError(f"--model-path does not exist: {model_path}")
        config = MLXLoadConfig(
            run_id=args.run_id,
            model_path=model_path,
            model_id=args.model_id,
            max_tokens=args.max_output_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            request_timeout_seconds=args.request_timeout,
        )
        slo = slo_from_args(args)
        prompts = read_prompts(args)
        points = points_from_args(args)
        check_points(points)
        plan = request_settings(config)
        plan["command_argv"] = _mlx_argv(args)
    except (OSError, UnicodeError, ValueError) as exc:
        return fail("Failed to configure sweep", exc)

    def output(info: Any) -> SweepOutput:
        return SweepOutput(
            output_dir=output_dir,
            mode=sweep_mode(args),
            load=load_settings(args),
            slo=slo,
            request_plan=plan,
            workload=workload_summary(args, prompts, None),
            points=points,
            requests_per_point=args.requests_per_point,
            attribution=mlx_attribution(mlx_worker.backend_block(config, info)),
            field_provenance=MLX_FIELD_PROVENANCE,
        )

    if args.dry_run:
        return write_sweep_plan(output(None), {"model_loaded": False}, fail)

    session = MLXSession(
        lambda: MLXLMRuntime(temperature=config.temperature, top_p=config.top_p),
        config,
        clock=time.perf_counter,
    )
    try:
        ensure_new_output_dir(output_dir)
        print(f"Loading {config.model_id} in this process")
        info = session.start()
    except (OSError, ValueError, RuntimeError, MemoryError, MLXCollectorError) as exc:
        session.close()
        return fail("Failed to load the model", exc)

    try:
        return run_sweep(
            output(info),
            send=make_mlx_sender(session, config),
            specs_for=_specs_for(args, prompts),
            cooldown_seconds=args.cooldown_seconds,
            max_dispatch_lag_ms=args.max_dispatch_lag_ms,
            server_metrics=None,
            describe_error=detail,
            fail=fail,
            client_cpu_withheld_reason=IN_PROCESS_CPU_REASON,
        )
    finally:
        session.close()


# sweep-ollama


def _think(value: str | None) -> bool | None:
    return None if value is None else value == "true"


def add_sweep_ollama_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "sweep-ollama",
        help=(
            "Run a load sweep against Ollama's native /api/chat or /api/generate, "
            "recording its server timers per request"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_BASE_URL,
        help=(
            f"Ollama root URL (default {DEFAULT_BASE_URL}). /api/chat or "
            "/api/generate and /api/version are added."
        ),
    )
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument(
        "--api",
        choices=sorted(API_PATHS),
        default=API_CHAT,
        help="Native API to call (default chat)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--api-key-env",
        default=None,
        help=(
            "Name of an environment variable holding a bearer token for a proxy "
            "in front of Ollama. Only the name is recorded."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        required=True,
        help="options.num_predict for every request; Ollama's default is no limit",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument(
        "--think",
        choices=("true", "false"),
        default=None,
        help="Send think: true or false; unset leaves the model's default",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    add_load_arguments(parser)
    add_remote_load_argument(parser)
    add_dry_run_argument(parser)
    parser.set_defaults(func=cmd_sweep_ollama)
    return parser


def _ollama_argv(args: argparse.Namespace) -> list[str]:
    return _argv(
        "sweep-ollama",
        [
            ("--run-id", args.run_id),
            ("--endpoint", args.endpoint),
            ("--model", args.model),
            ("--api", args.api),
            ("--output-dir", args.output_dir),
            ("--api-key-env", args.api_key_env),
            ("--max-output-tokens", args.max_output_tokens),
            ("--temperature", args.temperature),
            ("--top-p", args.top_p),
            ("--top-k", args.top_k),
            ("--seed", args.seed),
            ("--num-ctx", args.num_ctx),
            ("--think", args.think),
            ("--request-timeout", args.request_timeout),
            ("--allow-remote-load", args.allow_remote_load),
        ],
        args,
    )


def cmd_sweep_ollama(args: argparse.Namespace) -> int:
    detail, fail = _fail_with(args.api_key_env)
    output_dir = Path(args.output_dir)
    try:
        validate_load_args(args)
        settings = OllamaSettings(
            base_url=args.endpoint,
            model=args.model,
            api=args.api,
            num_predict=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            num_ctx=args.num_ctx,
            think=_think(args.think),
            request_timeout_seconds=args.request_timeout,
            credential_env_var=args.api_key_env,
        )
        refuse_remote_endpoint(args.endpoint, allowed=args.allow_remote_load)
        slo = slo_from_args(args)
        prompts = read_prompts(args)
        points = points_from_args(args)
        plan = ollama_request_plan(settings)
        plan["command_argv"] = _ollama_argv(args)
    except (OSError, UnicodeError, ValueError) as exc:
        return fail("Failed to configure sweep", exc)

    credential = (
        os.environ.get(args.api_key_env, "").strip() or None
        if args.api_key_env
        else None
    )

    def output(version: Any) -> SweepOutput:
        return SweepOutput(
            output_dir=output_dir,
            mode=sweep_mode(args),
            load=load_settings(args),
            slo=slo,
            request_plan=plan,
            workload=workload_summary(args, prompts, None),
            points=points,
            requests_per_point=args.requests_per_point,
            credential=credential,
            attribution=ollama_attribution(
                ollama_worker.backend_block(settings, version)
            ),
            field_provenance=OLLAMA_FIELD_PROVENANCE,
        )

    if args.dry_run:
        extra: dict[str, Any] = {
            "credential_env_var_present": (
                None if args.api_key_env is None else credential is not None
            )
        }
        return write_sweep_plan(output(None), extra, fail)

    transport = UrllibStreamingTransport()
    try:
        resolved = resolve_ollama_credential(settings, os.environ)
        version = probe_version(settings, transport=transport, credential=resolved)
        if version.version is None:
            raise ValueError(
                f"GET /api/version failed ({version.error}); is Ollama running "
                "at the server URL?"
            )
        ensure_new_output_dir(output_dir)
        send = make_ollama_sender(
            settings, transport=transport, environ=os.environ, run_id=args.run_id
        )
    except (OSError, ValueError, DispatchConfigError) as exc:
        return fail("Failed to configure sweep", exc)

    return run_sweep(
        output(version),
        send=send,
        specs_for=_specs_for(args, prompts),
        cooldown_seconds=args.cooldown_seconds,
        max_dispatch_lag_ms=args.max_dispatch_lag_ms,
        server_metrics=None,
        describe_error=detail,
        fail=fail,
    )


def add_backend_sweep_parsers(subparsers: Any) -> None:
    add_sweep_llama_cpp_parser(subparsers)
    add_sweep_mlx_parser(subparsers)
    add_sweep_ollama_parser(subparsers)


__all__ = [
    "add_backend_sweep_parsers",
    "add_sweep_llama_cpp_parser",
    "add_sweep_mlx_parser",
    "add_sweep_ollama_parser",
    "cmd_sweep_llama_cpp",
    "cmd_sweep_mlx",
    "cmd_sweep_ollama",
]
