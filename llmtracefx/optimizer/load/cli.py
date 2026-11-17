"""The ``sweep-api`` command: a concurrency or request rate sweep against one endpoint.

Registered by ``llmtracefx.optimizer.cli`` through :func:`add_sweep_api_parser`.
:func:`add_api_client_arguments` and :func:`base_api_config` are shared
with other load commands that drive the same OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import urllib.parse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text, sha256_text
from ..collectors.openai_api import (
    DEFAULT_RETAINED_EVENT_LIMIT,
    GLM_REASONING_EFFORT_LEVELS,
    THINKING_TYPES,
    APICollectionConfig,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    UrllibStreamingTransport,
    assert_credential_not_embedded,
    build_request_plan,
    redact_text_for_dry_run,
)
from ..schema import utc_now_iso
from ..server_metrics import (
    RUNTIME_CHOICES,
    ServerMetricsError,
    validate_metrics_url,
)
from .api_worker import make_api_sender, preflight, request_config
from .artifact import SWEEP_FILE_NAME, SWEEP_PLAN_FILE_NAME
from .dispatcher import (
    ARRIVAL_DISTRIBUTIONS,
    ARRIVAL_POISSON,
    DEFAULT_MAX_DISPATCH_LAG_MS,
    MAX_REQUESTS,
    DispatchConfigError,
)
from .goodput import SloTargets
from .sweep import (
    MODE_CONCURRENCY,
    MODE_REQUEST_RATE,
    POINT_COMPLETE,
    PrometheusPairing,
    SweepPoint,
    SweepRun,
    build_points,
    point_specs,
    run_points,
    sweep_document,
    sweep_plan_document,
)

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})


def _int_list(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _float_list(text: str) -> list[float]:
    try:
        values = [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from None
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def add_api_client_arguments(parser: argparse.ArgumentParser) -> None:
    """Endpoint, model, credential, sampling and timeout options shared with ``collect-api``."""
    parser.add_argument(
        "--run-id",
        required=True,
        help="Base run id; each request gets a unique suffix",
    )
    parser.add_argument(
        "--provider",
        default="openai-compatible",
        help="Short provider label recorded in evidence (never a secret)",
    )
    parser.add_argument(
        "--endpoint",
        required=True,
        help=(
            "Full chat-completions URL. Must be https for non-local hosts and "
            "must not embed credentials."
        ),
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional UTF-8 system prompt file; hashed, never copied",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help=(
            "Name of the environment variable holding the API key. Only the "
            "name is recorded."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds; no retries are performed",
    )
    parser.add_argument(
        "--retained-event-limit",
        type=int,
        default=DEFAULT_RETAINED_EVENT_LIMIT,
    )
    parser.add_argument(
        "--reasoning-effort", choices=GLM_REASONING_EFFORT_LEVELS, default=None
    )
    parser.add_argument("--thinking", choices=THINKING_TYPES, default=None)
    parser.add_argument("--clear-thinking", choices=("true", "false"), default=None)
    parser.add_argument(
        "--stream-include-usage",
        action="store_true",
        help=(
            "Send stream_options.include_usage so servers such as vLLM and "
            "SGLang report token counts in the stream. Without token counts "
            "TPOT cannot be derived."
        ),
    )


def add_server_metrics_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--server-metrics-url",
        default=None,
        help=(
            "Optional Prometheus /metrics URL of the serving engine, scraped "
            "before and after each load point"
        ),
    )
    parser.add_argument(
        "--server-metrics-runtime", choices=RUNTIME_CHOICES, default="auto"
    )
    parser.add_argument("--server-metrics-assume-isolated", action="store_true")
    parser.add_argument("--server-metrics-settle-timeout", type=float, default=2.0)


def base_api_config(
    args: argparse.Namespace, *, prompt: str, command_argv: Sequence[str]
) -> APICollectionConfig:
    """The shared collector config; per-request prompt and run id are set later."""
    return APICollectionConfig(
        run_id=args.run_id,
        provider=args.provider,
        endpoint=args.endpoint,
        model_id=args.model_id,
        prompt=prompt,
        output_dir=Path(args.output_dir),
        command_argv=tuple(command_argv),
        credential_env_var=args.api_key_env,
        system_prompt=None,
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
        ),
        model_revision=args.model_revision,
        stream_include_usage=args.stream_include_usage,
    )


def is_loopback_endpoint(endpoint: str) -> bool:
    host = urllib.parse.urlsplit(endpoint).hostname
    if not host:
        return False
    host = host.lower().rstrip(".")
    if host in _LOOPBACK_NAMES or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def add_sweep_api_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "sweep-api",
        help=(
            "Run a concurrency or request rate sweep against one "
            "OpenAI-compatible endpoint and report latency, goodput and "
            "scheduling fidelity per load point"
        ),
        # Prefix matching would let "--api-key <secret>" resolve to
        # "--api-key-env" and persist the secret as a variable name.
        allow_abbrev=False,
    )
    add_api_client_arguments(parser)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument(
        "--prompt-file",
        help="One UTF-8 prompt sent by every request; hashed, never copied",
    )
    prompts.add_argument(
        "--prompts-jsonl",
        help=(
            'JSON lines of {"prompt": "..."}; requests cycle through them in '
            "order. Hashed, never copied."
        ),
    )
    load = parser.add_mutually_exclusive_group(required=True)
    load.add_argument(
        "--concurrency",
        type=_int_list,
        help="Closed-loop concurrency levels, e.g. 1,2,4,8",
    )
    load.add_argument(
        "--request-rate",
        type=_float_list,
        help="Open-loop request rates per second, e.g. 0.5,1,2",
    )
    parser.add_argument(
        "--arrival",
        choices=ARRIVAL_DISTRIBUTIONS,
        default=ARRIVAL_POISSON,
        help="Arrival process for --request-rate (default: poisson)",
    )
    parser.add_argument(
        "--arrival-seed",
        type=int,
        default=None,
        help="Seed for poisson arrivals; required with --arrival poisson",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Cap on requests in flight for --request-rate (default: 256)",
    )
    parser.add_argument(
        "--requests-per-point",
        type=int,
        required=True,
        help="Requests dispatched at each load point",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Pause between load points so one point's tail does not reach the next",
    )
    parser.add_argument("--slo-ttft-ms", type=float, default=None)
    parser.add_argument("--slo-tpot-ms", type=float, default=None)
    parser.add_argument("--slo-e2e-ms", type=float, default=None)
    parser.add_argument(
        "--max-dispatch-lag-ms",
        type=float,
        default=DEFAULT_MAX_DISPATCH_LAG_MS,
        help="p99 client dispatch lag above which a point is marked degraded",
    )
    add_server_metrics_arguments(parser)
    parser.add_argument(
        "--allow-remote-load",
        action="store_true",
        help=(
            "Allow a non-loopback endpoint. A sweep sends many requests; "
            "against a hosted API every one of them is billed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=f"Write {SWEEP_PLAN_FILE_NAME} and send nothing",
    )
    parser.set_defaults(func=cmd_sweep_api)
    return parser


def _sweep_argv(args: argparse.Namespace) -> tuple[str, ...]:
    """A credential-free, fully explicit invocation for the record."""
    argv: list[str] = ["llmtracefx-optimizer", "sweep-api"]
    for flag, value in (
        ("--run-id", args.run_id),
        ("--provider", args.provider),
        ("--endpoint", args.endpoint),
        ("--model-id", args.model_id),
        ("--model-revision", args.model_revision),
        ("--prompt-file", args.prompt_file),
        ("--prompts-jsonl", args.prompts_jsonl),
        ("--system-prompt-file", args.system_prompt_file),
        ("--output-dir", args.output_dir),
        ("--api-key-env", args.api_key_env),
        ("--max-output-tokens", args.max_output_tokens),
        ("--temperature", args.temperature),
        ("--top-p", args.top_p),
        ("--seed", args.seed),
        ("--request-timeout", args.request_timeout),
        ("--retained-event-limit", args.retained_event_limit),
        ("--reasoning-effort", args.reasoning_effort),
        ("--thinking", args.thinking),
        ("--clear-thinking", args.clear_thinking),
        (
            "--concurrency",
            None if args.concurrency is None else _join(args.concurrency),
        ),
        (
            "--request-rate",
            None if args.request_rate is None else _join(args.request_rate),
        ),
        ("--arrival", args.arrival if args.request_rate is not None else None),
        ("--arrival-seed", args.arrival_seed),
        ("--max-in-flight", args.max_in_flight),
        ("--requests-per-point", args.requests_per_point),
        ("--cooldown-seconds", args.cooldown_seconds),
        ("--slo-ttft-ms", args.slo_ttft_ms),
        ("--slo-tpot-ms", args.slo_tpot_ms),
        ("--slo-e2e-ms", args.slo_e2e_ms),
        ("--max-dispatch-lag-ms", args.max_dispatch_lag_ms),
        ("--server-metrics-url", args.server_metrics_url),
        (
            "--server-metrics-runtime",
            args.server_metrics_runtime if args.server_metrics_url else None,
        ),
        (
            "--server-metrics-settle-timeout",
            args.server_metrics_settle_timeout if args.server_metrics_url else None,
        ),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    for flag, enabled in (
        ("--stream-include-usage", args.stream_include_usage),
        ("--server-metrics-assume-isolated", args.server_metrics_assume_isolated),
        ("--allow-remote-load", args.allow_remote_load),
        ("--dry-run", args.dry_run),
    ):
        if enabled:
            argv.append(flag)
    return tuple(argv)


def _join(values: Sequence[float]) -> str:
    return ",".join(f"{value:g}" for value in values)


def read_prompts_jsonl(path: Path) -> list[str]:
    """Prompts from ``{"prompt": "..."}`` lines. Any other key is refused, not ignored."""
    prompts: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"line {number} is not valid JSON") from None
        if not isinstance(item, dict) or set(item) != {"prompt"}:
            raise ValueError(
                f'line {number} must be an object with only a "prompt" key'
            )
        prompt = item["prompt"]
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"line {number}: prompt must be non-empty text")
        prompts.append(prompt)
    if not prompts:
        raise ValueError("the prompts file holds no prompts")
    return prompts


def _slo(args: argparse.Namespace) -> SloTargets | None:
    if (
        args.slo_ttft_ms is None
        and args.slo_tpot_ms is None
        and args.slo_e2e_ms is None
    ):
        return None
    return SloTargets(
        ttft_ms=args.slo_ttft_ms, tpot_ms=args.slo_tpot_ms, e2e_ms=args.slo_e2e_ms
    )


def _validate(args: argparse.Namespace) -> None:
    if not 1 <= args.requests_per_point <= MAX_REQUESTS:
        raise ValueError(f"--requests-per-point must be between 1 and {MAX_REQUESTS}")
    if args.cooldown_seconds < 0:
        raise ValueError("--cooldown-seconds must be at least 0")
    if args.concurrency is not None:
        for flag, value in (
            ("--arrival-seed", args.arrival_seed),
            ("--max-in-flight", args.max_in_flight),
        ):
            if value is not None:
                raise ValueError(f"{flag} only applies to --request-rate")
    elif args.arrival == ARRIVAL_POISSON and args.arrival_seed is None:
        raise ValueError(
            "--arrival poisson needs --arrival-seed so the schedule can be repeated"
        )
    elif args.arrival != ARRIVAL_POISSON and args.arrival_seed is not None:
        raise ValueError("--arrival-seed only applies to --arrival poisson")
    if not is_loopback_endpoint(args.endpoint) and not args.allow_remote_load:
        raise ValueError(
            "the endpoint is not on this machine. A sweep sends many requests "
            "and a hosted API bills each one; pass --allow-remote-load to "
            "confirm"
        )


def _workload(
    args: argparse.Namespace, prompts: Sequence[str], system_prompt: str | None
) -> dict[str, Any]:
    distinct = list(dict.fromkeys(sha256_text(prompt) for prompt in prompts))
    return {
        "prompt_source": "prompt_file" if args.prompt_file else "prompts_jsonl",
        "prompt_count": len(prompts),
        "distinct_prompt_sha256": distinct,
        "system_prompt_sha256": (
            None if system_prompt is None else sha256_text(system_prompt)
        ),
        "requests_per_point": args.requests_per_point,
        "order": "request i of each point uses prompt i modulo prompt_count",
        "stream_include_usage": args.stream_include_usage,
    }


def _load_settings(args: argparse.Namespace) -> dict[str, Any]:
    rate_mode = args.request_rate is not None
    return {
        "requests_per_point": args.requests_per_point,
        "cooldown_seconds": args.cooldown_seconds,
        "max_dispatch_lag_ms": args.max_dispatch_lag_ms,
        "arrival": args.arrival if rate_mode else None,
        "arrival_seed": args.arrival_seed if rate_mode else None,
        "levels": list(args.request_rate if rate_mode else args.concurrency),
    }


def _write(path: Path, document: dict[str, Any], credential: str | None) -> str:
    text = json.dumps(document, indent=2, allow_nan=False)
    # Defence in depth: every field is credential-free by construction,
    # and the rendered document is scrubbed once more before it lands.
    text = redact_text_for_dry_run(text, credential)
    atomic_write_text(path, text + "\n")
    return text


def _point_line(point: dict[str, Any]) -> str:
    counts = point["counts"]
    latency = point["latency"]
    parts = [
        f"{point['label']}: {counts['succeeded']}/{counts['requests']} succeeded",
    ]
    for name in ("ttft_ms", "e2e_ms"):
        stats = latency[name]
        if stats["count"]:
            parts.append(
                f"{name[:-3]} p50 {stats['p50']:.1f} ms p99 {stats['p99']:.1f} ms"
            )
    goodput = point.get("goodput")
    if goodput is not None:
        value = goodput["request_goodput_fraction"]["value"]
        bounds = goodput["request_goodput_fraction_bounds"]
        if value is not None:
            parts.append(f"goodput {value:.3f}")
        elif bounds is not None:
            parts.append(
                f"goodput between {bounds['lower']:.3f} and {bounds['upper']:.3f}"
            )
        else:
            parts.append("goodput withheld")
    scheduling = point["scheduling"]
    if scheduling["status"] != "ok":
        parts.append(f"scheduling degraded ({', '.join(scheduling['reasons'])})")
    if point["status"] != POINT_COMPLETE:
        parts.append(point["status"])
    metrics = point.get("server_metrics")
    if metrics is not None:
        parts.append(
            f"server metrics {metrics.get('attribution_status', metrics['status'])}"
        )
    return "  " + "; ".join(parts)


def cmd_sweep_api(args: argparse.Namespace) -> int:
    from ..cli import _scrubbed_detail

    def detail(exc: BaseException) -> str:
        return _scrubbed_detail(args.api_key_env, exc)

    def fail(prefix: str, exc: BaseException) -> int:
        print(f"{prefix}: {detail(exc)}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    try:
        _validate(args)
        slo = _slo(args)
        if args.prompt_file:
            prompts = [Path(args.prompt_file).read_text(encoding="utf-8")]
        else:
            prompts = read_prompts_jsonl(Path(args.prompts_jsonl))
        system_prompt = (
            Path(args.system_prompt_file).read_text(encoding="utf-8")
            if args.system_prompt_file
            else None
        )
        metrics_url = (
            None
            if args.server_metrics_url is None
            else validate_metrics_url(args.server_metrics_url)
        )
        points = build_points(
            concurrency=args.concurrency,
            request_rates=args.request_rate,
            arrival=args.arrival,
            seed=args.arrival_seed,
            max_in_flight=args.max_in_flight,
        )
        base = base_api_config(args, prompt=prompts[0], command_argv=_sweep_argv(args))

        def specs_for(point: SweepPoint) -> list[Any]:
            return point_specs(
                prompts,
                args.requests_per_point,
                point_index=point.index,
                system_prompt=system_prompt,
            )

        first_config = request_config(base, specs_for(points[0])[0], 0)
        assert_credential_not_embedded(first_config, os.environ)
        request_plan = build_request_plan(first_config, environ=os.environ).to_dict()
    except (
        OSError,
        UnicodeError,
        ValueError,
        OpenAIStreamCollectorError,
        ServerMetricsError,
    ) as exc:
        return fail("Failed to configure sweep", exc)

    credential = os.environ.get(args.api_key_env, "").strip() or None
    mode = MODE_REQUEST_RATE if args.request_rate is not None else MODE_CONCURRENCY
    load = _load_settings(args)
    workload = _workload(args, prompts, system_prompt)

    if args.dry_run:
        plan = sweep_plan_document(
            mode=mode,
            load=load,
            slo=slo,
            request_plan=request_plan,
            workload=workload,
            points=points,
            requests_per_point=args.requests_per_point,
        )
        plan["credential_env_var_present"] = credential is not None
        if metrics_url is not None:
            plan["server_metrics_url"] = metrics_url
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            _write(output_dir / SWEEP_PLAN_FILE_NAME, plan, credential)
        except OSError as exc:
            return fail("Failed to write the sweep plan", exc)
        print(
            f"Sweep plan written to {output_dir / SWEEP_PLAN_FILE_NAME}: "
            f"{len(points)} points, {plan['total_requests']} requests, "
            "nothing sent"
        )
        return 0

    try:
        for point in points:
            preflight(base, specs_for(point), environ=os.environ)
        stale = [
            output_dir / SWEEP_FILE_NAME,
            *output_dir.glob("server_metrics_point_*.json"),
        ]
        existing = [path.name for path in stale if path.exists()]
        if existing:
            raise ValueError(
                f"{output_dir} already holds sweep output ({', '.join(existing)}); "
                "choose a new --output-dir"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
    except (
        OSError,
        ValueError,
        OpenAIStreamCollectorError,
        DispatchConfigError,
    ) as exc:
        return fail("Failed to configure sweep", exc)

    started = utc_now_iso()
    completed: list[dict[str, Any]] = []

    def document(*, finished: bool, interrupted: bool = False) -> dict[str, Any]:
        return sweep_document(
            mode=mode,
            load=load,
            slo=slo,
            request_plan=request_plan,
            workload=workload,
            points=completed,
            planned_points=len(points),
            finished=finished,
            interrupted=interrupted,
            started_at_utc=started,
        )

    def on_point(point: dict[str, Any]) -> None:
        completed.append(point)
        print(_point_line(point))
        _write(output_dir / SWEEP_FILE_NAME, document(finished=False), credential)

    pairing = (
        None
        if metrics_url is None
        else PrometheusPairing(
            metrics_url,
            runtime=args.server_metrics_runtime,
            settle_timeout_seconds=args.server_metrics_settle_timeout,
            assume_isolated=args.server_metrics_assume_isolated,
        )
    )
    run = SweepRun(
        points=points,
        send=make_api_sender(
            base, transport=UrllibStreamingTransport(), environ=os.environ
        ),
        specs_for=specs_for,
        output_dir=output_dir,
        slo=slo,
        cooldown_seconds=args.cooldown_seconds,
        max_dispatch_lag_ms=args.max_dispatch_lag_ms,
        server_metrics=pairing,
        describe_error=detail,
        on_point=on_point,
    )
    print(
        f"Sweeping {len(points)} points x {args.requests_per_point} requests "
        f"({mode}). Results are concurrent-load measurements, not isolated "
        "single-request evidence."
    )
    try:
        run_points(run)
    except KeyboardInterrupt:
        _safe_write(output_dir, document(finished=False, interrupted=True), credential)
        print(
            f"Interrupted; {len(completed)} finished points kept in "
            f"{output_dir / SWEEP_FILE_NAME}",
            file=sys.stderr,
        )
        return 130
    except OSError as exc:
        return fail("Failed to write sweep output", exc)

    final = document(finished=True)
    try:
        _write(output_dir / SWEEP_FILE_NAME, final, credential)
    except OSError as exc:
        return fail("Failed to write sweep output", exc)
    return _report(final, output_dir)


def _safe_write(
    output_dir: Path, document: dict[str, Any], credential: str | None
) -> None:
    try:
        _write(output_dir / SWEEP_FILE_NAME, document, credential)
    except OSError:
        pass


def _report(document: dict[str, Any], output_dir: Path) -> int:
    summary = document["summary"]
    print(f"Sweep written to {output_dir / SWEEP_FILE_NAME}")
    problems: list[tuple[str, list[str]]] = [
        ("Incomplete points", summary["incomplete_points"]),
        ("Points with failed requests", summary["points_with_failures"]),
        ("Server metrics errors", summary["server_metrics_errors"]),
    ]
    for label, names in problems:
        if names:
            print(f"{label}: {', '.join(names)}", file=sys.stderr)
    if summary["degraded_points"]:
        print(
            "Client scheduling was degraded at: "
            f"{', '.join(summary['degraded_points'])}. Latencies at these "
            "points may reflect the load generator, not the server.",
            file=sys.stderr,
        )
    return 0 if summary["clean"] else 1


__all__ = [
    "add_api_client_arguments",
    "add_server_metrics_arguments",
    "add_sweep_api_parser",
    "base_api_config",
    "cmd_sweep_api",
    "is_loopback_endpoint",
    "read_prompts_jsonl",
]
