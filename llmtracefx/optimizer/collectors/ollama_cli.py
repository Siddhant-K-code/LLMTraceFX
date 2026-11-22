"""``collect-ollama``: one native Ollama request recorded as canonical evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ._shared import atomic_write_text
from .ollama import (
    API_CHAT,
    API_PATHS,
    DEFAULT_BASE_URL,
    EVIDENCE_FILE_NAME,
    OllamaCollectionConfig,
    OllamaError,
    OllamaSettings,
    collect_ollama,
    request_plan,
)
from .openai_api import UrllibStreamingTransport, redact_text_for_dry_run

_PROG = "llmtracefx-optimizer"


def add_collect_ollama_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = subparsers.add_parser(
        "collect-ollama",
        help=(
            "Stream one request through Ollama's native API and record its "
            "server timers beside client TTFT"
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
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="UTF-8 prompt file; the prompt is hashed, not copied into the record",
    )
    parser.add_argument("--system-prompt-file", default=None)
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
        help="options.num_predict; Ollama's default is no limit",
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
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional label for the model build, such as its digest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write request_plan.json and send nothing",
    )
    parser.set_defaults(func=cmd_collect_ollama)
    return parser


def _argv(args: argparse.Namespace) -> tuple[str, ...]:
    argv = [_PROG, "collect-ollama"]
    for flag, value in (
        ("--run-id", args.run_id),
        ("--endpoint", args.endpoint),
        ("--model", args.model),
        ("--api", args.api),
        ("--prompt-file", args.prompt_file),
        ("--system-prompt-file", args.system_prompt_file),
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
        ("--model-revision", args.model_revision),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    if args.dry_run:
        argv.append("--dry-run")
    return tuple(argv)


def _config(args: argparse.Namespace) -> OllamaCollectionConfig:
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
        think=None if args.think is None else args.think == "true",
        request_timeout_seconds=args.request_timeout,
        credential_env_var=args.api_key_env,
    )
    system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )
    return OllamaCollectionConfig(
        run_id=args.run_id,
        settings=settings,
        prompt=Path(args.prompt_file).read_text(encoding="utf-8"),
        output_dir=Path(args.output_dir),
        command_argv=_argv(args),
        system_prompt=system_prompt,
        model_revision=args.model_revision,
    )


def cmd_collect_ollama(args: argparse.Namespace) -> int:
    from ..cli import _scrubbed_detail

    def fail(prefix: str, exc: BaseException) -> int:
        print(f"{prefix}: {_scrubbed_detail(args.api_key_env, exc)}", file=sys.stderr)
        return 1

    try:
        config = _config(args)
    except (OSError, UnicodeError, OllamaError) as exc:
        return fail("Failed to configure Ollama collection", exc)

    credential = (
        os.environ.get(args.api_key_env, "").strip() or None
        if args.api_key_env
        else None
    )
    if args.dry_run:
        payload = {
            "dry_run": True,
            "network_request_performed": False,
            "credential_env_var_present": (
                None if args.api_key_env is None else credential is not None
            ),
            "plan": {
                **request_plan(config.settings),
                "command_argv": list(config.command_argv),
            },
        }
        text = redact_text_for_dry_run(
            json.dumps(payload, indent=2, allow_nan=False), credential
        )
        try:
            atomic_write_text(config.output_dir / "request_plan.json", text + "\n")
        except OSError as exc:
            return fail("Failed to write request plan", exc)
        print(text)
        return 0

    try:
        result = collect_ollama(
            config, transport=UrllibStreamingTransport(), environ=os.environ
        )
    except (OSError, OllamaError) as exc:
        return fail("Failed to collect Ollama evidence", exc)

    record = result.record
    print(f"Ollama experiment record written to {config.output_dir / 'record.json'}")
    print(f"Server timers written to {config.output_dir / EVIDENCE_FILE_NAME}")
    if record.outcome.success:
        return 0
    message = (
        "unknown failure"
        if record.error is None
        else (f"{record.error.category}: {record.error.message}")
    )
    return fail("Ollama request failed", RuntimeError(message))


__all__ = ["add_collect_ollama_parser", "cmd_collect_ollama"]
