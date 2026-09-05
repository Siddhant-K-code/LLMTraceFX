"""Command-line interface for evidence-first cache auditing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llmtracefx.evidence.core import canonical_json

from .adapters.base import CacheAuditAdapter
from .adapters.reference import ReferenceCacheAdapter
from .bundle import (
    CacheAuditBundleError,
    read_bundle,
    sanitize_bundle_records,
    verify_bundle,
    write_bundle,
)
from .runner import run_audit
from .schema import CacheConfig, PublicationMode, RequestSpec
from .workloads import adversarial_requests


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-cache-audit",
        description=(
            "Verify cache identity, reuse, prompt work, timing, memory, and "
            "correctness without conflating them"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="write the deterministic adversarial workload"
    )
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--block-size", type=int, default=4)

    run_parser = subparsers.add_parser(
        "run", help="run a cache audit and write an evidence bundle"
    )
    run_parser.add_argument(
        "--backend", choices=("reference", "mlx"), default="reference"
    )
    run_parser.add_argument("--workload", type=Path, default=None)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, default=None)
    run_parser.add_argument("--model-id", default=None)
    run_parser.add_argument("--tokenizer-id", default=None)
    run_parser.add_argument(
        "--publication-mode",
        choices=tuple(mode.value for mode in PublicationMode),
        default=PublicationMode.PRIVATE.value,
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--max-entries", type=int, default=32)
    run_parser.add_argument("--max-bytes", type=int, default=1 << 30)

    verify_parser = subparsers.add_parser(
        "verify", help="verify a bundle entirely offline"
    )
    verify_parser.add_argument("bundle", type=Path)

    report_parser = subparsers.add_parser(
        "report", help="verify a bundle and return its deterministic report paths"
    )
    report_parser.add_argument("bundle", type=Path)

    sanitize_parser = subparsers.add_parser(
        "sanitize", help="create a public-redacted bundle"
    )
    sanitize_parser.add_argument("bundle", type=Path)
    sanitize_parser.add_argument("--output-dir", type=Path, required=True)

    capabilities_parser = subparsers.add_parser(
        "capabilities", help="show backend observations and refusal reasons"
    )
    capabilities_parser.add_argument(
        "--backend", choices=("reference", "mlx", "vllm"), required=True
    )
    return parser


def _load_workload(path: Path | None) -> tuple[RequestSpec, ...]:
    if path is None:
        return adversarial_requests()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read workload: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "requests"}:
        raise ValueError("workload must contain only schema_version and requests")
    if value["schema_version"] != "1" or not isinstance(value["requests"], list):
        raise ValueError("unsupported workload schema")
    return tuple(RequestSpec.from_dict(item) for item in value["requests"])


def _capabilities(backend: str) -> dict[str, object]:
    if backend == "reference":
        return ReferenceCacheAdapter().capabilities().to_dict()
    if backend == "mlx":
        from .adapters.mlx import ProductionMLXRuntime, check_mlx_capabilities

        return check_mlx_capabilities(ProductionMLXRuntime()).to_dict()
    from .adapters.vllm import VLLMCapabilityConfig, assess_vllm_capabilities

    return assess_vllm_capabilities(VLLMCapabilityConfig.from_environment()).to_dict()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "compile":
        requests = adversarial_requests(block_size=args.block_size)
        payload = {
            "schema_version": "1",
            "requests": [request.to_dict() for request in requests],
        }
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(payload), encoding="utf-8")
        return {"compiled": True, "requests": len(requests), "output": str(args.output)}

    if args.command == "run":
        requests = _load_workload(args.workload)
        adapter: CacheAuditAdapter
        if args.backend == "reference":
            adapter = ReferenceCacheAdapter(
                max_entries=args.max_entries,
                max_bytes=args.max_bytes,
            )
            backend_version = "1"
            model_id = args.model_id or "synthetic-tiny-model"
            tokenizer_id = args.tokenizer_id or "integer-tokenizer-v1"
            model_artifact_digest = None
            cache_type = "token_trie"
        else:
            if args.model_path is None:
                raise ValueError("--model-path is required for the MLX backend")
            if args.model_id is None or args.tokenizer_id is None:
                raise ValueError(
                    "--model-id and --tokenizer-id are required for the MLX backend"
                )
            from .adapters.mlx import (
                REQUIRED_MLX_LM_VERSION,
                MLXLocalCacheAdapter,
            )

            adapter = MLXLocalCacheAdapter(
                model_path=args.model_path,
                max_cache_entries=args.max_entries,
                max_cache_bytes=args.max_bytes,
            )
            backend_version = REQUIRED_MLX_LM_VERSION
            model_id = args.model_id
            tokenizer_id = args.tokenizer_id
            model_artifact_digest = adapter.model_artifact_digest
            cache_type = "mlx_lru_prompt_cache"
        manifest, records = run_audit(
            adapter=adapter,
            requests=requests,
            cache_config=CacheConfig(
                namespace_id="synthetic-namespaces",
                cache_type=cache_type,
                max_entries=args.max_entries,
                max_bytes=args.max_bytes,
            ),
            output_dir=args.output_dir,
            backend_version=backend_version,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            model_artifact_digest=model_artifact_digest,
            publication_mode=PublicationMode(args.publication_mode),
            seed=args.seed,
        )
        return {
            "completed": True,
            "run_id": manifest.run_id,
            "requests": len(records),
            "output_dir": str(args.output_dir),
        }

    if args.command == "verify":
        return {"verified": True, **verify_bundle(args.bundle)}

    if args.command == "report":
        result = verify_bundle(args.bundle)
        return {
            "verified": True,
            **result,
            "html": str(args.bundle / "report.html"),
            "reuse_svg": str(args.bundle / "reuse-alignment.svg"),
        }

    if args.command == "sanitize":
        manifest, records = read_bundle(args.bundle)
        public_manifest, public_records = sanitize_bundle_records(manifest, records)
        write_bundle(args.output_dir, public_manifest, public_records)
        result = verify_bundle(args.output_dir)
        return {"sanitized": True, "verified": True, **result}

    return _capabilities(args.backend)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = _run(args)
    except (CacheAuditBundleError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
