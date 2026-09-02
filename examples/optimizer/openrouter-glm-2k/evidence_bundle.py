#!/usr/bin/env python3
"""Build and verify the public OpenRouter GLM 2K evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

PUBLIC_DIR = Path(__file__).resolve().parent
MAX_PUBLIC_FILE_BYTES = 4 * 1024 * 1024
MODEL_BUILDS = {
    "z-ai/glm-5.3-flash": "z-ai/glm-5.3-flash-20260826",
    "z-ai/glm-5.3": "z-ai/glm-5.3-20260816",
}
REQUESTS = (
    (
        "flash-structured-r1",
        "flash",
        1,
        "structured",
        "structured-json-profile-extraction-2k-autoregressive",
    ),
    (
        "flash-prose-r1",
        "flash",
        1,
        "prose",
        "prose-reasoning-two-train-gap-2k-autoregressive",
    ),
    (
        "flash-structured-r2",
        "flash",
        2,
        "structured",
        "structured-json-profile-extraction-2k-autoregressive",
    ),
    (
        "flash-prose-r2",
        "flash",
        2,
        "prose",
        "prose-reasoning-two-train-gap-2k-autoregressive",
    ),
    (
        "glm-structured-r1",
        "glm",
        1,
        "structured",
        "structured-json-profile-extraction-2k-autoregressive",
    ),
    (
        "glm-prose-r1",
        "glm",
        1,
        "prose",
        "prose-reasoning-two-train-gap-2k-autoregressive",
    ),
    (
        "glm-structured-r2",
        "glm",
        2,
        "structured",
        "structured-json-profile-extraction-2k-autoregressive",
    ),
    (
        "glm-prose-r2",
        "glm",
        2,
        "prose",
        "prose-reasoning-two-train-gap-2k-autoregressive",
    ),
)
HASHED_FILES = (
    "README.md",
    "budget-ledger.json",
    "budget-plan.json",
    "compare-policy.json",
    "comparison.html",
    "comparison.json",
    "experiment-manifest.json",
    "generation-metadata.json",
    "measurements.json",
    "pricing-manifest.json",
    "pricing-snapshot.json",
)
PUBLIC_FILES = (*HASHED_FILES, "SHA256SUMS")
COPY_FILES = (
    "budget-ledger.json",
    "budget-plan.json",
    "compare-policy.json",
    "comparison.html",
    "pricing-manifest.json",
    "pricing-snapshot.json",
)
FORBIDDEN_PATTERNS = (
    (re.compile(r"/(?:Users|home)/", re.IGNORECASE), "private home path"),
    (re.compile(r"\.cache[/\\]", re.IGNORECASE), "private cache path"),
    (re.compile(r"\bfile://", re.IGNORECASE), "local file URL"),
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]+\b"), "OpenRouter credential"),
    (re.compile(r"\b(?:gen|req)-[A-Za-z0-9_-]{8,}\b"), "provider identifier"),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "email address"),
    (
        re.compile(r"siddhant-git-ai|siddhantkhare2694", re.IGNORECASE),
        "private username",
    ),
)
FORBIDDEN_JSON_KEYS = {
    "collection_dir",
    "command",
    "final_record_path",
    "provider_request_id",
    "rate_limit_headers",
    "raw_prompt",
    "raw_response",
    "response_id",
}


class EvidenceError(ValueError):
    """Raised when the public evidence is unsafe or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{path.name} must be a regular non-symlink file")
    if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
        raise EvidenceError(f"{path.name} exceeds the public file size limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise EvidenceError(f"{path.name} is not valid UTF-8") from exc


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number {value!r} is not allowed")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path), parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"{path.name} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _check_json(value: Any, *, context: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{context} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_JSON_KEYS:
                raise EvidenceError(f"{context}.{key} is a forbidden private field")
            _check_json(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_json(item, context=f"{context}[{index}]")


def _scan_privacy(name: str, text: str) -> None:
    for pattern, label in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            raise EvidenceError(f"{name} contains {label}")


def _source_paths(
    root: Path, model: str, repetition: int, workload: str, run_id: str
) -> dict[str, Path]:
    run = root / "results" / model / f"rep-{repetition}" / workload / "runs" / run_id
    return {
        "verification": run / "verification.json",
        "record": run / "final_record.json",
        "api": run / "collection" / "api_evidence.json",
        "seal": run / "run.json",
    }


def _generation_rows(root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for model in ("flash", "glm"):
        payload = _load_json(
            root / "results" / model / "generation-metadata.sanitized.json"
        )
        for row in payload.get("rows", []):
            if not isinstance(row, dict) or not isinstance(row.get("request_id"), str):
                raise EvidenceError("generation metadata row is invalid")
            rows[row["request_id"]] = row
    return rows


def _sanitize_comparison_paths(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"record_paths", "verification_paths"} and isinstance(item, list):
                value[key] = ["measurements.json" for _path in item]
            else:
                _sanitize_comparison_paths(item)
    elif isinstance(value, list):
        for item in value:
            _sanitize_comparison_paths(item)


def _measurement(
    root: Path,
    request_id: str,
    model: str,
    repetition: int,
    workload: str,
    run_id: str,
    generation: dict[str, Any],
) -> dict[str, Any]:
    paths = _source_paths(root, model, repetition, workload, run_id)
    verification = _load_json(paths["verification"])
    record = _load_json(paths["record"])
    api = _load_json(paths["api"])
    plan = api["plan"]
    return {
        "request_id": request_id,
        "repetition": repetition,
        "workload": {
            "workload_id": verification["workload_id"],
            "workload_version": verification["workload_version"],
            "category": verification["category"],
            "context_tier": verification["context_tier"],
            "decode_mode": verification["decode_mode"],
            "prompt_sha256": verification["verified_prompt_hash"],
        },
        "system": {
            "requested_model_id": verification["api_model_id"],
            "resolved_model_build": generation["model"],
            "gateway": verification["provider"],
            "upstream_provider": generation["provider_name"],
            "route_slug": "z-ai/fp8",
            "quantization": "fp8",
            "runtime": record["runtime"],
        },
        "request_plan": {
            "schema_version": plan["schema_version"],
            "method": plan["method"],
            "endpoint_origin": plan["endpoint_origin"],
            "endpoint_path": plan["endpoint_path"],
            "endpoint_query_keys": plan["endpoint_query_keys"],
            "messages": plan["messages"],
            "request_parameters": plan["request_parameters"],
            "provider_extensions": plan["provider_extensions"],
            "request_timeout_seconds": plan["request_timeout_seconds"],
            "retained_event_limit": plan["retained_event_limit"],
            "config_hash": plan["config_hash"],
            "workload_hash": plan["workload_hash"],
        },
        "verification": {
            key: verification[key]
            for key in (
                "schema_version",
                "run_id",
                "status",
                "reason",
                "recorded_prompt_hash",
                "verified_prompt_hash",
                "run_binding_hash",
                "resumed",
                "outcome_success",
                "quality_score",
                "total_ms",
                "started_at",
                "ended_at",
                "artifacts_verified",
            )
        },
        "final_record": {
            "schema_version": record["schema_version"],
            "run_id": record["run_id"],
            "model": record["model"],
            "runtime": record["runtime"],
            "repetition": record["repetition"],
            "timing": record["timing"],
            "tokens": record["tokens"],
            "outcome": record["outcome"],
            "error": record["error"],
        },
        "api_evidence": {
            "schema_version": api["schema_version"],
            "collected_at": api["collected_at"],
            "success": api["success"],
            "response_model": api["response_model"],
            "finish_reason": api["finish_reason"],
            "finish_reason_classification": api["finish_reason_classification"],
            "finish_reason_code": api["finish_reason_code"],
            "usage": api["usage"],
            "timeline": api["timeline"],
            "statistics": api["statistics"],
            "stream_terminated_with_done": api["stream_terminated_with_done"],
            "stream_had_unterminated_event": api["stream_had_unterminated_event"],
            "reasoning_content_returned": api["reasoning_content_returned"],
            "reasoning_text_persisted": api["reasoning_text_persisted"],
            "failure": api["failure"],
        },
        "generation_metadata": {
            key: generation[key]
            for key in (
                "model",
                "provider_name",
                "service_tier",
                "native_tokens_prompt",
                "native_tokens_completion",
                "native_tokens_cached",
                "native_tokens_reasoning",
                "total_cost",
            )
        },
        "source_integrity": {
            f"{name}_sha256": _sha256(path) for name, path in paths.items()
        },
    }


def _code_hashes() -> dict[str, str]:
    root = PUBLIC_DIR.parents[2]
    paths = (
        "llmtracefx/optimizer/cli.py",
        "llmtracefx/optimizer/collectors/openai_api.py",
        "llmtracefx/optimizer/workloads/api_budget.py",
        "llmtracefx/optimizer/workloads/api_verify.py",
    )
    return {path: _sha256(root / path) for path in paths}


def build(root: Path) -> None:
    generation = _generation_rows(root)
    measurements = [
        _measurement(
            root,
            request_id,
            model,
            repetition,
            workload,
            run_id,
            generation[request_id],
        )
        for request_id, model, repetition, workload, run_id in REQUESTS
    ]
    _write_json(
        PUBLIC_DIR / "measurements.json",
        {"schema_version": "1", "rows": measurements},
    )
    _write_json(
        PUBLIC_DIR / "generation-metadata.json",
        {
            "schema_version": "1",
            "source_url": "https://openrouter.ai/api/v1/generation",
            "provider_identifiers_persisted": False,
            "rows": [
                {
                    "request_id": request_id,
                    **{
                        key: generation[request_id][key]
                        for key in (
                            "model",
                            "provider_name",
                            "service_tier",
                            "native_tokens_prompt",
                            "native_tokens_completion",
                            "native_tokens_cached",
                            "native_tokens_reasoning",
                            "total_cost",
                        )
                    },
                }
                for request_id, *_rest in REQUESTS
            ],
        },
    )

    for name in COPY_FILES:
        shutil.copyfile(root / name, PUBLIC_DIR / name)

    comparison = _load_json(root / "comparison.json")
    comparison["results_dirs"] = ["measurements.json"]
    comparison["pricing"]["manifest_path"] = "pricing-manifest.json"
    _sanitize_comparison_paths(comparison)
    _write_json(PUBLIC_DIR / "comparison.json", comparison)

    ledger = _load_json(root / "budget-ledger.json")
    starting = _load_json(root / "starting-account.json")
    ending = _load_json(root / "ending-account.json")
    account_delta = float(ending["usage_usd"]) - float(starting["usage_usd"])
    by_model: dict[str, dict[str, float]] = {}
    for model_id in MODEL_BUILDS:
        rows = [
            row
            for row in measurements
            if row["system"]["requested_model_id"] == model_id
        ]
        by_model[model_id] = {
            "measured_requests": len(rows),
            "passing_requests": sum(
                row["verification"]["quality_score"] == 1.0 for row in rows
            ),
            "mean_total_ms": sum(row["verification"]["total_ms"] for row in rows)
            / len(rows),
            "provider_reported_cost_usd": sum(
                row["api_evidence"]["usage"]["cost_usd"] for row in rows
            ),
        }
    manifest = {
        "schema_version": "1",
        "evidence_id": "openrouter-zai-glm-2k-20260902",
        "run": {
            "base_checkout_commit": ("a6077adaf7135e2a2e360aeae4a73b6b411b3493"),
            "execution_code_sha256": _code_hashes(),
            "paid_inference_requests": 8,
            "warmup_requests": 0,
            "measured_repetitions_per_workload_model": 2,
            "automatic_retries": 0,
            "status": "completed",
        },
        "systems": {
            "gateway": "OpenRouter",
            "requested_model_ids": list(MODEL_BUILDS),
            "resolved_model_builds": MODEL_BUILDS,
            "upstream_provider": "Z.AI",
            "route_slug": "z-ai/fp8",
            "quantization": "fp8",
            "fallbacks_allowed": False,
            "provider_parameter_support_required": True,
        },
        "matrix_contract": {
            "context_tier": "2k",
            "workloads": [
                "structured-json-profile-extraction@1",
                "prose-reasoning-two-train-gap@1",
            ],
            "max_output_tokens": 96,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": None,
            "seed_limitation": (
                "The pinned first-party Z.AI endpoints did not advertise seed "
                "support, so no seed was sent. Both hosted systems otherwise "
                "used the same sampling contract."
            ),
            "reasoning_effort": "low",
            "reasoning_mandatory": True,
            "unsafe_code_completion_requested": False,
            "ttft_definition": "first non-empty visible content at the client",
        },
        "budget": {
            "authorized_total_usd": ledger["authorized_total_usd"],
            "planned_request_count": ledger["planned_request_count"],
            "planned_ceiling_usd": ledger["planned_ceiling_usd"],
            "input_token_ceiling_method": (
                "UTF-8 byte length of the exact rendered prompt plus a fixed "
                "4,096-token reserve for provider chat framing and special "
                "tokens; every input token is then priced at the full prompt "
                "rate even when cache reads could be cheaper"
            ),
            "unallocated_reserve_usd": format(
                float(ledger["authorized_total_usd"])
                - float(ledger["planned_ceiling_usd"]),
                ".12f",
            ),
            "provider_reported_request_total_usd": ledger["cumulative_accounted_usd"],
            "manifest_computed_request_total_usd": format(
                sum(
                    float(entry["computed_observed_cost_usd"])
                    for entry in ledger["entries"]
                ),
                ".12f",
            ),
            "remaining_authorization_usd": ledger["remaining_authorized_usd"],
            "post_run_account_usage_delta_usd": format(account_delta, ".12f"),
            "account_delta_scope_limitation": (
                "The single immediate post-run key-usage query reflected only "
                "0.00039958 USD, equal to the Flash calls, while all eight final "
                "SSE usage blocks reported 0.00615262 USD. The account endpoint "
                "can lag and can include unrelated concurrent activity; its "
                "delta is separate corroborating evidence, not the experiment "
                "request total."
            ),
        },
        "results": by_model,
        "local_qwen3_8b_context": {
            "source_commit": ("a6077adaf7135e2a2e360aeae4a73b6b411b3493"),
            "source_pr": 56,
            "tier": "2k",
            "pass_rate": 1.0,
            "mean_total_ms": 2595.6080727501103,
            "correct_cases_per_minute": 23.115970638983462,
            "direct_ranking_excluded": True,
            "reasons": [
                "different model and system identity",
                "local MLX timing excludes hosted network and provider queueing",
                "local run disabled thinking while hosted GLM reasoning is mandatory",
                "local run used seed 20260831 while the pinned hosted endpoint did not advertise seed support",
                "local evidence is exploratory and made no clean-boot assertion",
            ],
        },
        "claims": {
            "ranking_scope": (
                "constraints-first, single-objective ranking within each exact "
                "hosted workload stratum only"
            ),
            "universal_winner_claimed": False,
            "prompts_or_responses_published": False,
            "reasoning_text_published": False,
            "credentials_or_account_identifiers_published": False,
        },
        "limitations": [
            "Two repetitions per workload/model are directional evidence, not a population latency study.",
            "Client timing combines network, gateway, queueing, and model execution.",
            "OpenRouter and upstream conditions can change after the captured timestamp.",
            "Mandatory reasoning used low effort; these results do not characterize high or max effort.",
            "Provider-reported zero reasoning tokens do not prove the model performed no hidden internal computation.",
            "Prompt caching affected some second repetitions and is reported per row rather than normalized away.",
            "The local Qwen3-8B evidence is contextual only and was excluded from direct ranking.",
        ],
    }
    _write_json(PUBLIC_DIR / "experiment-manifest.json", manifest)

    checksums = "".join(
        f"{_sha256(PUBLIC_DIR / name)}  {name}\n" for name in HASHED_FILES
    )
    (PUBLIC_DIR / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    verify()


def verify() -> None:
    actual = tuple(
        sorted(
            path.name
            for path in PUBLIC_DIR.iterdir()
            if path.is_file() and path.name != Path(__file__).name
        )
    )
    if actual != tuple(sorted(PUBLIC_FILES)):
        raise EvidenceError("public evidence file set is incomplete or unexpected")

    for name in PUBLIC_FILES:
        text = _read_text(PUBLIC_DIR / name)
        _scan_privacy(name, text)
        if name.endswith(".json"):
            value = _load_json(PUBLIC_DIR / name)
            _check_json(value)

    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in _read_text(PUBLIC_DIR / "SHA256SUMS").splitlines()
    }
    if set(expected) != set(HASHED_FILES):
        raise EvidenceError("SHA256SUMS does not name the exact public file set")
    for name in HASHED_FILES:
        if expected[name] != _sha256(PUBLIC_DIR / name):
            raise EvidenceError(f"checksum mismatch for {name}")

    measurements = _load_json(PUBLIC_DIR / "measurements.json")
    rows = measurements.get("rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise EvidenceError("measurements must contain exactly eight rows")
    request_ids = {row.get("request_id") for row in rows if isinstance(row, dict)}
    if request_ids != {request_id for request_id, *_rest in REQUESTS}:
        raise EvidenceError("measurement request identities drifted")
    for row in rows:
        if row["verification"]["status"] != "completed":
            raise EvidenceError("every published row must be completed")
        if row["verification"]["quality_score"] != 1.0:
            raise EvidenceError("every published row must pass its evaluator")
        if row["api_evidence"]["reasoning_text_persisted"] is not False:
            raise EvidenceError("reasoning text must never be persisted")
        requested = row["system"]["requested_model_id"]
        if row["system"]["resolved_model_build"] != MODEL_BUILDS[requested]:
            raise EvidenceError("resolved model build does not match the catalog pin")
        if row["system"]["upstream_provider"] != "Z.AI":
            raise EvidenceError("row did not resolve to the pinned Z.AI provider")

    ledger = _load_json(PUBLIC_DIR / "budget-ledger.json")
    seal = ledger.pop("ledger_sha256", None)
    actual_seal = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                ledger, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
    if seal != actual_seal:
        raise EvidenceError("budget ledger seal does not verify")
    if ledger["planned_request_count"] != 8:
        raise EvidenceError("budget ledger request count drifted")
    if any(entry["status"] != "completed" for entry in ledger["entries"]):
        raise EvidenceError("budget ledger contains an incomplete request")
    if float(ledger["cumulative_accounted_usd"]) > 5.0:
        raise EvidenceError("budget ledger exceeds the authorized USD 5 cap")

    manifest = _load_json(PUBLIC_DIR / "experiment-manifest.json")
    if manifest["claims"]["prompts_or_responses_published"]:
        raise EvidenceError("public manifest claims prompt or response publication")
    comparison = _load_json(PUBLIC_DIR / "comparison.json")
    if comparison["results_dirs"] != ["measurements.json"]:
        raise EvidenceError("comparison report still carries private result paths")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    subparsers.add_parser("verify")
    args = parser.parse_args()
    if args.command == "build":
        build(args.source)
    else:
        verify()
    print("OpenRouter GLM public evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
