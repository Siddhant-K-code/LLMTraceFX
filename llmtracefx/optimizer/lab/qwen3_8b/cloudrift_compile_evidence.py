"""Build and verify the CloudRift RTX 4090 vLLM compilation evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ...workloads.catalog import workload_by_id
from ...workloads.evaluators import evaluate_workload
from .cloudrift_runner import (
    BASE_IMAGE_REFERENCE,
    DERIVED_IMAGE_ID,
    RUNTIME_PINS,
    SAMPLING,
)
from .vllm_compile import (
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    canonical_json,
    workload_descriptors,
)

BUNDLE_FILES = (
    "README.md",
    "experiment-contract.json",
    "pricing-snapshot.json",
    "model-inventory.json",
    "runtime-image.json",
    "workload-contract.json",
    "lifecycle-records.jsonl",
    "request-records.jsonl",
    "correctness-report.json",
    "break-even.json",
    "cost-ledger.json",
    "teardown-report.json",
    "claim-matrix.json",
    "report.html",
    "break-even.svg",
    "evidence_bundle.py",
    "SHA256SUMS",
)
JSON_FILES = tuple(name for name in BUNDLE_FILES if name.endswith(".json"))
JSONL_FILES = tuple(name for name in BUNDLE_FILES if name.endswith(".jsonl"))
HASHED_FILES = tuple(sorted(set(BUNDLE_FILES) - {"SHA256SUMS"}))
RATE_USD_PER_HOUR = Decimal("0.39")
HARD_CAP_USD = Decimal("5.00")
EXPERIMENT_CUTOFF_HOURS = Decimal("8")
BOOT_AT = "2026-09-03T15:34:29Z"
SHUTDOWN_SCHEDULED_AT = "2026-09-03T16:34:57Z"
CONSOLE_TERMINATED_AT = "2026-09-03T22:19:00+05:30"
EXECUTION_BASE_HEAD = "741dc5b27a4603a9d9d93f531d4de4f31703ac6e"
COLLECTION_SOURCE_COMMIT = "9c0879351cc3e4f294b5c827d74dfc00182d53bb"
EXECUTED_RUNNER_SHA256 = (
    "sha256:42e3414895133a39a48996543ef0f980e02c12699b3d923e7c6c75819ca290fb"
)
EXPECTED_LIFECYCLE_RECORDS_SHA256 = (
    "ea05631d31b231426a3930a1c05c42826d823b15551f11cb7b4431f875f66f92"
)
EXPECTED_REQUEST_RECORDS_SHA256 = (
    "b06d3ebf613c7ac00676d96f7e699bcbe18cf1635b639ba68570fa2e3c86b42e"
)
EXPECTED_TEARDOWN_CAPTURE_SHA256 = (
    "sha256:2e049a0662c6c009fc2123f3c421438db8d4859a25c93cce240c6ebe488e0d4f"
)
EXPECTED_PROMPT_IDENTITIES = {
    "2k/structured-json-profile-extraction": (
        1611,
        "sha256:22ccec7442fcd2e0b3f1c4cb06adc68bef8b398b7cf6c4d86cd689140682b59a",
    ),
    "2k/prose-reasoning-two-train-gap": (
        1615,
        "sha256:adab92bdf2ab2e0f3cafd7b55748c2ac8833bf77d63eb1bd06f5c623c511971b",
    ),
    "8k/structured-json-profile-extraction": (
        6371,
        "sha256:cca1d3efb2b208bf207ce57cd1354be1fa950062aef41cd7294d72229d8a64c5",
    ),
    "8k/prose-reasoning-two-train-gap": (
        6375,
        "sha256:58dfa54afd6ca927b45c6b2334506796627099b9b89e5e94fb4a156d30242009",
    ),
    "16k/structured-json-profile-extraction": (
        12695,
        "sha256:c8990da4edda07fd4866c77f55363ae28d58cb15f38f92cf047694c9ca56477b",
    ),
    "16k/prose-reasoning-two-train-gap": (
        12699,
        "sha256:b1ca08f9a8af3588794a5022765bfcce6ea95faeae07d5fb7653144519af6842",
    ),
}
MEASURED_RUNNER_LIMITATION = (
    "The measured runner verified the staging and prompt receipts independently "
    "and mounted model and state read-only, but it did not rehash the live model "
    "or cross-check both receipt hashes before each measured cell."
)
HOST_RECEIPT_LIMITATION = (
    "No independent host orchestration receipt was retained for the fresh-container, "
    "cache-drop, timeout-wrapper, bind-mount, network, or Docker image-inspection "
    "controls. The public records prove ordered non-overlapping processes and no "
    "warmup requests, but not those additional host controls."
)
PROVENANCE = (
    "client_observed",
    "vllm",
    "cuda",
    "cloudrift_user_observed",
    "model_reported",
    "derived",
)
_PRIVATE_PATTERNS = (
    (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
    (
        re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IP address",
    ),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (
        re.compile(
            r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)" r"[A-Za-z0-9_-]{8,}\b"
        ),
        "credential-shaped value",
    ),
    (re.compile(r"\bGPU-[0-9a-f-]{16,}\b", re.I), "GPU UUID"),
)

README = """# Qwen3 8B vLLM compilation break-even on CloudRift

This bundle compares eager execution with vLLM compilation and CUDA graphs on
one fixed CloudRift RTX 4090. Both cells used the same immutable runtime, exact
model revision, token arrays, request order, and bounded generation settings.

Compilation did not repay its initialization cost within the 12 observed
requests. Repeating the exact request sequence without any other change yields
a modeled crossing at request 113. That crossing is an extrapolation, not an
observed request.

Twenty-two of 24 responses passed their deterministic workload evaluators.
Eager execution returned an incorrect `3.5` answer for both 16K prose requests;
compiled execution returned correct answers for all 12 requests. Eight of 12
paired outputs had identical token IDs. The boot-to-console-termination window
implies $0.484358 at the observed list rate. This remains a lower bound because
the provisioning-to-boot interval is unavailable. Provider-reported spend is
unavailable. The experiment containers, GPU processes, model data, runtime
images, result caches, and temporary public key were removed before the scheduled
OS shutdown. OS shutdown itself was not observed. The user confirmed CloudRift
console termination separately.

The collection runner verified the staging and prompt receipts independently,
and both cells mounted the model and state read-only. It did not rehash the live
16 GB model or cross-check the two receipt hashes before each measured cell.
The public verifier binds the retained inventory, prompt arrays, and collection
source, but this collection limitation cannot be retroactively removed.
No independent host orchestration receipt was retained for the fresh-container,
cache-drop, timeout-wrapper, bind-mount, network, or Docker image-inspection
controls. The records prove ordered non-overlapping processes and no warmup
requests, but those additional host controls remain unverified.

Run `python evidence_bundle.py verify` from this directory to verify the closed
file set, checksums, privacy rules, model and runtime pins, request contract,
correctness, break-even arithmetic, cost scope, and teardown status.
"""

WRAPPER = '''"""Verify the committed CloudRift vLLM compilation evidence."""

import sys
from pathlib import Path

from llmtracefx.optimizer.lab.qwen3_8b.cloudrift_compile_evidence import verify_bundle

if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: evidence_bundle.py verify")
    verify_bundle(Path(__file__).resolve().parent)
    print("CloudRift vLLM compilation evidence verified")
'''


class CloudRiftEvidenceError(ValueError):
    """Raised when CloudRift evidence is incomplete, inconsistent, or private."""


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return "sha256:" + _sha256_bytes(canonical_json(value).encode())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            CloudRiftEvidenceError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(value, dict):
        raise CloudRiftEvidenceError(f"{path.name} must contain an object")
    return value


def _official_inventory_files() -> list[dict[str, Any]]:
    manifest = _read_json(
        Path(__file__).parent / "data" / "qwen3-8b-conversion-manifest-v1.json"
    )
    source = manifest["source"]
    if not isinstance(source, dict):
        raise CloudRiftEvidenceError("packaged official source manifest is invalid")
    if (
        source["official_id"] != MODEL_ID
        or source["official_revision"] != MODEL_REVISION
        or source["expected_source_bytes"] != EXPECTED_MODEL_BYTES
    ):
        raise CloudRiftEvidenceError(
            "packaged official source manifest identity mismatch"
        )
    files = source["files"]
    if not isinstance(files, list) or any(not isinstance(item, dict) for item in files):
        raise CloudRiftEvidenceError("packaged official inventory is invalid")
    return [dict(item) for item in files]


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    observed = body.pop(field, None)
    if observed != _sha256_json(body):
        raise CloudRiftEvidenceError(f"{field} does not verify")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise CloudRiftEvidenceError(f"{path.name} contains a non-object")
        values.append(value)
    return values


def _request_records(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    descriptors = workload_descriptors()
    for cell in cells:
        mode = cell["mode"]
        requests = cell["requests"]
        if len(requests) != 12:
            raise CloudRiftEvidenceError("each cell must contain 12 requests")
        for descriptor, raw in zip(descriptors, requests, strict=True):
            if any(raw[key] != value for key, value in descriptor.to_dict().items()):
                raise CloudRiftEvidenceError("request descriptor drifted")
            outcome = evaluate_workload(
                workload_by_id(descriptor.workload_id), raw["decoded_output"]
            )
            evaluation = {
                "cell_id": cell["cell_id"],
                "ordinal": descriptor.ordinal,
                "request_id": descriptor.request_id,
                "success": outcome.success,
                "quality_score": outcome.quality_score,
                "quality_metric": outcome.quality_metric,
                "notes": outcome.notes,
                "evaluator": "evaluate_workload",
            }
            evaluations.append(evaluation)
            records.append(
                {
                    **descriptor.to_dict(),
                    "cell_id": cell["cell_id"],
                    "mode": mode,
                    "started_at": raw["started_at"],
                    "ended_at": raw["ended_at"],
                    "latency_seconds": raw["latency_seconds"],
                    "ttft_seconds": raw["ttft_seconds"],
                    "input_token_count": raw["input_token_count"],
                    "input_token_ids_sha256": raw["input_token_ids_sha256"],
                    "output_token_count": raw["output_token_count"],
                    "output_token_ids": raw["output_token_ids"],
                    "decoded_output": raw["decoded_output"],
                    "finish_reason": raw["finish_reason"],
                    "output_tokens_per_second": raw["output_tokens_per_second"],
                    "terminal": raw["terminal"],
                    "correctness": outcome.success,
                    "provenance": {
                        "started_at": "client_observed",
                        "ended_at": "client_observed",
                        "latency_seconds": "client_observed",
                        "ttft_seconds": "vllm",
                        "input_token_count": "derived",
                        "input_token_ids_sha256": "derived",
                        "output_token_count": "model_reported",
                        "output_token_ids": "model_reported",
                        "decoded_output": "model_reported",
                        "finish_reason": "model_reported",
                        "output_tokens_per_second": "derived",
                        "correctness": "derived",
                    },
                }
            )
    return records, evaluations


def _correctness_report(
    requests: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_mode = {
        mode: [record for record in requests if record["mode"] == mode]
        for mode in ("eager", "compiled")
    }
    pairs = zip(by_mode["eager"], by_mode["compiled"], strict=True)
    mismatches = [
        eager["ordinal"]
        for eager, compiled in pairs
        if eager["output_token_ids"] != compiled["output_token_ids"]
    ]
    return {
        "schema_version": "1",
        "total_requests": len(evaluations),
        "successful_requests": sum(item["success"] for item in evaluations),
        "all_requests_correct": all(item["success"] for item in evaluations),
        "successful_requests_by_mode": {
            mode: sum(
                item["success"]
                for item in evaluations
                if item["cell_id"] == f"rtx4090-{mode}"
            )
            for mode in ("eager", "compiled")
        },
        "paired_output_token_identity_matches": 12 - len(mismatches),
        "paired_output_token_identity_mismatched_ordinals": mismatches,
        "evaluations": list(evaluations),
    }


def _lifecycle_records(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell in cells:
        host = _dt(cell["host_invocation_started_at"])
        process = _dt(cell["process_started_at"])
        initialization = _dt(cell["initialization_started_at"])
        ready = _dt(cell["initialization_ready_at"])
        ended = _dt(cell["ended_at"])
        hardware = dict(cell["hardware"])
        hardware.pop("gpu_uuid_sha256", None)
        records.append(
            {
                "cell_id": cell["cell_id"],
                "mode": cell["mode"],
                "host_invocation_started_at": cell["host_invocation_started_at"],
                "process_started_at": cell["process_started_at"],
                "initialization_started_at": cell["initialization_started_at"],
                "initialization_ready_at": cell["initialization_ready_at"],
                "ended_at": cell["ended_at"],
                "host_to_process_seconds": (process - host).total_seconds(),
                "initialization_seconds": (ready - initialization).total_seconds(),
                "request_phase_seconds": (ended - ready).total_seconds(),
                "host_lifecycle_seconds": (ended - host).total_seconds(),
                "compilation_seconds": cell["compilation_seconds"],
                "compilation_seconds_unobservable_reason": cell[
                    "compilation_seconds_unobservable_reason"
                ],
                "cuda_graph_seconds": cell["cuda_graph_seconds"],
                "cuda_graph_seconds_unobservable_reason": cell[
                    "cuda_graph_seconds_unobservable_reason"
                ],
                "peak_gpu_memory_mib": cell["peak_gpu_memory_mib"],
                "hardware": hardware,
                "runtime": cell["runtime"],
                "runtime_image": cell["runtime_image"],
                "resolved_execution_config": cell["resolved_execution_config"],
                "terminal": cell["terminal"],
                "provenance": {
                    "host_lifecycle": "client_observed",
                    "initialization": "client_observed",
                    "compile_configuration": "vllm",
                    "compilation_seconds": "vllm",
                    "cuda_graph_seconds": "vllm",
                    "peak_gpu_memory_mib": "cuda",
                    "hardware": "cuda",
                },
            }
        )
    return records


def _break_even(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_mode = {cell["mode"]: cell for cell in cells}
    eager = by_mode["eager"]
    compiled = by_mode["compiled"]
    cumulative: list[dict[str, Any]] = []
    observed: int | None = None
    for index in range(12):
        eager_seconds = (
            _dt(eager["requests"][index]["ended_at"])
            - _dt(eager["initialization_started_at"])
        ).total_seconds()
        compiled_seconds = (
            _dt(compiled["requests"][index]["ended_at"])
            - _dt(compiled["initialization_started_at"])
        ).total_seconds()
        if observed is None and compiled_seconds <= eager_seconds:
            observed = index + 1
        cumulative.append(
            {
                "request_count": index + 1,
                "eager_seconds": eager_seconds,
                "compiled_seconds": compiled_seconds,
                "compiled_minus_eager_seconds": compiled_seconds - eager_seconds,
            }
        )
    eager_init = (
        _dt(eager["initialization_ready_at"]) - _dt(eager["initialization_started_at"])
    ).total_seconds()
    compiled_init = (
        _dt(compiled["initialization_ready_at"])
        - _dt(compiled["initialization_started_at"])
    ).total_seconds()
    initialization_penalty = compiled_init - eager_init
    savings = [
        eager_request["latency_seconds"] - compiled_request["latency_seconds"]
        for eager_request, compiled_request in zip(
            eager["requests"], compiled["requests"], strict=True
        )
    ]
    cycle_saving = sum(savings)
    extrapolated: int | None = None
    if initialization_penalty <= 0:
        extrapolated = 0
    elif cycle_saving > 0:
        for request_count in range(1, 1_000_001):
            cycles, prefix = divmod(request_count, 12)
            total_saving = cycles * cycle_saving + sum(savings[:prefix])
            if total_saving >= initialization_penalty:
                extrapolated = request_count
                break
    return {
        "schema_version": "1",
        "observed_break_even_request_count": observed,
        "observed_lower_bound_request_count": 12 if observed is None else None,
        "cumulative_initialization_to_terminal": cumulative,
        "initialization_penalty_seconds": initialization_penalty,
        "exact_cycle_request_count": 12,
        "exact_cycle_latency_saving_seconds": cycle_saving,
        "modeled_repeated_cycle_break_even_request_count": extrapolated,
        "modeled_repeated_cycle_break_even_cost_crossing_request_count": extrapolated,
        "extrapolation_assumption": (
            "Exact ordered 12-request latency savings repeat unchanged; "
            "initialization occurs once per serving lifecycle."
        ),
        "provenance": {
            "observed": "derived",
            "modeled_repeated_cycle": "derived",
        },
    }


def _render_report(
    lifecycle: Sequence[Mapping[str, Any]],
    break_even: Mapping[str, Any],
    correctness: Mapping[str, Any],
    cost: Mapping[str, Any],
) -> str:
    rows = []
    for record in lifecycle:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record['mode']))}</td>"
            f"<td>{record['initialization_seconds']:.3f} s</td>"
            f"<td>{record['request_phase_seconds']:.3f} s</td>"
            f"<td>{record['host_lifecycle_seconds']:.3f} s</td>"
            f"<td>{record['peak_gpu_memory_mib']:,} MiB</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3 8B vLLM compilation break-even</title>
  <style>
    body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 960px; padding: 0 1rem; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d0d5dd; padding: .6rem; text-align: left; }}
    .finding {{ border-left: .4rem solid #b54708; background: #fffaeb; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Qwen3 8B vLLM compilation break-even</h1>
  <p>Scope: one CloudRift RTX 4090 VM, one pinned Qwen3 8B revision, and one
  exact 12-request sequence in eager and compiled modes. Modal and MLX results
  are excluded from this ranking.</p>
  <p class="finding"><strong>No observed break-even through request 12.</strong>
  Repeating the exact sequence yields a modeled crossing at request
  {break_even['modeled_repeated_cycle_break_even_request_count']}.</p>
  <table>
    <thead><tr><th>Mode</th><th>Initialization</th><th>Request phase</th><th>Host lifecycle</th><th>Peak GPU memory</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Correct responses: {correctness['successful_requests']} of
  {correctness['total_requests']}. List-rate lower bound through the scheduled
  shutdown boundary:
  ${cost['inferred_spend_usd_through_scheduled_shutdown_boundary']}.</p>
  <p>Boot-to-console-termination inferred spend:
  ${cost['final_inferred_spend_through_console_termination_usd']}.
  Provider-reported spend is unavailable. Provider-console termination was
  confirmed by the user; OS shutdown was scheduled but not observed.</p>
  <p>Collection limitation: the measured runner did not rehash the live model
  or cross-check both retained receipt hashes before each cell. No independent
  host receipt was retained for the fresh-container, cache-drop, timeout,
  bind-mount, network, or Docker image-inspection controls.</p>
</body>
</html>
"""


def _render_svg(break_even: Mapping[str, Any]) -> str:
    rows = break_even["cumulative_initialization_to_terminal"]
    points_eager = " ".join(
        f"{50 + row['request_count'] * 60},{330 - row['eager_seconds'] * 1.6:.1f}"
        for row in rows
    )
    points_compiled = " ".join(
        f"{50 + row['request_count'] * 60},{330 - row['compiled_seconds'] * 1.6:.1f}"
        for row in rows
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="820" height="380" viewBox="0 0 820 380" role="img" aria-labelledby="title desc">
<title id="title">Cumulative initialization and request time through request 12</title>
<desc id="desc">Compiled execution remains slower through every observed request.</desc>
<rect width="820" height="380" fill="white"/>
<line x1="50" y1="330" x2="780" y2="330" stroke="#667085"/>
<line x1="50" y1="40" x2="50" y2="330" stroke="#667085"/>
<polyline points="{points_eager}" fill="none" stroke="#175cd3" stroke-width="3"/>
<polyline points="{points_compiled}" fill="none" stroke="#b54708" stroke-width="3"/>
<text x="55" y="25" font-family="system-ui" font-size="16">Seconds from initialization start</text>
<text x="610" y="360" font-family="system-ui" font-size="14">Observed request count</text>
<text x="570" y="55" font-family="system-ui" font-size="14" fill="#175cd3">Eager</text>
<text x="650" y="55" font-family="system-ui" font-size="14" fill="#b54708">Compiled</text>
</svg>
"""


def _cost_ledger() -> dict[str, Any]:
    scheduled_seconds = int((_dt(SHUTDOWN_SCHEDULED_AT) - _dt(BOOT_AT)).total_seconds())
    scheduled_inferred = RATE_USD_PER_HOUR * Decimal(scheduled_seconds) / Decimal(3600)
    console_seconds = int((_dt(CONSOLE_TERMINATED_AT) - _dt(BOOT_AT)).total_seconds())
    console_inferred = RATE_USD_PER_HOUR * Decimal(console_seconds) / Decimal(3600)
    return {
        "schema_version": "1",
        "rate_source": "user-observed CloudRift console screenshot",
        "usd_per_hour": "0.390000",
        "vm_boot_at": BOOT_AT,
        "os_shutdown_scheduled_at": SHUTDOWN_SCHEDULED_AT,
        "accounted_seconds_through_scheduled_shutdown_boundary": scheduled_seconds,
        "inferred_spend_usd_through_scheduled_shutdown_boundary": (
            f"{scheduled_inferred:.6f}"
        ),
        "inferred_spend_scope": "scheduled-shutdown-boundary list-rate lower bound",
        "provider_reported_spend_usd": None,
        "console_terminated_at": CONSOLE_TERMINATED_AT,
        "accounted_seconds_boot_to_console_termination": console_seconds,
        "final_inferred_spend_through_console_termination_usd": (
            f"{console_inferred:.6f}"
        ),
        "final_inferred_spend_scope": (
            "boot-to-console-termination list-rate lower bound"
        ),
        "hard_cap_usd": "5.000000",
        "eight_hour_cutoff_cost_usd": "3.120000",
        "remaining_hard_cap_at_console_termination_usd": (
            f"{HARD_CAP_USD - console_inferred:.6f}"
        ),
        "credits_treated_as_zero_spend": False,
        "limitation": (
            "The provider did not report spend, and the provisioning-to-boot "
            "interval is unavailable."
        ),
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "claims": [
            {
                "claim": "Compilation did not break even through request 12.",
                "state": "supported",
                "provenance": "derived",
                "artifact": "break-even.json",
            },
            {
                "claim": "The exact repeated cycle crosses at request 113.",
                "state": "modeled",
                "provenance": "derived",
                "artifact": "break-even.json",
            },
            {
                "claim": "Twenty-two of 24 bounded responses passed evaluation.",
                "state": "supported",
                "provenance": "derived",
                "artifact": "correctness-report.json",
            },
            {
                "claim": "Eight of 12 paired outputs had identical token IDs.",
                "state": "supported",
                "provenance": "model_reported",
                "artifact": "correctness-report.json",
            },
            {
                "claim": "Peak GPU memory was measured for both cells.",
                "state": "supported",
                "provenance": "cuda",
                "artifact": "lifecycle-records.jsonl",
            },
            {
                "claim": "Provider-reported CloudRift spend is known.",
                "state": "unsupported",
                "provenance": "cloudrift_user_observed",
                "artifact": "cost-ledger.json",
            },
            {
                "claim": "CloudRift console termination is confirmed.",
                "state": "supported",
                "provenance": "cloudrift_user_observed",
                "artifact": "teardown-report.json",
            },
        ],
    }


def build_bundle(raw_dir: Path, output_dir: Path) -> None:
    """Build a deterministic sanitized bundle from the sealed private records."""
    staging = _read_json(raw_dir / "staging-receipt.json")
    prompts = _read_json(raw_dir / "prompt-token-ids.json")
    eager = _read_json(raw_dir / "eager.json")
    compiled = _read_json(raw_dir / "compiled.json")
    teardown_private = _read_json(raw_dir / "teardown-final.json")
    _verify_seal(staging, "receipt_sha256")
    _verify_seal(prompts, "prompt_ids_sha256")
    _verify_seal(eager, "cell_sha256")
    _verify_seal(compiled, "cell_sha256")
    cells = [eager, compiled]
    if [cell["mode"] for cell in cells] != ["eager", "compiled"]:
        raise CloudRiftEvidenceError("exactly eager then compiled cells are required")
    gpu_identities = {
        cell["hardware"].get("gpu_uuid_sha256")
        for cell in cells
        if isinstance(cell.get("hardware"), dict)
    }
    if len(gpu_identities) != 1 or None in gpu_identities:
        raise CloudRiftEvidenceError("cells do not share one private GPU identity")

    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "1",
        "provider": "CloudRift",
        "scope": "single fixed RTX 4090 VM",
        "execution_base_head": EXECUTION_BASE_HEAD,
        "collection_source_commit": COLLECTION_SOURCE_COMMIT,
        "collection_source_sha256": EXECUTED_RUNNER_SHA256,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "file_count": EXPECTED_MODEL_FILE_COUNT,
            "total_bytes": EXPECTED_MODEL_BYTES,
        },
        "cells": [
            {
                "cell_id": "rtx4090-eager",
                "mode": "eager",
                "enforce_eager": True,
                "compilation_mode": "NONE",
                "cuda_graph_mode": "NONE",
            },
            {
                "cell_id": "rtx4090-compiled",
                "mode": "compiled",
                "enforce_eager": False,
                "compilation_mode": "VLLM_COMPILE",
                "cuda_graph_mode": "FULL_AND_PIECEWISE",
            },
        ],
        "request_count_per_cell": 12,
        "sampling": SAMPLING,
        "isolation": {
            "sequence": ["rtx4090-eager", "rtx4090-compiled"],
            "max_live_cells": 1,
            "planned_hard_timeout_seconds_per_cell": 2700,
            "fresh_container_per_cell": None,
            "host_page_cache_dropped_between_cells": None,
            "model_warmup_requests": 0,
            "public_endpoint": False,
        },
        "representative_safe_command": [
            "timeout",
            "2700",
            "docker",
            "run",
            "--rm",
            "--gpus",
            "device=0",
            "--cpus",
            "4",
            "--memory",
            "32g",
            "--shm-size",
            "8g",
            "--pids-limit",
            "4096",
            "--network",
            "none",
            "--env",
            "CLOUDRIFT_HOST_INVOCATION_STARTED_AT=<utc-timestamp>",
            "--mount",
            "type=bind,src=<model>,dst=/model,readonly",
            "--mount",
            "type=bind,src=<state>,dst=/state,readonly",
            "--mount",
            "type=bind,src=<output>,dst=/output",
            "--entrypoint",
            "/usr/bin/python3",
            "llmtracefx-vllm:0.28.0-te415",
            "-m",
            "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_runner",
            "<eager-or-compiled>",
            "--model-path",
            "/model",
            "--state-path",
            "/state",
            "--output",
            "/output/<cell>.json",
        ],
        "provenance_domains": list(PROVENANCE),
        "limitations": [
            MEASURED_RUNNER_LIMITATION,
            (
                "Compilation and CUDA graph component durations were not retained "
                "and remain null."
            ),
            HOST_RECEIPT_LIMITATION,
        ],
    }
    pricing = {
        "schema_version": "1",
        "source": "user-observed CloudRift console screenshot",
        "observed_at": "2026-09-03",
        "public_api_quote": False,
        "region": "ap-east-tw-kn-1",
        "shape": {
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_memory_gb_displayed": 24,
            "ram_gb_displayed": 47,
            "disk_gb_displayed": 280,
            "shared_public_ip": True,
        },
        "usd_per_hour": "0.390000",
        "hard_cap_usd": "5.000000",
        "experiment_cutoff_hours": "8.000000",
        "cutoff_cost_usd": "3.120000",
        "contingency_usd": "1.880000",
    }
    inventory = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "file_count": staging["model_file_count"],
        "total_bytes": staging["model_bytes"],
        "files": staging["inventory"],
        "verification": "size and SHA-256 verified after download",
    }
    runtime = {
        "schema_version": "1",
        "base_image_reference": BASE_IMAGE_REFERENCE,
        "derived_image_id": DERIVED_IMAGE_ID,
        "overlay": ["typing_extensions==4.15.0"],
        "packages": RUNTIME_PINS,
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_memory_mib": 24564,
        "driver_version": "580.159.03",
        "same_private_gpu_identity_verified": True,
        "derived_image_id_verification": (
            "runner-reported constant; independent Docker image-inspection receipt "
            "not retained"
        ),
    }
    workload = {
        "schema_version": "1",
        "request_order": [
            descriptor.to_dict() for descriptor in workload_descriptors()
        ],
        "prompts": prompts["prompts"],
        "prompt_ids_sha256": prompts["prompt_ids_sha256"],
        "generation": SAMPLING,
        "token_identity": "exact token ID arrays reused by both cells",
    }
    lifecycle = _lifecycle_records(cells)
    requests, evaluations = _request_records(cells)
    correctness = _correctness_report(requests, evaluations)
    break_even = _break_even(cells)
    cost = _cost_ledger()
    teardown = {
        "schema_version": "1",
        "cleanup_ended_at": teardown_private["cleanup_ended_at"],
        "experiment_containers_remaining": teardown_private[
            "experiment_containers_remaining"
        ],
        "gpu_processes_remaining": teardown_private["gpu_processes_remaining"],
        "experiment_directory_removed": teardown_private[
            "experiment_directory_removed"
        ],
        "model_cache_removed": teardown_private["model_cache_removed"],
        "runtime_images_removed": teardown_private["runtime_images_removed"],
        "result_cache_removed": teardown_private["result_cache_removed"],
        "temporary_public_key_removed": teardown_private[
            "temporary_public_key_removed"
        ],
        "capture_sha256": "sha256:"
        + _sha256_bytes((raw_dir / "teardown-final.json").read_bytes()),
        "os_shutdown_scheduled": teardown_private["os_shutdown_scheduled"],
        "os_shutdown_observed": None,
        "os_shutdown_observation_unavailable_reason": (
            "The temporary key was removed before the scheduled shutdown, so "
            "subsequent SSH failure cannot distinguish shutdown from denied access."
        ),
        "provider_console_termination_confirmed": True,
        "provider_console_terminated_at": CONSOLE_TERMINATED_AT,
        "provider_console_confirmation_provenance": (
            "user confirmation relayed by the coordinator"
        ),
        "status": "complete",
    }
    claims = _claim_matrix()
    for name, value in (
        ("experiment-contract.json", contract),
        ("pricing-snapshot.json", pricing),
        ("model-inventory.json", inventory),
        ("runtime-image.json", runtime),
        ("workload-contract.json", workload),
        ("correctness-report.json", correctness),
        ("break-even.json", break_even),
        ("cost-ledger.json", cost),
        ("teardown-report.json", teardown),
        ("claim-matrix.json", claims),
    ):
        _write_json(output_dir / name, value)
    _write_jsonl(output_dir / "lifecycle-records.jsonl", lifecycle)
    _write_jsonl(output_dir / "request-records.jsonl", requests)
    (output_dir / "README.md").write_text(README, encoding="utf-8")
    (output_dir / "report.html").write_text(
        _render_report(lifecycle, break_even, correctness, cost), encoding="utf-8"
    )
    (output_dir / "break-even.svg").write_text(
        _render_svg(break_even), encoding="utf-8"
    )
    (output_dir / "evidence_bundle.py").write_text(WRAPPER, encoding="utf-8")
    checksums = [
        f"{_sha256_bytes((output_dir / name).read_bytes())}  {name}"
        for name in HASHED_FILES
    ]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    verify_bundle(output_dir)


def _scan_privacy(root: Path) -> None:
    for path in root.iterdir():
        if path.name == "SHA256SUMS":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, description in _PRIVATE_PATTERNS:
            if pattern.search(text):
                raise CloudRiftEvidenceError(f"{path.name} contains {description}")


def verify_bundle(root: Path) -> None:
    """Verify the closed, deterministic public evidence bundle."""
    actual = {
        path.name for path in root.iterdir() if not path.name.startswith("__pycache__")
    }
    if actual != set(BUNDLE_FILES):
        raise CloudRiftEvidenceError(
            f"bundle file set differs: {sorted(actual ^ set(BUNDLE_FILES))}"
        )
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_388_608:
            raise CloudRiftEvidenceError(f"{name} is not a bounded regular file")
    expected_lines = [
        f"{_sha256_bytes((root / name).read_bytes())}  {name}" for name in HASHED_FILES
    ]
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != (
        "\n".join(expected_lines) + "\n"
    ):
        raise CloudRiftEvidenceError("SHA256SUMS does not verify")
    if (
        _sha256_bytes((root / "lifecycle-records.jsonl").read_bytes())
        != EXPECTED_LIFECYCLE_RECORDS_SHA256
        or _sha256_bytes((root / "request-records.jsonl").read_bytes())
        != EXPECTED_REQUEST_RECORDS_SHA256
    ):
        raise CloudRiftEvidenceError("canonical raw record identity drifted")
    _scan_privacy(root)
    for name in JSON_FILES:
        value = _read_json(root / name)
        expected = (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
        if (root / name).read_text(encoding="utf-8") != expected:
            raise CloudRiftEvidenceError(f"{name} is not canonical JSON")
    contract = _read_json(root / "experiment-contract.json")
    expected_cells = [
        {
            "cell_id": "rtx4090-eager",
            "mode": "eager",
            "enforce_eager": True,
            "compilation_mode": "NONE",
            "cuda_graph_mode": "NONE",
        },
        {
            "cell_id": "rtx4090-compiled",
            "mode": "compiled",
            "enforce_eager": False,
            "compilation_mode": "VLLM_COMPILE",
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
        },
    ]
    if (
        contract["collection_source_commit"] != COLLECTION_SOURCE_COMMIT
        or contract["collection_source_sha256"] != EXECUTED_RUNNER_SHA256
        or contract["model"]
        != {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
            "file_count": EXPECTED_MODEL_FILE_COUNT,
            "total_bytes": EXPECTED_MODEL_BYTES,
        }
        or contract["cells"] != expected_cells
        or contract["request_count_per_cell"] != 12
        or contract["sampling"] != SAMPLING
        or contract["limitations"]
        != [
            MEASURED_RUNNER_LIMITATION,
            (
                "Compilation and CUDA graph component durations were not retained "
                "and remain null."
            ),
            HOST_RECEIPT_LIMITATION,
        ]
        or contract["isolation"]
        != {
            "sequence": ["rtx4090-eager", "rtx4090-compiled"],
            "max_live_cells": 1,
            "planned_hard_timeout_seconds_per_cell": 2700,
            "fresh_container_per_cell": None,
            "host_page_cache_dropped_between_cells": None,
            "model_warmup_requests": 0,
            "public_endpoint": False,
        }
    ):
        raise CloudRiftEvidenceError("experiment contract binding drifted")
    lifecycle = _load_jsonl(root / "lifecycle-records.jsonl")
    requests = _load_jsonl(root / "request-records.jsonl")
    if len(lifecycle) != 2 or [item["mode"] for item in lifecycle] != [
        "eager",
        "compiled",
    ]:
        raise CloudRiftEvidenceError("exactly two ordered cells are required")
    if len(requests) != 24:
        raise CloudRiftEvidenceError("exactly 24 request records are required")
    if _dt(lifecycle[1]["host_invocation_started_at"]) < _dt(lifecycle[0]["ended_at"]):
        raise CloudRiftEvidenceError("measured cells overlap")
    for expected_cell, record in zip(expected_cells, lifecycle, strict=True):
        expected_reasons = {
            "eager": (
                "not_applicable_eager_mode",
                "not_applicable_eager_mode",
            ),
            "compiled": (
                "vllm_compilation_time_not_exposed_or_nonpositive",
                "stable_component_timing_not_exposed",
            ),
        }
        compilation_reason, graph_reason = expected_reasons[record["mode"]]
        if (
            record["cell_id"] != expected_cell["cell_id"]
            or record["mode"] != expected_cell["mode"]
            or record["resolved_execution_config"]
            != {
                "enforce_eager": expected_cell["enforce_eager"],
                "compilation_mode": expected_cell["compilation_mode"],
                "cuda_graph_mode": expected_cell["cuda_graph_mode"],
            }
            or record["compilation_seconds"] is not None
            or record["cuda_graph_seconds"] is not None
            or record["compilation_seconds_unobservable_reason"] != compilation_reason
            or record["cuda_graph_seconds_unobservable_reason"] != graph_reason
            or record["initialization_seconds"]
            != {"eager": 64.600270, "compiled": 119.143334}[record["mode"]]
            or record["initialization_seconds"]
            != (
                _dt(record["initialization_ready_at"])
                - _dt(record["initialization_started_at"])
            ).total_seconds()
            or record["host_lifecycle_seconds"]
            != (
                _dt(record["ended_at"]) - _dt(record["host_invocation_started_at"])
            ).total_seconds()
            or record["request_phase_seconds"]
            != (
                _dt(record["ended_at"]) - _dt(record["initialization_ready_at"])
            ).total_seconds()
        ):
            raise CloudRiftEvidenceError("lifecycle measurement binding drifted")
    if any(not item["terminal"] for item in lifecycle + requests):
        raise CloudRiftEvidenceError("all records must be terminal")
    if any(item["ttft_seconds"] is None for item in requests):
        raise CloudRiftEvidenceError("request TTFT is incomplete")
    if any(item["ttft_seconds"] > item["latency_seconds"] for item in requests):
        raise CloudRiftEvidenceError("request timing is invalid")
    correctness = _read_json(root / "correctness-report.json")
    workload = _read_json(root / "workload-contract.json")
    prompts = workload["prompts"]
    descriptors = workload_descriptors()
    if (
        workload["request_order"]
        != [descriptor.to_dict() for descriptor in descriptors]
        or workload["generation"] != SAMPLING
        or workload["token_identity"] != "exact token ID arrays reused by both cells"
        or workload["prompt_ids_sha256"]
        != _sha256_json({"schema_version": "1", "prompts": prompts})
    ):
        raise CloudRiftEvidenceError("prompt token seal does not verify")
    if set(prompts) != set(EXPECTED_PROMPT_IDENTITIES):
        raise CloudRiftEvidenceError("prompt identity set drifted")
    for key, (expected_count, expected_hash) in EXPECTED_PROMPT_IDENTITIES.items():
        token_ids = prompts[key]
        if (
            not isinstance(token_ids, list)
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
            or len(token_ids) != expected_count
            or _sha256_json(token_ids) != expected_hash
        ):
            raise CloudRiftEvidenceError(f"prompt token identity drifted: {key}")
    expected_evaluations: list[dict[str, Any]] = []
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in ("eager", "compiled"):
        mode_requests = [item for item in requests if item["mode"] == mode]
        if len(mode_requests) != 12:
            raise CloudRiftEvidenceError("request modes are incomplete")
        by_mode[mode] = mode_requests
        for descriptor, request in zip(descriptors, mode_requests, strict=True):
            if any(
                request[key] != value for key, value in descriptor.to_dict().items()
            ):
                raise CloudRiftEvidenceError("request descriptor drifted")
            key = f"{descriptor.context_tier}/{descriptor.workload_id}"
            token_ids = prompts[key]
            if (
                request["cell_id"] != f"rtx4090-{mode}"
                or request["terminal"] is not True
                or request["warmup"] is not False
                or request["input_token_count"] != len(token_ids)
                or request["input_token_ids_sha256"] != _sha256_json(token_ids)
                or request["output_token_count"] != len(request["output_token_ids"])
                or any(
                    type(token_id) is not int or token_id < 0
                    for token_id in request["output_token_ids"]
                )
                or not math.isclose(
                    request["latency_seconds"],
                    (
                        _dt(request["ended_at"]) - _dt(request["started_at"])
                    ).total_seconds(),
                    rel_tol=0,
                    abs_tol=0.001,
                )
                or not 0 <= request["ttft_seconds"] <= request["latency_seconds"]
            ):
                raise CloudRiftEvidenceError("request token identity drifted")
            expected_rate = request["output_token_count"] / request["latency_seconds"]
            if not math.isclose(
                request["output_tokens_per_second"],
                expected_rate,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise CloudRiftEvidenceError("request output rate drifted")
            outcome = evaluate_workload(
                workload_by_id(descriptor.workload_id),
                request["decoded_output"],
            )
            if request["correctness"] != outcome.success:
                raise CloudRiftEvidenceError("request correctness drifted")
            expected_evaluations.append(
                {
                    "cell_id": request["cell_id"],
                    "ordinal": descriptor.ordinal,
                    "request_id": descriptor.request_id,
                    "success": outcome.success,
                    "quality_score": outcome.quality_score,
                    "quality_metric": outcome.quality_metric,
                    "notes": outcome.notes,
                    "evaluator": "evaluate_workload",
                }
            )
    expected_successes = sum(item["success"] for item in expected_evaluations)
    expected_by_mode = {
        mode: sum(item["correctness"] for item in mode_requests)
        for mode, mode_requests in by_mode.items()
    }
    expected_matches = sum(
        eager["output_token_ids"] == compiled["output_token_ids"]
        for eager, compiled in zip(by_mode["eager"], by_mode["compiled"], strict=True)
    )
    expected_mismatches = [
        eager["ordinal"]
        for eager, compiled in zip(by_mode["eager"], by_mode["compiled"], strict=True)
        if eager["output_token_ids"] != compiled["output_token_ids"]
    ]
    if (
        correctness["total_requests"] != 24
        or correctness["successful_requests"] != expected_successes
        or correctness["all_requests_correct"] != (expected_successes == 24)
        or correctness["successful_requests_by_mode"] != expected_by_mode
        or correctness["paired_output_token_identity_matches"] != expected_matches
        or correctness["paired_output_token_identity_mismatched_ordinals"]
        != expected_mismatches
        or correctness["evaluations"] != expected_evaluations
    ):
        raise CloudRiftEvidenceError("correctness report does not recompute")
    if (
        correctness["successful_requests"] != 22
        or correctness["successful_requests_by_mode"] != {"compiled": 12, "eager": 10}
        or correctness["paired_output_token_identity_matches"] != 8
        or correctness["paired_output_token_identity_mismatched_ordinals"]
        != [7, 8, 11, 12]
    ):
        raise CloudRiftEvidenceError("headline correctness result drifted")
    break_even = _read_json(root / "break-even.json")
    reconstructed_cells = [
        {
            **record,
            "requests": by_mode[record["mode"]],
        }
        for record in lifecycle
    ]
    if break_even != _break_even(reconstructed_cells):
        raise CloudRiftEvidenceError("break-even result does not verify")
    if (
        break_even["observed_break_even_request_count"] is not None
        or break_even["observed_lower_bound_request_count"] != 12
        or break_even["modeled_repeated_cycle_break_even_request_count"] != 113
        or break_even["modeled_repeated_cycle_break_even_cost_crossing_request_count"]
        != 113
        or break_even["initialization_penalty_seconds"] != 54.543064
    ):
        raise CloudRiftEvidenceError("headline break-even result drifted")
    cost = _read_json(root / "cost-ledger.json")
    if cost != _cost_ledger():
        raise CloudRiftEvidenceError("cost scopes are invalid")
    teardown = _read_json(root / "teardown-report.json")
    if (
        teardown["experiment_containers_remaining"] != 0
        or teardown["gpu_processes_remaining"] != 0
        or not teardown["experiment_directory_removed"]
        or not teardown["model_cache_removed"]
        or not teardown["runtime_images_removed"]
        or not teardown["result_cache_removed"]
        or not teardown["temporary_public_key_removed"]
        or not teardown["os_shutdown_scheduled"]
        or teardown["os_shutdown_observed"] is not None
        or not teardown["os_shutdown_observation_unavailable_reason"]
        or not teardown["provider_console_termination_confirmed"]
        or teardown["provider_console_terminated_at"] != CONSOLE_TERMINATED_AT
        or teardown["capture_sha256"] != EXPECTED_TEARDOWN_CAPTURE_SHA256
        or teardown["cleanup_ended_at"] != "2026-09-03T16:33:57Z"
        or teardown["provider_console_confirmation_provenance"]
        != "user confirmation relayed by the coordinator"
        or teardown["status"] != "complete"
    ):
        raise CloudRiftEvidenceError("teardown status is invalid")
    for item in lifecycle:
        if item["hardware"]["gpu_name"] != "NVIDIA GeForce RTX 4090":
            raise CloudRiftEvidenceError("hardware identity drifted")
        if "gpu_uuid_sha256" in item["hardware"]:
            raise CloudRiftEvidenceError("GPU UUID derivative is forbidden")
        if item["runtime"] != RUNTIME_PINS:
            raise CloudRiftEvidenceError("runtime identity drifted")
    runtime = _read_json(root / "runtime-image.json")
    if runtime != {
        "schema_version": "1",
        "base_image_reference": BASE_IMAGE_REFERENCE,
        "derived_image_id": DERIVED_IMAGE_ID,
        "overlay": ["typing_extensions==4.15.0"],
        "packages": RUNTIME_PINS,
        "gpu": "NVIDIA GeForce RTX 4090",
        "gpu_memory_mib": 24564,
        "driver_version": "580.159.03",
        "same_private_gpu_identity_verified": True,
        "derived_image_id_verification": (
            "runner-reported constant; independent Docker image-inspection receipt "
            "not retained"
        ),
    }:
        raise CloudRiftEvidenceError("runtime or same-GPU binding drifted")
    inventory = _read_json(root / "model-inventory.json")
    if (
        inventory["model_id"] != MODEL_ID
        or inventory["revision"] != MODEL_REVISION
        or inventory["license"] != "Apache-2.0"
        or inventory["file_count"] != EXPECTED_MODEL_FILE_COUNT
        or inventory["total_bytes"] != EXPECTED_MODEL_BYTES
        or inventory["files"] != _official_inventory_files()
        or inventory["verification"] != "size and SHA-256 verified after download"
    ):
        raise CloudRiftEvidenceError("model inventory binding drifted")
    pricing = _read_json(root / "pricing-snapshot.json")
    if pricing != {
        "schema_version": "1",
        "source": "user-observed CloudRift console screenshot",
        "observed_at": "2026-09-03",
        "public_api_quote": False,
        "region": "ap-east-tw-kn-1",
        "shape": {
            "gpu": "NVIDIA GeForce RTX 4090",
            "gpu_memory_gb_displayed": 24,
            "ram_gb_displayed": 47,
            "disk_gb_displayed": 280,
            "shared_public_ip": True,
        },
        "usd_per_hour": "0.390000",
        "hard_cap_usd": "5.000000",
        "experiment_cutoff_hours": "8.000000",
        "cutoff_cost_usd": "3.120000",
        "contingency_usd": "1.880000",
    }:
        raise CloudRiftEvidenceError("pricing contract drifted")
    if _read_json(root / "claim-matrix.json") != _claim_matrix():
        raise CloudRiftEvidenceError("claim matrix drifted")
    if (root / "README.md").read_text(encoding="utf-8") != README:
        raise CloudRiftEvidenceError("README drifted")
    if (root / "report.html").read_text(encoding="utf-8") != _render_report(
        lifecycle, break_even, correctness, cost
    ):
        raise CloudRiftEvidenceError("HTML report drifted")
    if (root / "break-even.svg").read_text(encoding="utf-8") != _render_svg(break_even):
        raise CloudRiftEvidenceError("SVG report drifted")
    for document in (
        lifecycle,
        requests,
        correctness,
        break_even,
        cost,
        teardown,
    ):
        encoded = json.dumps(document, allow_nan=False)
        if "NaN" in encoded or "Infinity" in encoded:
            raise CloudRiftEvidenceError("non-finite metric found")
    if not math.isclose(
        lifecycle[1]["initialization_seconds"] - lifecycle[0]["initialization_seconds"],
        break_even["initialization_penalty_seconds"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise CloudRiftEvidenceError("initialization penalty drifted")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_bundle(args.raw_dir, args.output_dir)
    else:
        verify_bundle(args.bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
