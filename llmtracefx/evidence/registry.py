"""Closed registry of public evidence sources and their claim contracts."""

from __future__ import annotations

from typing import Any

CATALOG_SCHEMA_VERSION = "1"
VERIFIER_VERSION = "1"

CLAIM_DIMENSIONS = (
    "timing",
    "quality",
    "cost",
    "memory",
    "process_attribution",
    "model_fit",
    "deployment_readiness",
)


def _claim(state: str, provenance: str) -> dict[str, str]:
    return {"state": state, "provenance": provenance}


def _claims(**values: tuple[str, str]) -> dict[str, dict[str, str]]:
    return {dimension: _claim(*values[dimension]) for dimension in CLAIM_DIMENSIONS}


SOURCES: tuple[dict[str, Any], ...] = (
    {
        "evidence_id": "metal-attribution-m5-pro-20260831",
        "kind": "metal_attribution",
        "status": "verified",
        "outcome": "completed",
        "public_path": "examples/metal_evidence/public",
        "bundle_schema_version": "1",
        "adapter": "metal_public_v1",
        "artifact_files": (
            "SHA256SUMS",
            "capture-summary.csv",
            "capture-summary.json",
            "dispatch-attribution.svg",
            "experiment-manifest.json",
            "unrelated-interval-share.svg",
        ),
        "captured_at": "2026-08-31T13:18:34+00:00",
        "source_commit": None,
        "model": {
            "id": None,
            "revision": None,
            "quantization": None,
        },
        "runtime": {
            "name": "Apple Instruments Metal System Trace",
            "version": "26.6",
            "provider": "local",
        },
        "hardware": {
            "system": "Apple M5 Pro",
            "architecture": "arm64",
        },
        "workload": {
            "identity": "metal-evidence-workload",
            "context": None,
            "request": "five controlled dispatch-count captures",
        },
        "measurements": (
            {
                "scope": "Metal interval count",
                "provenance": "measured_native and target-process attributed",
            },
            {
                "scope": "unrelated interval share",
                "provenance": "derived from measured interval counts",
            },
        ),
        "claims": _claims(
            timing=("unsupported", "interval counts are not utilization or time"),
            quality=("not_applicable", "no model output was evaluated"),
            cost=("not_applicable", "local trace capture has no spend claim"),
            memory=("unsupported", "GPU memory footprint was not measured"),
            process_attribution=(
                "supported",
                "target-process interval counts and trace-wide counts",
            ),
            model_fit=("not_applicable", "no model was loaded"),
            deployment_readiness=("not_applicable", "not a deployment run"),
        ),
        "supported_claims": (
            "Trace-wide Metal interval totals include unrelated processes.",
            "Target-process attribution matched each controlled dispatch count.",
        ),
        "unsupported_claims": (
            "GPU utilization or busy percentage",
            "kernel time, memory bandwidth, occupancy, power, or energy",
            "GPU memory footprint",
        ),
        "budget": {
            "scope": "not_applicable",
            "authorized_usd": None,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": None,
            "limitation": "No cost measurement was in scope.",
        },
        "dependencies": (),
        "limitations": (
            "Interval count is not utilization or elapsed GPU time.",
            "Only sanitized aggregate trace evidence is public.",
        ),
    },
    {
        "evidence_id": "qwen38-27b-m5-pro-lab-oom-20260831",
        "kind": "model_lab",
        "status": "verified",
        "outcome": "oom",
        "public_path": "examples/optimizer/m5-pro-qwen3.8-27b",
        "bundle_schema_version": "1",
        "adapter": "legacy_pinned_v1",
        "artifact_files": ("README.md", "evidence-summary.json", "report.html"),
        "captured_at": "2026-08-31T14:57:58.660151Z",
        "source_commit": None,
        "model": {
            "id": "mlx-community/Qwen3.8-27B-4bit",
            "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
            "quantization": "MLX affine 4-bit, group size 64",
        },
        "runtime": {
            "name": "mlx-vlm",
            "version": "0.6.8",
            "provider": "local",
        },
        "hardware": {
            "system": "Apple M5 Pro, 24 GiB unified memory",
            "architecture": "arm64",
        },
        "workload": {
            "identity": "structured-json-profile-extraction warmup",
            "context": "2K tier; 1,657 actual input tokens",
            "request": "96 maximum output tokens",
        },
        "measurements": (
            {
                "scope": "host wall-clock total",
                "provenance": "measured_wall_clock",
            },
            {
                "scope": "system memory pressure and swap",
                "provenance": "macOS memory_pressure and sysctl",
            },
        ),
        "claims": _claims(
            timing=("supported", "host wall-clock total before OOM"),
            quality=("unsupported", "no evaluator result was produced"),
            cost=("not_applicable", "local execution has no spend claim"),
            memory=(
                "supported",
                "system memory pressure and swap; MLX peak is unavailable",
            ),
            process_attribution=("not_applicable", "single local model run"),
            model_fit=(
                "supported",
                "OOM during the exact recorded warmup and host state",
            ),
            deployment_readiness=(
                "unsupported",
                "a failed exploratory warmup is not a deployment assessment",
            ),
        ),
        "supported_claims": (
            "The exact 27B checkpoint OOMed during the recorded 2K warmup.",
            "No 8K or 16K tier was attempted after the failure.",
        ),
        "unsupported_claims": (
            "universal 24 GiB capacity boundary",
            "quality, throughput, GPU utilization, power, or kernel timing",
            "causal allocation attribution",
        ),
        "budget": {
            "scope": "not_applicable",
            "authorized_usd": None,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": None,
            "limitation": "Local execution did not record cost.",
        },
        "dependencies": (),
        "limitations": (
            "Exploratory host state; no clean-boot assertion.",
            "Peak memory is unavailable for the failed attempt.",
        ),
    },
    {
        "evidence_id": "qwen38-27b-m5-pro-fit-frontier-20260901",
        "kind": "fit_frontier",
        "status": "verified",
        "outcome": "oom",
        "public_path": (
            "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/exploratory"
        ),
        "bundle_schema_version": "1",
        "adapter": "legacy_pinned_v1",
        "artifact_files": ("fit-frontier-report.html", "fit-frontier-summary.json"),
        "captured_at": "2026-09-01T05:30:24.941062Z",
        "source_commit": None,
        "model": {
            "id": "mlx-community/Qwen3.8-27B-4bit",
            "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
            "quantization": "MLX affine 4-bit, group size 64",
        },
        "runtime": {
            "name": "mlx-vlm",
            "version": "0.6.8",
            "provider": "local",
        },
        "hardware": {
            "system": "Apple M5 Pro, 24 GiB unified memory",
            "architecture": "arm64",
        },
        "workload": {
            "identity": "m5-pro-qwen3.8-27b-fit-frontier-v1",
            "context": "t256 attempted; larger tiers skipped",
            "request": "bounded exploratory fit-frontier row",
        },
        "measurements": (
            {
                "scope": "host wall-clock total",
                "provenance": "measured_wall_clock",
            },
            {
                "scope": "approximate system headroom and swap",
                "provenance": "macOS memory_pressure and sysctl",
            },
        ),
        "claims": _claims(
            timing=("supported", "host wall-clock total before OOM"),
            quality=("unsupported", "no quality result was produced"),
            cost=("not_applicable", "local execution has no spend claim"),
            memory=(
                "supported",
                "approximate system headroom and swap; not GPU memory",
            ),
            process_attribution=("not_applicable", "single local model run"),
            model_fit=("supported", "bounded t256 OOM in the recorded host state"),
            deployment_readiness=(
                "unsupported",
                "exploratory frontier is not a deployment assessment",
            ),
        ),
        "supported_claims": (
            "The exact checkpoint OOMed at t256 in the recorded machine state.",
            "Larger frontier tiers were skipped after the stop gate.",
        ),
        "unsupported_claims": (
            "universal memory-capacity boundary",
            "peak system or GPU memory",
            "quality, utilization, bandwidth, power, energy, or kernel time",
        ),
        "budget": {
            "scope": "not_applicable",
            "authorized_usd": None,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": None,
            "limitation": "Local execution did not record cost.",
        },
        "dependencies": (
            {
                "evidence_id": "qwen38-27b-m5-pro-lab-oom-20260831",
                "relation": "same_model_as",
            },
        ),
        "limitations": (
            "Exploratory run without a clean-boot assertion.",
            "Available memory is approximate system headroom, not GPU memory.",
        ),
    },
    {
        "evidence_id": "qwen38-27b-m5-pro-clean-boot-autopsy-20260901",
        "kind": "oom_autopsy",
        "status": "verified",
        "outcome": "oom",
        "public_path": (
            "examples/optimizer/m5-pro-qwen3.8-27b-oom-autopsy/publication"
        ),
        "bundle_schema_version": "1",
        "adapter": "oom_autopsy_v1",
        "artifact_files": (
            "SHA256SUMS",
            "autopsy-plan.json",
            "evidence-manifest.json",
            "mlx-memory-by-stage.svg",
            "oom-autopsy-checkpoints.csv",
            "oom-autopsy-report.html",
            "oom-autopsy-summary.json",
        ),
        "captured_at": "2026-09-01T17:45:36.921331Z",
        "source_commit": "2519bc8da309656d2e2ce2a7063f19b0dfb4c9ed",
        "model": {
            "id": "mlx-community/Qwen3.8-27B-4bit",
            "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
            "quantization": "MLX affine 4-bit, group size 64",
        },
        "runtime": {
            "name": "mlx-vlm",
            "version": "0.6.8",
            "provider": "local",
        },
        "hardware": {
            "system": "Apple M5 Pro, 24 GiB unified memory",
            "architecture": "arm64",
        },
        "workload": {
            "identity": "m5-pro-qwen3.8-27b-oom-autopsy-v1",
            "context": "t256; 256 actual prompt tokens",
            "request": "clean-boot publication autopsy",
        },
        "measurements": (
            {
                "scope": "MLX active/cache/peak allocator counters",
                "provenance": "MLX allocator; non-additive with RSS and swap",
            },
            {
                "scope": "process RSS",
                "provenance": "host process current/max RSS",
            },
            {
                "scope": "system swap and approximate headroom",
                "provenance": "sysctl and macOS memory_pressure",
            },
        ),
        "claims": _claims(
            timing=("supported", "stage-boundary host wall-clock observations"),
            quality=("unsupported", "no first token or evaluator result exists"),
            cost=("not_applicable", "local execution has no spend claim"),
            memory=(
                "supported",
                "separate MLX allocator, RSS, swap, and headroom scopes",
            ),
            process_attribution=("not_applicable", "single autopsy child process"),
            model_fit=(
                "supported",
                "clean-boot OOM for the exact checkpoint and t256 workload",
            ),
            deployment_readiness=(
                "unsupported",
                "OOM autopsy does not assess deployment readiness",
            ),
        ),
        "supported_claims": (
            "The exact checkpoint OOMed before first token at t256 after clean boot.",
            "MLX allocator, RSS, swap, and headroom scopes remain non-additive.",
        ),
        "unsupported_claims": (
            "universal memory-capacity or 24 GiB boundary",
            "causal allocation attribution",
            "quality, throughput, utilization, power, energy, or kernel time",
        ),
        "budget": {
            "scope": "not_applicable",
            "authorized_usd": None,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": None,
            "limitation": "Local execution did not record cost.",
        },
        "dependencies": (
            {
                "evidence_id": "qwen38-27b-m5-pro-lab-oom-20260831",
                "relation": "same_model_as",
            },
            {
                "evidence_id": "qwen38-27b-m5-pro-fit-frontier-20260901",
                "relation": "same_model_as",
            },
        ),
        "limitations": (
            "Bounded to one exact checkpoint, runtime, machine state, and workload.",
            "Discrete checkpoints do not provide causal allocation attribution.",
        ),
    },
    {
        "evidence_id": "qwen3-8b-m5-pro-control-20260902",
        "kind": "positive_control",
        "status": "verified",
        "outcome": "completed",
        "public_path": "examples/optimizer/qwen3-8b-m5-control",
        "bundle_schema_version": "1",
        "adapter": "sha256_allowlist_v1",
        "artifact_files": (
            "README.md",
            "SHA256SUMS",
            "control-manifest.json",
            "conversion-preflight-refusal-example.json",
            "conversion-summary.json",
            "evidence-manifest.json",
            "evidence-summary.json",
            "report.html",
        ),
        "captured_at": "2026-09-02T08:08:36.185127Z",
        "source_commit": "6b82cf276ee1e1cef03a0c92847082f872c8feba",
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "quantization": "MLX affine 4-bit, group size 64; self-converted",
        },
        "runtime": {
            "name": "mlx-lm",
            "version": "0.31.3",
            "provider": "local",
        },
        "hardware": {
            "system": "Apple M5 Pro, 24 GiB unified memory",
            "architecture": "arm64",
        },
        "workload": {
            "identity": "pinned LLMTraceFX workload catalog",
            "context": "requested 2K, 8K, and 16K tiers",
            "request": "four measured runs per tier; thinking disabled",
        },
        "measurements": (
            {
                "scope": "host wall-clock timing",
                "provenance": "measured_wall_clock",
            },
            {
                "scope": "MLX active/cache/peak allocator counters",
                "provenance": "measured_native; not RSS or swap",
            },
            {
                "scope": "decode rate and correct cases per minute",
                "provenance": "derived from measured counts and wall time",
            },
        ),
        "claims": _claims(
            timing=("supported", "host wall-clock prefill/decode/total"),
            quality=("supported", "pinned evaluator pass rate and score"),
            cost=("not_applicable", "local execution has no spend claim"),
            memory=("supported", "MLX allocator counters; not RSS or swap"),
            process_attribution=("not_applicable", "single isolated row process"),
            model_fit=(
                "supported",
                "this self-converted 8B checkpoint completed through requested 16K",
            ),
            deployment_readiness=(
                "unsupported",
                "exploratory benchmark is not a deployment assessment",
            ),
        ),
        "supported_claims": (
            "The exact self-converted Qwen3-8B checkpoint completed all recorded tiers.",
            "All twelve evaluated rows passed their pinned evaluator.",
        ),
        "unsupported_claims": (
            "comparability with the Qwen3.8-27B checkpoint",
            "clean-boot publication performance",
            "GPU utilization, bandwidth, power, energy, or kernel timing",
        ),
        "budget": {
            "scope": "not_applicable",
            "authorized_usd": None,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": None,
            "limitation": "Local execution did not record cost.",
        },
        "dependencies": (),
        "limitations": (
            "Exploratory run without a clean-boot assertion.",
            "Different model and system identity from every 27B result.",
        ),
    },
    {
        "evidence_id": "openrouter-glm-2k-comparison-20260902",
        "kind": "hosted_comparison",
        "status": "verified",
        "outcome": "comparison",
        "public_path": "examples/optimizer/openrouter-glm-2k",
        "bundle_schema_version": "1",
        "adapter": "openrouter_glm_v1",
        "artifact_files": (
            "README.md",
            "SHA256SUMS",
            "budget-ledger.json",
            "budget-plan.json",
            "compare-policy.json",
            "comparison.html",
            "comparison.json",
            "evidence_bundle.py",
            "experiment-manifest.json",
            "generation-metadata.json",
            "measurements.json",
            "pricing-manifest.json",
            "pricing-snapshot.json",
        ),
        "captured_at": "2026-09-02T11:44:23.771009Z",
        "source_commit": "a6077adaf7135e2a2e360aeae4a73b6b411b3493",
        "model": {
            "id": "z-ai/glm-5.3 and z-ai/glm-5.3-flash",
            "revision": ("z-ai/glm-5.3-20260816 and z-ai/glm-5.3-flash-20260826"),
            "quantization": "fp8",
        },
        "runtime": {
            "name": "OpenRouter hosted API",
            "version": "captured provider route",
            "provider": "OpenRouter / Z.AI",
        },
        "hardware": {
            "system": "provider managed and undisclosed",
            "architecture": None,
        },
        "workload": {
            "identity": (
                "structured-json-profile-extraction@1 and "
                "prose-reasoning-two-train-gap@1"
            ),
            "context": "2K tier",
            "request": "eight requests; low reasoning; no retries",
        },
        "measurements": (
            {
                "scope": "client timing",
                "provenance": "network, gateway, queueing, and execution combined",
            },
            {
                "scope": "provider request usage cost",
                "provenance": "final SSE usage blocks",
            },
            {
                "scope": "account usage delta",
                "provenance": "separate lag-prone corroborating observation",
            },
        ),
        "claims": _claims(
            timing=("supported", "client-observed hosted request timing"),
            quality=("supported", "pinned evaluators for eight completed requests"),
            cost=(
                "supported",
                "provider usage, plan, and account delta remain separate scopes",
            ),
            memory=("unsupported", "provider memory was not exposed"),
            process_attribution=("not_applicable", "hosted provider internals"),
            model_fit=("unsupported", "hosted completion does not prove hardware fit"),
            deployment_readiness=(
                "unsupported",
                "comparison does not establish production readiness",
            ),
        ),
        "supported_claims": (
            "Eight pinned hosted requests completed and passed their evaluators.",
            "Provider-reported request usage totaled USD 0.00615262.",
        ),
        "unsupported_claims": (
            "universal winner or cross-system local ranking",
            "server-only timing, provider memory, or hardware fit",
            "production readiness",
        ),
        "budget": {
            "scope": "provider_reported_request_usage",
            "authorized_usd": 5.0,
            "planned_usd": 0.0745326,
            "reported_usd": 0.00615262,
            "inferred_usd": None,
            "limitation": (
                "The local ledger is user-writable; account delta can lag and "
                "include unrelated activity."
            ),
        },
        "dependencies": (
            {
                "evidence_id": "qwen3-8b-m5-pro-control-20260902",
                "relation": "compares",
            },
        ),
        "limitations": (
            "Client timing includes network, gateway, queueing, and execution.",
            "Two repetitions per workload/model are directional evidence.",
        ),
    },
    {
        "evidence_id": "modal-glm53flash-preflight-20260902",
        "kind": "provider_preflight",
        "status": "verified",
        "outcome": "refused",
        "public_path": "examples/optimizer/modal-glm53flash-preflight",
        "bundle_schema_version": "1",
        "adapter": "modal_preflight_v1",
        "artifact_files": (
            "README.md",
            "SHA256SUMS",
            "budget-plan.json",
            "evidence_bundle.py",
            "experiment-manifest.json",
            "inventory-summary.json",
            "pricing.json",
            "report.html",
        ),
        "captured_at": "2026-09-02T21:57:45+05:30",
        "source_commit": "debd8fa3f2d4bbed3ccdaad40fd1be80e264fe87",
        "model": {
            "id": "zai-org/GLM-5.3-Flash",
            "revision": "03eb5366286afd40d2221b1d9c63a6dd1ba4832e",
            "quantization": "native FP8 e4m3 with dynamic activation scaling",
        },
        "runtime": {
            "name": "vLLM",
            "version": "unverified source revision candidate",
            "provider": "Modal",
        },
        "hardware": {
            "system": "planned 4x H200; not provisioned",
            "architecture": "linux/amd64 image manifest",
        },
        "workload": {
            "identity": "GLM-5.3-Flash deployment preflight",
            "context": "planned 131,072 tokens",
            "request": "no authenticated or paid execution",
        },
        "measurements": (
            {
                "scope": "modeled lifecycle cost",
                "provenance": "pricing plan; not provider usage",
            },
            {
                "scope": "experiment-attributable spend",
                "provenance": "inferred zero because no resource was created",
            },
        ),
        "claims": _claims(
            timing=("unsupported", "no deployment or request ran"),
            quality=("unsupported", "no model output exists"),
            cost=(
                "supported",
                "modeled cost and inferred zero spend are distinct scopes",
            ),
            memory=("unsupported", "no runtime memory was measured"),
            process_attribution=("not_applicable", "no process ran"),
            model_fit=("unsupported", "planned hardware fit was not proven"),
            deployment_readiness=(
                "supported",
                "preflight refusal and exact stop gates are recorded",
            ),
        ),
        "supported_claims": (
            "Paid execution was refused by three explicit stop gates.",
            "No Modal resource was created and attributable spend is inferred zero.",
        ),
        "unsupported_claims": (
            "hardware fit, startup, readiness, benchmark, or ranking",
            "runtime memory, utilization, bandwidth, power, or energy",
        ),
        "budget": {
            "scope": "modeled_plan_and_inferred_zero_spend",
            "authorized_usd": 10.0,
            "planned_usd": None,
            "reported_usd": None,
            "inferred_usd": 0.0,
            "limitation": "No provider-reported usage was available.",
        },
        "dependencies": (),
        "limitations": (
            "No model load, deployment, readiness probe, or smoke request ran.",
            "The framework source revision remained unverified.",
        ),
    },
    {
        "evidence_id": "cloudrift-glm53flash-preflight-20260902",
        "kind": "provider_preflight",
        "status": "verified",
        "outcome": "refused",
        "public_path": "examples/optimizer/cloudrift-glm53flash-preflight",
        "bundle_schema_version": "1",
        "adapter": "cloudrift_preflight_v1",
        "artifact_files": (
            "README.md",
            "SHA256SUMS",
            "budget-plan.json",
            "evidence_bundle.py",
            "experiment-manifest.json",
            "model-inventory-reference.json",
            "provider-snapshot.json",
            "report.html",
        ),
        "captured_at": "2026-09-02T22:38:00+05:30",
        "source_commit": "0dbdcf5e745f123e13d38d09296f629c24abd748",
        "model": {
            "id": "zai-org/GLM-5.3-Flash",
            "revision": "03eb5366286afd40d2221b1d9c63a6dd1ba4832e",
            "quantization": "native FP8 e4m3 with dynamic activation scaling",
        },
        "runtime": {
            "name": "unselected; official recipe unverified",
            "version": None,
            "provider": "CloudRift",
        },
        "hardware": {
            "system": "observed 8x V100; required 8x H200 unavailable",
            "architecture": None,
        },
        "workload": {
            "identity": "GLM-5.3-Flash provider preflight",
            "context": "no model request",
            "request": "no authenticated, provisioning, or paid execution",
        },
        "measurements": (
            {
                "scope": "observed console inventory",
                "provenance": "user-observed aggregate; identifiers removed",
            },
            {
                "scope": "experiment-attributable spend",
                "provenance": "inferred zero because no resource was created",
            },
        ),
        "claims": _claims(
            timing=("unsupported", "no deployment or request ran"),
            quality=("unsupported", "no model output exists"),
            cost=(
                "supported",
                "planned caps and inferred zero spend; no provider usage",
            ),
            memory=(
                "supported",
                "aggregate listed V100 memory and exact model inventory comparison",
            ),
            process_attribution=("not_applicable", "no process ran"),
            model_fit=(
                "supported",
                "available V100 memory was below the exact model inventory",
            ),
            deployment_readiness=(
                "supported",
                "preflight refusal and exact stop gates are recorded",
            ),
        ),
        "supported_claims": (
            "Paid execution was refused by seven explicit stop gates.",
            "The observed V100 listing was below the exact model inventory.",
            "No CloudRift resource was created and spend is inferred zero.",
        ),
        "unsupported_claims": (
            "H200 fit, startup, readiness, benchmark, throughput, or SLA",
            "runtime utilization, bandwidth, power, or energy",
        ),
        "budget": {
            "scope": "planned_caps_and_inferred_zero_spend",
            "authorized_usd": 80.0,
            "planned_usd": 60.0,
            "reported_usd": None,
            "inferred_usd": 0.0,
            "limitation": "No provider billing result was available without access.",
        },
        "dependencies": (
            {
                "evidence_id": "modal-glm53flash-preflight-20260902",
                "relation": "same_model_as",
            },
        ),
        "limitations": (
            "No provisioning, model load, readiness probe, or smoke request ran.",
            "The required H200 inventory and official runtime recipe were unverified.",
        ),
    },
    {
        "evidence_id": "qwen3-8b-cloudrift-vllm-compile-20260903",
        "kind": "compile_break_even",
        "status": "verified",
        "outcome": "comparison",
        "public_path": "examples/optimizer/qwen3-8b-vllm-compile-break-even",
        "bundle_schema_version": "1",
        "adapter": "cloudrift_compile_v1",
        "artifact_files": (
            "README.md",
            "SHA256SUMS",
            "break-even.json",
            "break-even.svg",
            "claim-matrix.json",
            "correctness-report.json",
            "cost-ledger.json",
            "evidence_bundle.py",
            "experiment-contract.json",
            "lifecycle-records.jsonl",
            "model-inventory.json",
            "pricing-snapshot.json",
            "report.html",
            "request-records.jsonl",
            "runtime-image.json",
            "teardown-report.json",
            "workload-contract.json",
        ),
        "captured_at": "2026-09-03T16:30:38.954381+00:00",
        "source_commit": "9c0879351cc3e4f294b5c827d74dfc00182d53bb",
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "quantization": "bfloat16",
        },
        "runtime": {
            "name": "vLLM",
            "version": "0.28.0",
            "provider": "CloudRift",
        },
        "hardware": {
            "system": "NVIDIA GeForce RTX 4090, 24,564 MiB",
            "architecture": "CUDA 13.0",
        },
        "workload": {
            "identity": "qwen3-8b-vllm-compile-break-even-v1",
            "context": "2K, 8K, and 16K tiers; exact pinned token arrays",
            "request": "12 requests per cell; 96 maximum output tokens",
        },
        "measurements": (
            {
                "scope": "initialization, TTFT, and complete response latency",
                "provenance": "client-observed and vLLM metrics",
            },
            {
                "scope": "peak GPU memory",
                "provenance": "sampled nvidia-smi used memory",
            },
            {
                "scope": "list-rate lifecycle cost through OS shutdown",
                "provenance": "derived lower bound from user-observed console rate",
            },
        ),
        "claims": _claims(
            timing=(
                "supported",
                "measured initialization and 24 bounded request records",
            ),
            quality=(
                "supported",
                "22 of 24 deterministic workload evaluator results passed",
            ),
            cost=(
                "supported",
                "boot-to-console list-rate inference; provider spend is unavailable",
            ),
            memory=("supported", "sampled peak device memory for both cells"),
            process_attribution=("not_applicable", "one isolated cell at a time"),
            model_fit=("supported", "both exact-revision cells completed"),
            deployment_readiness=(
                "unsupported",
                "bounded benchmark is not a production deployment assessment",
            ),
        ),
        "supported_claims": (
            "Compilation did not repay initialization through request 12.",
            "Exact-sequence repeated-cycle extrapolation crosses at request 113.",
            "Twenty-two of 24 bounded responses passed deterministic evaluation.",
            "Compiled output passed 12 of 12; eager output passed 10 of 12.",
            "Eight of 12 paired outputs had identical token IDs.",
            "Both isolated cells completed on the fixed RTX 4090.",
            "The user confirmed provider console termination.",
        ),
        "unsupported_claims": (
            "general break-even outside the exact workload and runtime",
            "provider-reported spend or provisioning-to-boot cost",
            "production readiness, SLA, power, energy, or bandwidth",
            "direct component timing for compilation or CUDA graph capture",
        ),
        "budget": {
            "scope": "list_rate_lower_bound_through_os_shutdown",
            "authorized_usd": 5.0,
            "planned_usd": 3.12,
            "reported_usd": None,
            "inferred_usd": 0.484358,
            "limitation": (
                "Provider spend and provisioning-to-boot duration are unavailable."
            ),
        },
        "dependencies": (
            {
                "evidence_id": "qwen3-8b-m5-pro-control-20260902",
                "relation": "uses_workload_contract",
            },
        ),
        "limitations": (
            "Break-even at request 113 is modeled from repeated exact-cycle savings.",
            "Compilation and CUDA graph component durations were not retained.",
            "Provider-reported spend and provisioning-to-boot cost are unavailable.",
            "MLX results are an incompatible scope and are not ranked here.",
        ),
    },
)

ADAPTERS = {
    "metal_public_v1": {
        "name": "metal-evidence.verify_public_bundle",
        "version": VERIFIER_VERSION,
    },
    "legacy_pinned_v1": {
        "name": "historical-immutable-artifact-set",
        "version": VERIFIER_VERSION,
    },
    "oom_autopsy_v1": {
        "name": "oom-autopsy.verify_bundle",
        "version": VERIFIER_VERSION,
    },
    "sha256_allowlist_v1": {
        "name": "sha256-allowlist",
        "version": VERIFIER_VERSION,
    },
    "openrouter_glm_v1": {
        "name": "openrouter-glm.verify",
        "version": VERIFIER_VERSION,
    },
    "modal_preflight_v1": {
        "name": "modal-preflight.verify",
        "version": VERIFIER_VERSION,
    },
    "cloudrift_preflight_v1": {
        "name": "cloudrift-preflight.verify",
        "version": VERIFIER_VERSION,
    },
    "cloudrift_compile_v1": {
        "name": "cloudrift-compile.verify",
        "version": VERIFIER_VERSION,
    },
}

LEGACY_PINNED_SHA256 = {
    "examples/optimizer/m5-pro-qwen3.8-27b/README.md": (
        "abb2fa916c61e76b3a681c3d691314c3e02ad6613b4abdc64b3af01f12592beb"
    ),
    "examples/optimizer/m5-pro-qwen3.8-27b/evidence-summary.json": (
        "e09ee3e130e91824b0d7e9c29336e7742f862954b53237edd943a6ffa67da5d4"
    ),
    "examples/optimizer/m5-pro-qwen3.8-27b/report.html": (
        "42523fca6a40a739c3d36cf105a174b8ee180b2c990ce0fa98e58d4d609607f3"
    ),
    (
        "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/exploratory/"
        "fit-frontier-report.html"
    ): "5830a7ca5dc2d3c2e49c9f2477ad44b9adeef694c2e8a5149acf5d52c4b7fc32",
    (
        "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/exploratory/"
        "fit-frontier-summary.json"
    ): "7e8116b3fd4a4d639ae0693fc74eaf8f08bc93bb02bb33fdd6556c650ffc4c8a",
}

__all__ = [
    "ADAPTERS",
    "CATALOG_SCHEMA_VERSION",
    "CLAIM_DIMENSIONS",
    "LEGACY_PINNED_SHA256",
    "SOURCES",
    "VERIFIER_VERSION",
]
