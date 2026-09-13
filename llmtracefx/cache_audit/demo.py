"""Output-first deterministic public demo for the cache truth auditor."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from llmtracefx.evidence.core import PRIVACY_PATTERNS, canonical_json

from .adapters.reference import ReferenceCacheAdapter
from .bundle import BUNDLE_FILES, verify_bundle
from .runner import run_audit, source_commit
from .schema import CacheConfig, PublicationMode, RequestEvidence
from .workloads import public_demo_requests

DEMO_COMMAND = "make kv-cache-demo"
DEMO_MAX_ENTRIES = 7
DEMO_SEED = 20260913
CLAIM_BOUNDARIES = (
    "Proves the deterministic auditor, independent oracle, synthetic engine "
    "attestation, output evaluator, evidence schemas, privacy checks, hashes, "
    "and verifier behavior only.",
    "Does not prove MLX or vLLM speedup, production cache correctness, provider "
    "identity, GPU performance, latency improvement, or runtime memory savings.",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _row(record: RequestEvidence) -> dict[str, Any]:
    return {
        "case": record.spec.request_id,
        "input_length": record.spec.input_token_count,
        "expected_reusable_tokens": record.reuse.policy_reusable_tokens.value,
        "expected_reusable_blocks": record.reuse.reusable_blocks.value,
        "engine_attested_reusable_tokens": record.reuse.engine_cached_tokens.value,
        "engine_attested_reusable_blocks": record.reuse.engine_cached_blocks.value,
        "observed_prompt_work_tokens": record.reuse.observed_prompt_tokens.value,
        "typed_verdict": (
            "unclassified" if record.verdict is None else record.verdict.value
        ),
        "output_identity": record.output.token_identity.value,
        "evaluator_correctness": record.output.correctness.value,
        "claim_eligibility": record.eligibility.to_dict(),
        "timing": None,
        "runtime_memory": None,
    }


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def render_truth_table(rows: list[dict[str, Any]]) -> str:
    """Render a stable, compact ASCII table."""

    headers = (
        "case",
        "input",
        "expected t/b",
        "attested t/b",
        "prompt work",
        "verdict",
        "output",
        "evaluator",
        "claims (output/perf/quality)",
    )
    body = [
        (
            row["case"],
            _cell(row["input_length"]),
            f"{_cell(row['expected_reusable_tokens'])}/"
            f"{_cell(row['expected_reusable_blocks'])}",
            f"{_cell(row['engine_attested_reusable_tokens'])}/"
            f"{_cell(row['engine_attested_reusable_blocks'])}",
            _cell(row["observed_prompt_work_tokens"]),
            row["typed_verdict"],
            _cell(row["output_identity"]),
            _cell(row["evaluator_correctness"]),
            "/".join(
                (
                    row["claim_eligibility"]["output_equivalence"],
                    row["claim_eligibility"]["performance"],
                    row["claim_eligibility"]["quality"],
                )
            ),
        )
        for row in rows
    ]
    widths = [
        max(len(str(value)) for value in (header, *(row[index] for row in body)))
        for index, header in enumerate(headers)
    ]

    def line(values: tuple[object, ...]) -> str:
        return " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join((line(headers), divider, *(line(row) for row in body)))


def _verify_public_text(name: str, text: str) -> None:
    for pattern, label in PRIVACY_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"{name} contains {label}")


def build_demo(output_dir: Path) -> dict[str, Any]:
    """Build and verify the deterministic public demo in a fresh directory."""

    if output_dir.exists():
        raise ValueError(f"output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    bundle_dir = output_dir / "bundle"
    generator_commit, generator_commit_at = source_commit()
    if generator_commit is None or generator_commit_at is None:
        raise ValueError("the deterministic public demo requires a Git checkout")
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(max_entries=DEMO_MAX_ENTRIES),
        requests=public_demo_requests(),
        cache_config=CacheConfig(
            namespace_id="synthetic-namespaces",
            cache_type="token_trie",
            max_entries=DEMO_MAX_ENTRIES,
            max_bytes=1 << 30,
        ),
        output_dir=bundle_dir,
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        publication_mode=PublicationMode.PUBLIC_SYNTHETIC,
        seed=DEMO_SEED,
        created_at=generator_commit_at,
        generated_at=generator_commit_at,
    )
    verification = verify_bundle(bundle_dir)
    rows = [_row(record) for record in records]
    table_document = {
        "schema_version": "1",
        "rows": rows,
        "claim_boundaries": list(CLAIM_BOUNDARIES),
    }
    table_text = canonical_json(table_document)
    _verify_public_text("truth-table.json", table_text)
    (output_dir / "truth-table.json").write_text(table_text, encoding="utf-8")

    demo_manifest = {
        "schema_version": "1",
        "title": "LLMTraceFX KV-cache truth auditor deterministic public demo",
        "command": DEMO_COMMAND,
        "verification_command": (
            "uv run --offline --no-sync python -I "
            "<output-dir>/bundle/evidence_bundle.py verify "
            "--public-dir <output-dir>/bundle --package-root ."
        ),
        "generator_commit": manifest.generator_commit,
        "workload_digest": manifest.workload_digest,
        "run_id": manifest.run_id,
        "case_count": len(rows),
        "bundle_checksum_manifest_sha256": _sha256(bundle_dir / "SHA256SUMS"),
        "report_sha256": _sha256(bundle_dir / "report.html"),
        "standalone_verifier_sha256": _sha256(bundle_dir / "evidence_bundle.py"),
        "truth_table_sha256": _sha256(output_dir / "truth-table.json"),
        "verification": verification,
        "privacy_mode": manifest.publication_mode.value,
        "timing_measurements": None,
        "runtime_memory_measurements": None,
        "claim_boundaries": list(CLAIM_BOUNDARIES),
    }
    manifest_text = canonical_json(demo_manifest)
    _verify_public_text("demo-manifest.json", manifest_text)
    (output_dir / "demo-manifest.json").write_text(manifest_text, encoding="utf-8")

    files = [
        *(f"bundle/{name}" for name in BUNDLE_FILES),
        "demo-manifest.json",
        "truth-table.json",
    ]
    checksums = "".join(
        f"{_sha256(output_dir / name).removeprefix('sha256:')}  {name}\n"
        for name in sorted(files)
    )
    (output_dir / "DEMO-SHA256SUMS").write_text(checksums, encoding="ascii")

    verifier = subprocess.run(
        [
            sys.executable,
            "-I",
            str(bundle_dir / "evidence_bundle.py"),
            "verify",
            "--public-dir",
            str(bundle_dir),
            "--package-root",
            str(Path(__file__).resolve().parents[2]),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if verifier.returncode != 0:
        raise RuntimeError(
            "standalone verifier failed: "
            + (verifier.stderr.strip() or verifier.stdout.strip())
        )
    return {
        "manifest": demo_manifest,
        "rows": rows,
        "table": render_truth_table(rows),
        "standalone_verifier": json.loads(verifier.stdout),
    }
