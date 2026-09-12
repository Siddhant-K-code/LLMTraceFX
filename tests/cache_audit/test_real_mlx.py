from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import fields, replace
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import llmtracefx.cache_audit.bundle as cache_bundle
import llmtracefx.cache_audit.real_mlx as real_mlx_module
import llmtracefx.cache_audit.runner as cache_runner
from llmtracefx.cache_audit.adapters.base import (
    AdapterAuditIdentity,
    CacheAuditCapability,
)
from llmtracefx.cache_audit.adapters.mlx import (
    REQUIRED_MLX_LM_VERSION,
    REQUIRED_MLX_VERSION,
    MLXLocalCacheAdapter,
    MLXStageObservation,
)
from llmtracefx.cache_audit.adapters.reference import ReferenceCacheAdapter
from llmtracefx.cache_audit.real_mlx import (
    CALIBRATION_ARRAY_NAMES,
    DEFAULT_CONVERSION_SUMMARY,
    EXPECTED_CONVERSION_SUMMARY_SHA256,
    EXPECTED_MODEL_ARTIFACT_DIGEST,
    LANE_IDS,
    MODEL_ID,
    REPLICATE_IDS,
    SCHEDULE_AFFINE_PERMUTATIONS,
    TOKENIZER_ID,
    FrozenMLXWorkload,
    LifecycleMLXAdapter,
    RealMLXExperimentError,
    StageRecorder,
    _digest_bytes,
    _experiment_cache_config,
    _json_bytes,
    _request_stage_memory,
    _safe_environment,
    _write_json,
    _write_recursive_checksums,
    assemble_aggregate,
    block_schedule,
    calibrate_outputs,
    compile_workload,
    load_workload,
    requests_for_replicate,
    sanitize_aggregate,
    verify_aggregate,
)
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import (
    CacheConfig,
    EvidenceBasis,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
    TerminalState,
    Verdict,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion import conversion_manifest_hash
from llmtracefx.optimizer.lab.qwen3_8b.conversion_manifest import ConversionManifest
from llmtracefx.optimizer.schema import Measurement, MetricProvenance


@pytest.fixture(autouse=True)
def _isolated_canonical_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        real_mlx_module,
        "CANONICAL_OUTPUT_WORKSPACE",
        tmp_path / "run",
    )
    monkeypatch.setattr(
        real_mlx_module,
        "CANONICAL_ATTEMPT_MARKER",
        tmp_path / "canonical-attempts" / "qwen3-4b-99469aa8-attempt-1.json",
    )


class FakeTokenizer:
    def _id(self, word: str) -> int:
        return 10 + sum((index + 1) * ord(char) for index, char in enumerate(word))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt and not enable_thinking
        tokens: list[int] = []
        for message in messages:
            tokens.append(1 if message["role"] == "user" else 3)
            tokens.extend(self._id(word) for word in message["content"].split())
            tokens.append(2)
        tokens.append(3)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [self._id(word) for word in text.split()]


class MLXIdentityReference:
    backend = "mlx_lm_local"

    def __init__(
        self,
        record_transform: Callable[[RequestEvidence], RequestEvidence] | None = None,
        stage_observer: Callable[[MLXStageObservation], None] | None = None,
    ) -> None:
        self._record_transform = record_transform
        self._stage_observer = stage_observer

    def capabilities(self) -> CacheAuditCapability:
        return CacheAuditCapability(backend=self.backend, supported=True)

    def audit_identity(self) -> AdapterAuditIdentity:
        return AdapterAuditIdentity(
            backend_version=REQUIRED_MLX_LM_VERSION,
            runtime_identity={
                "mlx": REQUIRED_MLX_VERSION,
                "mlx_lm": REQUIRED_MLX_LM_VERSION,
                "platform_machine": "arm64",
                "platform_system": "Darwin",
            },
            model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
            cache_type="mlx_lru_prompt_cache",
            max_entries=2,
            max_bytes=1 << 63,
        )

    def run(self, requests: tuple[RequestSpec, ...]) -> list[Any]:
        records = [
            replace(
                record,
                timing=replace(
                    record.timing,
                    client_ttft=Measurement(
                        value=0.01,
                        unit="s",
                        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
                    ),
                    in_process_first_token=Measurement(
                        value=0.009,
                        unit="s",
                        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
                    ),
                    total=Measurement(
                        value=0.02,
                        unit="s",
                        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
                    ),
                    scope="runtime_cache_fetch_and_mlx_lm_stream_generate",
                    exclusions=(
                        "prompt_cache_insertion",
                        "no_cache_baseline",
                        "harness_inspection_between_fetch_and_generation",
                    ),
                ),
                memory=replace(
                    record.memory,
                    runtime_active_bytes=replace(
                        record.memory.runtime_active_bytes,
                        value=100 + record.spec.order,
                        basis=EvidenceBasis.OBSERVED,
                        source="mlx.get_active_memory",
                        scope="process_global_allocator_gauge",
                        limitations=(),
                    ),
                    runtime_peak_bytes=replace(
                        record.memory.runtime_peak_bytes,
                        value=200 + record.spec.order,
                        basis=EvidenceBasis.OBSERVED,
                        source="mlx.get_peak_memory_after_reset_before_cache_fetch",
                        scope=(
                            "process_global_allocator_peak_from_cache_fetch_through_generation"
                        ),
                        limitations=(),
                    ),
                    allocator_cache_bytes=replace(
                        record.memory.allocator_cache_bytes,
                        value=50 + record.spec.order,
                        basis=EvidenceBasis.OBSERVED,
                        source="mlx.get_cache_memory",
                        scope="process_global_allocator_gauge",
                        limitations=(),
                    ),
                ),
                output=replace(
                    record.output,
                    output_token_ids=record.output.output_token_ids[:3],
                    baseline_token_ids=record.output.baseline_token_ids[:3],
                ),
            )
            for record in ReferenceCacheAdapter(max_entries=2, max_bytes=1 << 63).run(
                requests
            )
        ]
        if self._record_transform is not None:
            records = [self._record_transform(record) for record in records]
        if self._stage_observer is not None:
            for _, block_records in groupby(
                records, key=lambda record: record.spec.request_id.rsplit(":", 1)[0]
            ):
                grouped_records = tuple(block_records)
                self._stage_observer(
                    MLXStageObservation(
                        request_id=None,
                        stage="lifecycle_ready",
                        active_bytes=1,
                        peak_bytes=2,
                        allocator_cache_bytes=3,
                        logical_cache_bytes=4,
                    )
                )
                for record in grouped_records:
                    for stage in _captured_measurement_stages(record):
                        self._stage_observer(
                            MLXStageObservation(
                                request_id=record.spec.request_id,
                                stage=stage,
                                active_bytes=1,
                                peak_bytes=2,
                                allocator_cache_bytes=3,
                                logical_cache_bytes=4,
                            )
                        )
                for record in grouped_records:
                    if record.terminal_state is TerminalState.COMPLETED:
                        self._stage_observer(
                            MLXStageObservation(
                                request_id=record.spec.request_id,
                                stage="request_after_baseline",
                                active_bytes=1,
                                peak_bytes=2,
                                allocator_cache_bytes=3,
                                logical_cache_bytes=4,
                            )
                        )
        return records


def _captured_measurement_stages(record: RequestEvidence) -> tuple[str, ...]:
    stages = (
        "request_before_lookup",
        "request_after_lookup",
        "request_after_generation",
        "request_after_insertion",
    )
    if record.terminal_state is TerminalState.COMPLETED:
        return stages
    limitation_codes = {item.code for item in record.limitations}
    if "exact_empty_remainder_unsupported" in limitation_codes:
        return stages[:2]
    if "non_trimmable_cache_reuse_unsupported" in limitation_codes:
        return stages[:1]
    if limitation_codes & {
        "quantized_cache_unsupported",
        "rotating_cache_unsupported",
    }:
        return ()
    return stages


def _reference_output(tokens: tuple[int, ...]) -> tuple[int, ...]:
    payload = b"|".join(str(token).encode("ascii") for token in tokens)
    return tuple(hashlib.sha256(payload).digest()[:3])


def _calibration_outputs(
    lane: real_mlx_module.FrozenMLXLane,
) -> dict[str, tuple[int, ...]]:
    return {
        name: _reference_output(getattr(lane, name)) for name in CALIBRATION_ARRAY_NAMES
    }


def _calibrated_workload() -> FrozenMLXWorkload:
    workload = compile_workload(FakeTokenizer())
    return replace(
        workload,
        lanes=tuple(
            replace(lane, calibration_outputs=_calibration_outputs(lane))
            for lane in workload.lanes
        ),
    )


def _runtime_packages() -> dict[str, dict[str, str | int]]:
    return json.loads(json.dumps(real_mlx_module._expected_runtime_package_identity()))


def test_runtime_distribution_closure_and_expected_identity_are_exact() -> None:
    assert real_mlx_module._RUNTIME_DISTRIBUTION_VERSIONS == {
        "anyio": "4.9.0",
        "certifi": "2025.7.9",
        "click": "8.1.8",
        "filelock": "3.32.4",
        "fsspec": "2026.7.0",
        "h11": "0.16.0",
        "hf-xet": "1.6.0",
        "httpcore": "1.0.9",
        "httpx": "0.28.1",
        "huggingface-hub": "1.13.0",
        "idna": "3.15",
        "jinja2": "3.1.6",
        "markupsafe": "3.0.2",
        "markdown-it-py": "3.0.0",
        "mdurl": "0.1.2",
        "mlx": "0.32.2",
        "mlx-lm": "0.31.3",
        "mlx-metal": "0.32.2",
        "numpy": "2.2.6",
        "packaging": "26.3",
        "protobuf": "6.33.5",
        "pyyaml": "6.0.2",
        "pygments": "2.20.0",
        "regex": "2026.7.19",
        "rich": "14.0.0",
        "safetensors": "0.8.0",
        "sentencepiece": "0.2.2",
        "shellingham": "1.5.4",
        "sniffio": "1.3.1",
        "tokenizers": "0.23.1",
        "tqdm": "4.70.0",
        "transformers": "5.16.1",
        "typer": "0.16.0",
        "typing-extensions": "4.14.1",
    }
    expected = real_mlx_module._expected_runtime_package_identity()
    assert set(expected) == set(real_mlx_module._RUNTIME_DISTRIBUTION_VERSIONS)
    assert all(
        set(entry)
        == {
            "distribution",
            "version",
            "trusted_root",
            "file_count",
            "total_bytes",
            "tree_sha256",
        }
        for entry in expected.values()
    )


def test_runtime_package_identity_requires_canonical_expected_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _runtime_packages()
    monkeypatch.setattr(
        real_mlx_module,
        "_compute_runtime_package_identity",
        lambda _output_workspace: expected,
    )
    assert real_mlx_module._runtime_package_identity(tmp_path / "run") == expected

    changed = json.loads(json.dumps(expected))
    changed["mlx"]["tree_sha256"] = "sha256:" + "0" * 64
    monkeypatch.setattr(
        real_mlx_module,
        "_compute_runtime_package_identity",
        lambda _output_workspace: changed,
    )
    with pytest.raises(RealMLXExperimentError, match="canonical expected tree"):
        real_mlx_module._runtime_package_identity(tmp_path / "run")


def test_expected_runtime_identity_rejects_allowlist_tampering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tampered = tmp_path / "runtime-identity.json"
    tampered.write_bytes(
        real_mlx_module.EXPECTED_RUNTIME_PACKAGE_IDENTITIES.read_bytes()
    )
    tampered.write_bytes(
        tampered.read_bytes().replace(b'"anyio"', b'"anyio-tampered"', 1)
    )
    monkeypatch.setattr(
        real_mlx_module, "EXPECTED_RUNTIME_PACKAGE_IDENTITIES", tampered
    )
    with pytest.raises(RealMLXExperimentError, match="identity digest mismatch"):
        real_mlx_module._expected_runtime_package_identity()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mach RSS is macOS-only")
def test_current_rss_uses_in_process_mach_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_subprocess(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("current RSS must not launch a subprocess")

    monkeypatch.setattr(real_mlx_module.subprocess, "run", refuse_subprocess)
    rss = real_mlx_module._current_rss_bytes()
    assert isinstance(rss, int)
    assert rss > 0
    monkeypatch.setattr(
        real_mlx_module.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )
    assert real_mlx_module._current_rss_bytes() is None


@pytest.mark.skipif(platform.system() != "Darwin", reason="sandbox-exec is macOS-only")
def test_exact_sandbox_policy_model_free_smoke(tmp_path: Path) -> None:
    real_mlx_module._ensure_non_repository_cwd(tmp_path)
    code = """
from llmtracefx.cache_audit.bundle import _git_package_digest, package_source_digest
from pathlib import Path
from llmtracefx.cache_audit.real_mlx import (
    _PROJECT_ROOT,
    RealMLXExperimentError,
    _current_rss_bytes,
    _system_memory_free_percent,
    _system_swap_used_bytes,
    _validate_supervisor_source,
)
from llmtracefx.cache_audit.runner import source_commit
commit, _ = source_commit()
assert commit is not None
assert _current_rss_bytes() is not None
assert _system_swap_used_bytes() is not None
assert _system_memory_free_percent() is not None
assert package_source_digest().startswith("sha256:")
assert _git_package_digest(_PROJECT_ROOT, commit).startswith("sha256:")
try:
    validated = _validate_supervisor_source(commit)
except RealMLXExperimentError as exc:
    assert str(exc) == "installed package source digest does not match expected commit tree"
else:
    assert validated == package_source_digest()
"""
    command = [
        "/usr/bin/sandbox-exec",
        "-p",
        real_mlx_module.SANDBOX_POLICY,
        sys.executable,
        "-I",
        "-c",
        code,
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=real_mlx_module._offline_child_environment("a" * 32),
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_external_venv_output_siblings_hash_complete_safe_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "venv/lib/python3.13/site-packages"
    scripts = tmp_path / "venv/bin"
    output_parent = tmp_path / "output"
    site_packages.mkdir(parents=True)
    scripts.mkdir(parents=True)
    output_parent.mkdir()
    distributions: dict[str, Any] = {}

    class FakeDistribution:
        def __init__(self, name: str, version: str) -> None:
            package_name = name.replace("-", "_")
            self.version = version
            self.metadata = {"Name": name}
            self.files = [
                Path(package_name) / "__init__.py",
                Path(package_name) / "native.so",
                Path(package_name) / "data" / "runtime.json",
                Path(f"{package_name}-1.dist-info") / "METADATA",
                Path(f"{package_name}-1.dist-info") / "RECORD",
                Path("../../../bin") / f"{package_name}-tool",
            ]

        def locate_file(self, item: object) -> Path:
            return site_packages / Path(str(item))

    for (
        distribution_name,
        version,
    ) in real_mlx_module._RUNTIME_DISTRIBUTION_VERSIONS.items():
        package_name = distribution_name.replace("-", "_")
        package = site_packages / package_name
        (package / "data").mkdir(parents=True)
        dist_info = site_packages / f"{package_name}-1.dist-info"
        dist_info.mkdir()
        (package / "__init__.py").write_text(f"{distribution_name}\n", encoding="ascii")
        (package / "native.so").write_bytes(b"\x00native")
        (package / "data" / "runtime.json").write_text(
            '{"runtime":true}\n', encoding="ascii"
        )
        (dist_info / "METADATA").write_text(
            f"Name: {distribution_name}\nVersion: {version}\n", encoding="ascii"
        )
        (dist_info / "RECORD").write_text(
            f"absolute-root={site_packages}\n", encoding="ascii"
        )
        (scripts / f"{package_name}-tool").write_text(
            f"#!/bin/sh\n# {site_packages}\n", encoding="ascii"
        )
        distributions[distribution_name] = FakeDistribution(distribution_name, version)

    monkeypatch.setattr(
        real_mlx_module,
        "_trusted_site_roots",
        lambda: (("purelib", site_packages, scripts),),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_expected_distribution_uniqueness",
        lambda _root: distributions,
    )

    output_workspace = output_parent / "run"
    assert output_workspace.parent.parent == scripts.parent.parent
    identity = real_mlx_module._compute_runtime_package_identity(output_workspace)
    assert all(
        entry["file_count"] == 4
        and entry["total_bytes"] > 0
        and entry["trusted_root"] == "purelib"
        and str(entry["tree_sha256"]).startswith("sha256:")
        for entry in identity.values()
    )
    with pytest.raises(RealMLXExperimentError, match="must not overlap"):
        real_mlx_module._compute_runtime_package_identity(
            scripts.parent / "nested-output"
        )
    runtime_data = site_packages / "mlx_lm" / "data" / "runtime.json"
    runtime_data.write_text('{"runtime":false}\n', encoding="ascii")
    tampered = real_mlx_module._compute_runtime_package_identity(output_workspace)
    assert tampered["mlx-lm"]["tree_sha256"] != identity["mlx-lm"]["tree_sha256"]
    assert tampered["mlx"]["tree_sha256"] == identity["mlx"]["tree_sha256"]

    distributions["mlx-lm"].files.append(Path("mlx_lm") / "injected.metallib")
    (site_packages / "mlx_lm" / "injected.metallib").write_bytes(b"injected")
    injected = real_mlx_module._compute_runtime_package_identity(output_workspace)
    assert injected["mlx-lm"]["file_count"] == 5
    assert injected["mlx-lm"]["tree_sha256"] != tampered["mlx-lm"]["tree_sha256"]

    duplicate = site_packages / "mlx_lm_duplicate.dist-info"
    duplicate.mkdir()
    (duplicate / "METADATA").write_text(
        "Name: mlx_lm\nVersion: 0.31.3\n", encoding="ascii"
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_expected_distribution_uniqueness",
        lambda _root: (_ for _ in ()).throw(
            RealMLXExperimentError(
                "runtime package closure has duplicate distributions"
            )
        ),
    )
    with pytest.raises(RealMLXExperimentError, match="duplicate distributions"):
        real_mlx_module._compute_runtime_package_identity(output_workspace)


def test_runtime_package_identity_rejects_missing_and_symlinked_tree_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site_packages = tmp_path / "venv/lib/python3.13/site-packages"
    scripts = tmp_path / "venv/bin"
    output_parent = tmp_path / "output"
    package = site_packages / "mlx"
    package.mkdir(parents=True)
    scripts.mkdir()
    output_parent.mkdir()
    (package / "__init__.py").write_text("mlx\n", encoding="ascii")

    class FakeDistribution:
        version = real_mlx_module.REQUIRED_MLX_VERSION
        metadata = {"Name": "mlx"}
        files = [Path("mlx/__init__.py"), Path("mlx/missing.so")]

        @staticmethod
        def locate_file(item: object) -> Path:
            return site_packages / Path(str(item))

    monkeypatch.setattr(
        real_mlx_module,
        "_trusted_site_roots",
        lambda: (("purelib", site_packages, scripts),),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_expected_distribution_uniqueness",
        lambda _root: {"mlx": FakeDistribution()},
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_RUNTIME_DISTRIBUTION_VERSIONS",
        {"mlx": real_mlx_module.REQUIRED_MLX_VERSION},
    )
    with pytest.raises(RealMLXExperimentError, match="missing file"):
        real_mlx_module._compute_runtime_package_identity(output_parent / "run")

    FakeDistribution.files = [Path("mlx/__init__.py"), Path("mlx/linked.so")]
    (package / "linked.so").symlink_to(package / "__init__.py")
    with pytest.raises(RealMLXExperimentError, match="symlink"):
        real_mlx_module._compute_runtime_package_identity(output_parent / "run")

    (package / "linked.so").unlink()
    FakeDistribution.files = [Path("mlx/__init__.py"), Path("mlx/__pycache__/bad.pyc")]
    with pytest.raises(RealMLXExperimentError, match="unsafe bytecode"):
        real_mlx_module._compute_runtime_package_identity(output_parent / "run")

    FakeDistribution.files = [Path("mlx/__init__.py"), Path("../../escape.py")]
    with pytest.raises(RealMLXExperimentError, match="missing file|out of root"):
        real_mlx_module._compute_runtime_package_identity(output_parent / "run")


def test_runtime_package_identity_rejects_local_startup_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site-packages"
    scripts = tmp_path / "bin"
    site.mkdir()
    scripts.mkdir()
    (site / "unsafe.pth").write_text("import unsafe\n")
    monkeypatch.setattr(
        real_mlx_module,
        "_trusted_site_roots",
        lambda: (("purelib", site, scripts),),
    )
    with pytest.raises(RealMLXExperimentError, match="startup hook"):
        real_mlx_module._compute_runtime_package_identity(tmp_path / "run")


def test_heavy_process_gate_counts_every_nonexcluded_large_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3
    monkeypatch.setattr(
        real_mlx_module,
        "_process_inventory",
        lambda: [
            (10, 1, 2 * gib),
            (11, 10, 2 * gib),
            (20, 1, gib),
            (22, 1, 3 * gib),
            (21, 1, gib - 1),
        ],
    )

    assert real_mlx_module._heavy_process_summary({10}) == (
        2,
        ("other_large_process",),
    )


def test_any_replicate_failure_is_ineligible() -> None:
    def rows(failed_reason: str) -> list[dict[str, str]]:
        return [
            {
                "event": "finalized",
                "replicate_id": replicate_id,
                "status": "failed" if replicate_id == "replicate-5" else "complete",
                "reason": (
                    failed_reason if replicate_id == "replicate-5" else "completed"
                ),
            }
            for replicate_id in REPLICATE_IDS
        ]

    assert not real_mlx_module._aggregate_eligibility_from_rows(
        rows("preflight_host_chip_mismatch")
    )
    assert not real_mlx_module._aggregate_eligibility_from_rows(
        rows("preflight_host_memory_mismatch")
    )
    assert not real_mlx_module._aggregate_eligibility_from_rows(
        rows("preflight_memory_below_floor")
    )
    complete = rows("unused")
    complete[-1].update(status="complete", reason="completed")
    assert real_mlx_module._aggregate_eligibility_from_rows(complete)


def test_import_shadow_scan_is_narrow_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "output-parent"
    output_parent.mkdir()
    output_workspace = output_parent / "run"
    (repository / "unrelated-ignored-directory").mkdir()
    monkeypatch.setattr(real_mlx_module, "_PROJECT_ROOT", repository)

    real_mlx_module._reject_import_shadows(output_workspace)
    (output_parent / "mlx.py").write_text("raise RuntimeError('shadowed')\n")
    with pytest.raises(RealMLXExperimentError, match="import shadow"):
        real_mlx_module._reject_import_shadows(output_workspace)
    (output_parent / "mlx.py").unlink()
    (repository / "mlx_lm").mkdir()
    with pytest.raises(RealMLXExperimentError, match="import shadow"):
        real_mlx_module._reject_import_shadows(output_workspace)
    (repository / "mlx_lm").rmdir()
    (output_parent / "numpy.py").write_text("raise RuntimeError('shadowed')\n")
    with pytest.raises(RealMLXExperimentError, match="import shadow"):
        real_mlx_module._reject_import_shadows(output_workspace)
    (output_parent / "numpy.py").unlink()
    (repository / "tokenizers").mkdir()
    with pytest.raises(RealMLXExperimentError, match="import shadow"):
        real_mlx_module._reject_import_shadows(output_workspace)


def test_source_validation_ignores_checkout_dirt_and_rejects_package_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_commit = "a" * 40
    monkeypatch.setattr(
        real_mlx_module,
        "source_commit",
        lambda: (expected_commit, "2026-01-01T00:00:00Z"),
    )
    calls: list[list[str]] = []

    def clean_git(
        command: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = expected_commit + "\n" if "rev-parse" in command else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(real_mlx_module.subprocess, "run", clean_git)
    monkeypatch.setattr(
        real_mlx_module,
        "package_source_digest",
        lambda: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_git_package_digest",
        lambda *_args: "sha256:" + "c" * 64,
    )
    with pytest.raises(RealMLXExperimentError, match="does not match expected commit"):
        real_mlx_module._validate_supervisor_source(expected_commit)
    assert not any("status" in command for command in calls)


def test_workload_invariants_and_counterbalanced_schedule() -> None:
    workload = compile_workload(FakeTokenizer())
    expected_lengths = {
        "1k": (1025, 513),
        "4k": (4097, 2049),
    }
    for lane in workload.lanes:
        base, eviction = expected_lengths[lane.lane_id]
        assert len(lane.base) == base
        assert {field.name for field in fields(lane)} == {
            "lane_id",
            "base",
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "calibration_outputs",
        }
        assert {len(lane.eviction_a), len(lane.eviction_b), len(lane.eviction_c)} == {
            eviction
        }
        assert lane.mutation_137[137] != lane.base[137]
        assert lane.mutation_256[256] != lane.base[256]
        assert [
            index
            for index, (left, right) in enumerate(
                zip(lane.base, lane.mutation_137, strict=True)
            )
            if left != right
        ] == [137]
        assert [
            index
            for index, (left, right) in enumerate(
                zip(lane.base, lane.mutation_256, strict=True)
            )
            if left != right
        ] == [256]
        assert lane.suffix_change[: base - 32] == lane.base[: base - 32]
        assert lane.suffix_change[-16:] == lane.base[-16:]
        assert lane.different_ids[0] == lane.base[0]
        assert lane.different_ids != lane.base
        assert all(
            left != right
            for left, right in zip(
                lane.suffix_change[-32:-16], lane.base[-32:-16], strict=True
            )
        )
        assert lane.base[-1] == lane.different_ids[-1] == lane.suffix_change[-1] == 3
        assert lane.eviction_a[-1] == lane.eviction_b[-1] == lane.eviction_c[-1] == 3
    calibrated_source = _calibrated_workload()
    payload = calibrated_source.to_dict()
    assert all(
        set(payload["lanes"][lane_id]["arrays"])
        == {
            "base",
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "calibration_outputs",
        }
        for lane_id in LANE_IDS
    )
    calibrated = FrozenMLXWorkload.from_dict(payload)
    tampered_payload = json.loads(json.dumps(payload))
    tampered_payload["lanes"]["1k"]["lane_id"] = "4k"
    with pytest.raises(RealMLXExperimentError, match="lane identity"):
        FrozenMLXWorkload.from_dict(tampered_payload)
    schedules = [block_schedule(item) for item in REPLICATE_IDS]
    assert len(set(schedules)) == len(REPLICATE_IDS)
    assert len({schedule[0] for schedule in schedules}) == len(REPLICATE_IDS)
    assert SCHEDULE_AFFINE_PERMUTATIONS == (
        (0, 1),
        (1, 3),
        (2, 5),
        (3, 9),
        (4, 11),
        (5, 13),
    )
    assert all(
        block.split(":", 1)[0] != schedules[0][index + 1].split(":", 1)[0]
        for index, block in enumerate(schedules[0][:-1])
    )
    adjacency_sets = {
        frozenset(zip(schedule[:-1], schedule[1:], strict=True))
        for schedule in schedules
    }
    assert len(adjacency_sets) == len(REPLICATE_IDS)
    for lane_id in LANE_IDS:
        interior = f"{lane_id}:interior-mutation"
        boundary = f"{lane_id}:allocation-step-mutation"
        positions = [
            (schedule.index(interior), schedule.index(boundary))
            for schedule in schedules
        ]
        assert all(abs(left - right) != 1 for left, right in positions)
        assert {left < right for left, right in positions} == {False, True}
    for replicate_id, schedule in zip(REPLICATE_IDS, schedules, strict=True):
        assert len(schedule) == 14
        assert set(schedule) == {
            f"{lane_id}:{case}"
            for lane_id in LANE_IDS
            for case in (
                "cold-exact-duplicate",
                "interior-mutation",
                "allocation-step-mutation",
                "same-length-different-ids",
                "suffix-only-change",
                "namespace-isolation",
                "capacity-eviction",
            )
        }
        requests = requests_for_replicate(calibrated, replicate_id)
        observed = tuple(
            dict.fromkeys(request.request_id.rsplit(":", 1)[0] for request in requests)
        )
        assert observed == schedule
        assert len(requests) == 36
        assert {
            lane_id: sum(
                request.request_id.startswith(f"{lane_id}:") for request in requests
            )
            for lane_id in LANE_IDS
        } == {"1k": 18, "4k": 18}
        assert max(request.output_tokens for request in requests) == 8
        for lane_id in LANE_IDS:
            lane_requests = [
                request
                for request in requests
                if request.request_id.startswith(f"{lane_id}:")
            ]
            assert {request.input_token_count for request in lane_requests} == {
                *expected_lengths[lane_id],
            }
            eviction_requests = [
                request
                for request in requests
                if request.request_id.startswith(f"{lane_id}:capacity-eviction:")
            ]
            namespaces = {
                request.request_id.rsplit(":", 1)[-1]: request.namespace_id
                for request in eviction_requests
            }
            assert namespaces["a-seed"] == namespaces["a-hit"] == namespaces["a-miss"]
            assert len({namespaces["a-seed"], namespaces["b"], namespaces["c"]}) == 3


def test_qwen3_4b_conversion_metadata_is_exact() -> None:
    summary_bytes = DEFAULT_CONVERSION_SUMMARY.read_bytes()
    assert (
        _digest_bytes(summary_bytes) == f"sha256:{EXPECTED_CONVERSION_SUMMARY_SHA256}"
    )
    summary = json.loads(summary_bytes)
    assert summary["source"]["official_id"] == "Qwen/Qwen3-4B"
    assert (
        summary["source"]["official_revision"]
        == "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert summary["source"]["license"] == "Apache-2.0"
    assert summary["converter"]["version"] == "0.31.3"
    assert (
        summary["converter"]["git_revision"]
        == "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
    )
    assert summary["parameters"]["q_group_size"] == 64
    assert summary["parameters"]["q_bits"] == 4
    assert summary["parameters"]["q_mode"] == "affine"
    assert summary["output"]["repository_id"] == MODEL_ID
    assert summary["output"]["total_bytes"] == 2_274_515_269
    assert len(summary["output"]["files"]) == 8
    assert EXPECTED_MODEL_ARTIFACT_DIGEST == (
        "sha256:057a37f4ebc76420f8ab2edb17bc8442e050c8d13f7334f829356e2f9cab6802"
    )
    assert TOKENIZER_ID == ("Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c")
    manifest = json.loads(
        (
            DEFAULT_CONVERSION_SUMMARY.parent / "qwen3-4b-conversion-manifest-v1.json"
        ).read_text()
    )
    assert (
        conversion_manifest_hash(ConversionManifest.from_dict(manifest))
        == summary["conversion_manifest_hash"]
    )
    assert manifest["source"]["official_id"] == "Qwen/Qwen3-4B"
    assert (
        manifest["source"]["official_revision"]
        == "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert manifest["source"]["license"] == "Apache-2.0"
    assert manifest["expected_output_bytes"] == 2_300_000_000


def test_calibrated_workload_refuses_unpinned_digests(tmp_path: Path) -> None:
    calibrated = _calibrated_workload()
    path = tmp_path / "calibrated.json"
    _write_json(path, calibrated.to_dict())

    with pytest.raises(RealMLXExperimentError, match="contract mismatch"):
        load_workload(path, calibrated=True)


def test_stage_records_have_explicit_nonconflated_scopes(tmp_path: Path) -> None:
    recorder = StageRecorder(
        "replicate-0",
        rss_reader=lambda: 123,
        swap_reader=lambda: 456,
        memory_reader=lambda: 78.0,
    )
    recorder(
        MLXStageObservation(
            request_id="block:request",
            stage="request_after_lookup",
            active_bytes=1,
            peak_bytes=2,
            allocator_cache_bytes=3,
            logical_cache_bytes=4,
        )
    )
    output = tmp_path / "stages.jsonl"
    recorder.write(output)
    row = json.loads(output.read_text())
    assert row["allocator"]["scope"] == "mlx_process_global_allocator"
    assert row["logical_cache"]["scope"] == "current_lru_entry_when_observable"
    assert row["process_rss"] == {
        "bytes": 123,
        "scope": "current_replicate_child_process_only",
    }
    assert row["system_swap"] == {"used_bytes": 456, "scope": "system_wide"}
    assert row["system_memory"] == {
        "free_percent": 78.0,
        "scope": "system_wide_memory_pressure",
    }
    assert row["thermal_power"] == {
        "thermal_state": None,
        "power_watts": None,
        "scope": "unavailable_without_safe_collector",
    }
    assert "pid" not in output.read_text()


def test_request_stage_memory_excludes_fresh_baseline_observation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stages.jsonl"
    rows = []
    for stage, rss, swap, free in (
        ("request_before_lookup", 100, 200, 80.0),
        ("request_after_generation", 150, 250, 75.0),
        ("request_after_baseline", 9_999, 8_888, 1.0),
    ):
        rows.append(
            {
                "request_id": "request",
                "stage": stage,
                "process_rss": {"bytes": rss},
                "system_swap": {"used_bytes": swap},
                "system_memory": {"free_percent": free},
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )

    assert _request_stage_memory(path) == {
        "request": {
            "process_rss_bytes": 150,
            "system_swap_used_bytes": 250,
            "system_memory_free_percent": 75.0,
        }
    }


def test_calibration_repeats_base_with_fresh_adapters() -> None:
    workload = compile_workload(FakeTokenizer())
    creations = 0
    observed_inputs: list[tuple[int, ...]] = []

    def factory() -> Any:
        nonlocal creations
        creations += 1
        adapter = ReferenceCacheAdapter()
        original = adapter.run

        def run(requests: tuple[RequestSpec, ...]) -> list[Any]:
            observed_inputs.extend(
                request.input_token_ids or () for request in requests
            )
            records = original(requests)
            return [
                replace(
                    record,
                    output=replace(
                        record.output,
                        output_token_ids=_reference_output(
                            record.spec.input_token_ids or ()
                        ),
                        baseline_token_ids=_reference_output(
                            record.spec.input_token_ids or ()
                        ),
                    ),
                )
                for record in records
            ]

        adapter.run = run  # type: ignore[method-assign]
        return adapter

    calibrated = calibrate_outputs(workload, factory)
    assert creations == 18
    assert observed_inputs == [
        tokens
        for lane in workload.lanes
        for tokens in (
            *(getattr(lane, name) for name in CALIBRATION_ARRAY_NAMES),
            lane.base,
        )
    ]
    for lane in calibrated.lanes:
        assert lane.calibration_outputs == _calibration_outputs(lane)
        assert all(
            len(output) == 3 for output in (lane.calibration_outputs or {}).values()
        )


def test_failed_replicate_finishes_with_one_valid_terminal_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = tmp_path / "missing-workload.json"
    output = tmp_path / "replicate"
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _expected: "sha256:" + "b" * 64,
    )

    with pytest.raises(RealMLXExperimentError):
        real_mlx_module.run_replicate(
            workload,
            output,
            replicate_id="replicate-0",
            model_dir=tmp_path / "model",
            expected_commit="a" * 40,
        )

    assert real_mlx_module.verify_replicate(
        output,
        replicate_id="replicate-0",
        public=False,
    ) == {
        "replicate_id": "replicate-0",
        "status": "failed",
        "reason": "replicate_execution_failed",
    }
    attempt = json.loads((output / "attempt.json").read_text())
    assert attempt["reason"] == "replicate_execution_failed"


def test_lifecycle_adapter_batches_each_block_before_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped: list[tuple[str, ...]] = []

    class Probe:
        mlx_version = REQUIRED_MLX_VERSION
        mlx_lm_version = REQUIRED_MLX_LM_VERSION
        platform_machine = "arm64"
        platform_system = "Darwin"

    class FakeAdapter:
        def __init__(self, **_: Any) -> None:
            pass

        def run(self, requests: tuple[RequestSpec, ...]) -> list[Any]:
            grouped.append(tuple(request.request_id for request in requests))
            return []

    monkeypatch.setattr(real_mlx_module, "MLXLocalCacheAdapter", FakeAdapter)
    monkeypatch.setattr(
        real_mlx_module,
        "check_mlx_capabilities",
        lambda *_args, **_kwargs: CacheAuditCapability(
            backend="mlx_lm_local", supported=True
        ),
    )
    adapter = LifecycleMLXAdapter(
        runtime_factory=Probe,
        model="model",
        tokenizer="tokenizer",
        model_key="key",
        model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
        correctness_evaluator=lambda _: True,
        stage_observer=lambda _: None,
    )
    adapter.run(
        (
            _request_spec("1k:cold-exact-duplicate:control", 0),
            _request_spec("1k:cold-exact-duplicate:treatment", 1),
            _request_spec("4k:cold-exact-duplicate:control", 2),
        )
    )

    assert grouped == [
        (
            "1k:cold-exact-duplicate:control",
            "1k:cold-exact-duplicate:treatment",
        ),
        ("4k:cold-exact-duplicate:control",),
    ]


def test_replicate_warmup_uses_one_direct_generation_per_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _calibrated_workload()
    creations = 0
    generated_inputs: list[tuple[int, ...]] = []
    cache_creations = 0
    teardowns = 0

    class WarmupRuntime:
        def __init__(self, **_kwargs: Any) -> None:
            nonlocal creations
            creations += 1

        def make_cache(self, _model: Any) -> object:
            nonlocal cache_creations
            cache_creations += 1
            return object()

        def generate(
            self,
            _model: Any,
            _tokenizer: Any,
            _cache: Any,
            token_ids: tuple[int, ...],
            *,
            max_tokens: int,
            prompt_progress_callback: Any,
        ) -> Any:
            assert max_tokens == real_mlx_module.MAX_OUTPUT_TOKENS
            assert callable(prompt_progress_callback)
            generated_inputs.append(token_ids)
            lane = next(lane for lane in workload.lanes if lane.base == token_ids)
            for token in (lane.calibration_outputs or {})["base"]:
                yield type("Step", (), {"token": token})()

        def synchronize(self) -> None:
            pass

    def teardown() -> dict[str, Any]:
        nonlocal teardowns
        teardowns += 1
        return {}

    monkeypatch.setattr(real_mlx_module, "ProductionMLXRuntime", WarmupRuntime)
    monkeypatch.setattr(real_mlx_module, "output_is_cache_ok", lambda *_args: True)
    monkeypatch.setattr(real_mlx_module, "_teardown", teardown)

    assert (
        real_mlx_module._run_warmup_lanes(
            workload,
            model="loaded-model",
            tokenizer="loaded-tokenizer",
        )
        == LANE_IDS
    )
    assert creations == 2
    assert cache_creations == 2
    assert teardowns == 2
    assert generated_inputs == [lane.base for lane in workload.lanes]


def _request_spec(request_id: str, order: int) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        scenario=real_mlx_module.ScenarioKind.COLD,
        order=order,
        input_token_ids=(order + 1,),
        input_token_count=1,
    )


def _make_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_last: bool = False,
    record_transform: Callable[[RequestEvidence], RequestEvidence] | None = None,
) -> Path:
    expected_commit = "a" * 40
    monkeypatch.setattr(
        cache_runner,
        "source_commit",
        lambda: (expected_commit, "2026-01-01T00:00:00Z"),
    )
    monkeypatch.setattr(
        cache_bundle,
        "_verify_manifest_chronology",
        lambda *_args, **_kwargs: "verified",
    )
    workspace = tmp_path / "run-workspace"
    workspace.mkdir()
    attempts = workspace / "attempts"
    attempts.mkdir()
    (workspace / "private-artifacts").mkdir()
    workload = _calibrated_workload()
    teardown = {
        "schema_version": "1",
        "cache_lifecycles_released": True,
        "garbage_collection_completed": True,
        "mlx_allocator_cleanup": "completed",
        "allocator_cache_bytes_after": 0,
        "scope": "current_replicate_child_process",
    }
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        workload.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes},
    )
    for index, replicate_id in enumerate(REPLICATE_IDS):
        replicate = attempts / replicate_id
        replicate.mkdir()
        if failed_last and index == 5:
            _write_json(
                replicate / "attempt.json",
                {
                    "schema_version": "1",
                    "replicate_id": replicate_id,
                    "status": "failed",
                    "replacement": False,
                    "reason": "preflight_heavy_process_present",
                    "failed_at": "2026-01-01T00:00:00.000000Z",
                },
            )
            _write_json(
                replicate / "teardown.json",
                {
                    **teardown,
                    "cache_lifecycles_released": None,
                    "garbage_collection_completed": None,
                    "mlx_allocator_cleanup": "externally_terminated",
                    "allocator_cache_bytes_after": None,
                },
            )
            continue
        bundle = replicate / "bundle"
        requests = requests_for_replicate(workload, replicate_id)
        recorder = StageRecorder(
            replicate_id,
            rss_reader=lambda: 100,
            swap_reader=lambda: 200,
            memory_reader=lambda: 80.0,
        )
        run_audit(
            adapter=MLXIdentityReference(record_transform, recorder),
            requests=requests,
            cache_config=CacheConfig(
                namespace_id="experiment-namespaces",
                cache_type="mlx_lru_prompt_cache",
                max_entries=2,
                max_bytes=1 << 63,
            ),
            output_dir=bundle,
            backend_version=REQUIRED_MLX_LM_VERSION,
            model_id=MODEL_ID,
            tokenizer_id=TOKENIZER_ID,
            model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
            publication_mode=PublicationMode.PRIVATE,
            seed=index,
        )
        _write_json(
            replicate / "attempt.json",
            {
                "schema_version": "2",
                "replicate_id": replicate_id,
                "status": "complete",
                "rotation": list(block_schedule(replicate_id)),
                "request_count": len(requests),
                "lane_request_counts": {"1k": 18, "4k": 18},
                "independent_unit": True,
                "replacement": False,
                "expected_commit": expected_commit,
                "warmup_lanes": list(LANE_IDS),
                "warmup_excluded": True,
                "frozen_workload_digest": workload.to_dict()["workload_digest"],
                "frozen_lane_digests": {
                    lane.lane_id: lane.to_dict()["lane_digest"]
                    for lane in workload.lanes
                },
                "model_artifact_digest": EXPECTED_MODEL_ARTIFACT_DIGEST,
                "model_id": MODEL_ID,
                "tokenizer_id": TOKENIZER_ID,
                "runtime_identity_digest": _digest_bytes(
                    _json_bytes(real_mlx_module._EXPECTED_RUNTIME_IDENTITY)
                ),
                "runtime_packages_digest": _digest_bytes(
                    _json_bytes(_runtime_packages())
                ),
                "cache_config_digest": _digest_bytes(
                    _json_bytes(_experiment_cache_config().to_dict())
                ),
            },
        )
        _write_json(
            replicate / "environment.json",
            {
                **_safe_environment(instance_id=f"{index + 1:032x}"),
                "mlx": REQUIRED_MLX_VERSION,
                "mlx_lm": REQUIRED_MLX_LM_VERSION,
            },
        )
        _write_json(replicate / "workload.json", workload.to_dict())
        recorder.write(replicate / "stages.jsonl")
        _write_json(replicate / "teardown.json", teardown)
    package_digest = real_mlx_module.package_source_digest()
    binding = real_mlx_module._run_binding(
        expected_commit=expected_commit,
        package_digest=package_digest,
        workload=workload,
        runtime_packages=_runtime_packages(),
        run_attempt=1,
        attempt_marker_digest="sha256:" + "1" * 64,
    )
    ledger = workspace / "run-ledger.jsonl"
    real_mlx_module._create_run_ledger(ledger, binding)
    sequence = len(REPLICATE_IDS) + 1
    observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": ("host_process_table_excluding_supervisor_and_child_tree"),
        },
    }
    for replicate_id in REPLICATE_IDS:
        attempt = json.loads((attempts / replicate_id / "attempt.json").read_text())
        preflight_failed = attempt["status"] == "failed"
        preflight_observation = (
            {
                **observation,
                "heavy_process_count": 1,
                "heavy_process_categories": ["other_large_process"],
            }
            if preflight_failed
            else observation
        )
        real_mlx_module._ledger_event(
            ledger,
            sequence=sequence,
            replicate_id=replicate_id,
            event="preflight",
            status="failed" if preflight_failed else "passed",
            reason=attempt.get("reason") if preflight_failed else None,
            observation=preflight_observation,
        )
        sequence += 1
        if attempt["status"] == "complete":
            real_mlx_module._ledger_event(
                ledger,
                sequence=sequence,
                replicate_id=replicate_id,
                event="started",
                status="running",
                reason=None,
                child_instance_id_digest=(
                    "sha256:"
                    + hashlib.sha256(
                        f"{REPLICATE_IDS.index(replicate_id) + 1:032x}".encode()
                    ).hexdigest()
                ),
            )
            sequence += 1
            real_mlx_module._ledger_event(
                ledger,
                sequence=sequence,
                replicate_id=replicate_id,
                event="postflight",
                status="passed",
                reason=None,
                observation=observation,
                elapsed_seconds=0.5,
            )
        else:
            real_mlx_module._ledger_event(
                ledger,
                sequence=sequence,
                replicate_id=replicate_id,
                event="postflight",
                status="not_started",
                reason=attempt["reason"],
                observation=observation,
            )
        sequence += 1
        real_mlx_module._ledger_event(
            ledger,
            sequence=sequence,
            replicate_id=replicate_id,
            event="finalized",
            status=attempt["status"],
            reason=attempt.get("reason"),
            elapsed_seconds=float(sequence),
            attempt_digest=real_mlx_module._directory_digest(attempts / replicate_id),
        )
        sequence += 1
    results = real_mlx_module._public_results_from_private(attempts)
    results_digest = None if failed_last else _digest_bytes(_json_bytes(results))
    real_mlx_module._finalize_run_ledger(
        ledger,
        sequence=sequence,
        complete_replicates=5 if failed_last else 6,
        results_digest=results_digest,
    )
    return workspace


def test_terminal_adapter_records_survive_replicate_and_aggregate_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal_codes = {
        "1k:cold-exact-duplicate:exact": "exact_empty_remainder_unsupported",
        "4k:suffix-only-change:variant": "quantized_cache_unsupported",
    }
    adapter_stub = SimpleNamespace(backend="mlx_lm_local")

    def capture_terminal_record(record: RequestEvidence) -> RequestEvidence:
        code = terminal_codes.get(record.spec.request_id)
        if code is None:
            return record
        return MLXLocalCacheAdapter._unsupported_record(
            cast(MLXLocalCacheAdapter, adapter_stub), record.spec, code=code
        )

    workspace = _make_attempts(
        tmp_path,
        monkeypatch,
        record_transform=capture_terminal_record,
    )
    captured_rows = real_mlx_module._parse_jsonl(
        workspace / "attempts" / "replicate-0" / "stages.jsonl"
    )
    assert [
        row["stage"]
        for row in captured_rows
        if row["request_id"] == "1k:cold-exact-duplicate:exact"
    ] == ["request_before_lookup", "request_after_lookup"]
    assert not any(
        row["request_id"] == "4k:suffix-only-change:variant" for row in captured_rows
    )
    for replicate_id in REPLICATE_IDS:
        assert (
            real_mlx_module.verify_replicate(
                workspace / "attempts" / replicate_id,
                replicate_id=replicate_id,
                public=False,
            )["status"]
            == "complete"
        )

    derivation_identity = {
        "expected_commit": "a" * 40,
        "generator_package_digest": real_mlx_module.package_source_digest(),
    }
    aggregate = tmp_path / "terminal-aggregate"
    assemble_aggregate(workspace, aggregate, **derivation_identity)
    assert (
        verify_aggregate(
            aggregate,
            expected_commit=derivation_identity["expected_commit"],
            generator_package_digest=derivation_identity["generator_package_digest"],
        )["verified"]
        is True
    )
    public = tmp_path / "terminal-public"
    sanitize_aggregate(
        aggregate,
        public,
        expected_commit=derivation_identity["expected_commit"],
        generator_package_digest=derivation_identity["generator_package_digest"],
    )
    assert (
        verify_aggregate(
            public,
            expected_commit=derivation_identity["expected_commit"],
            generator_package_digest=derivation_identity["generator_package_digest"],
        )["verified"]
        is True
    )
    _, public_records = cache_bundle.read_bundle(
        public / "replicates" / "replicate-0" / "bundle",
        data_only=True,
    )
    public_codes = {
        limitation.code
        for record in public_records
        for limitation in record.limitations
    }
    assert "exact_empty_remainder_unsupported" in public_codes
    assert "quantized_cache_unsupported" in public_codes
    assert "future_unsupported" not in public_codes
    results = json.loads((aggregate / "results.json").read_text())
    assert results["comparisons"]["1k:cold-exact"]["missing_pair_count"] == len(
        REPLICATE_IDS
    )
    assert results["comparisons"]["4k:suffix-only-change"]["missing_pair_count"] == len(
        REPLICATE_IDS
    )


def test_unknown_terminal_stage_bound_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_stub = SimpleNamespace(backend="mlx_lm_local")

    def capture_unknown_terminal(record: RequestEvidence) -> RequestEvidence:
        if record.spec.request_id != "1k:cold-exact-duplicate:exact":
            return record
        return MLXLocalCacheAdapter._unsupported_record(
            cast(MLXLocalCacheAdapter, adapter_stub),
            record.spec,
            code="future_unsupported",
        )

    workspace = _make_attempts(
        tmp_path,
        monkeypatch,
        record_transform=capture_unknown_terminal,
    )
    with pytest.raises(
        RealMLXExperimentError,
        match="terminal record has an unknown stage bound",
    ):
        real_mlx_module.verify_replicate(
            workspace / "attempts" / "replicate-0",
            replicate_id="replicate-0",
            public=False,
        )


def test_run_binding_rejects_zero_marker_digest() -> None:
    with pytest.raises(
        RealMLXExperimentError,
        match="canonical attempt marker digest is invalid",
    ):
        real_mlx_module._run_binding(
            expected_commit="a" * 40,
            package_digest=real_mlx_module.package_source_digest(),
            workload=_calibrated_workload(),
            runtime_packages=_runtime_packages(),
            run_attempt=1,
            attempt_marker_digest="sha256:" + "0" * 64,
        )


def test_aggregate_regeneration_checksums_and_public_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    derivation_identity = {
        "expected_commit": "a" * 40,
        "generator_package_digest": real_mlx_module.package_source_digest(),
    }
    private = tmp_path / "private"
    assert all(
        (
            workspace / "attempts" / replicate_id / "bundle" / "evidence_bundle.py"
        ).is_file()
        for replicate_id in REPLICATE_IDS
    )
    with pytest.raises(RealMLXExperimentError, match="run workspace allowlist"):
        assemble_aggregate(
            workspace / "attempts",
            tmp_path / "bare-attempts",
            **derivation_identity,
        )
    for mismatch in (
        {"expected_commit": "b" * 40},
        {"generator_package_digest": "sha256:" + "0" * 64},
    ):
        identity = {**derivation_identity, **mismatch}
        with pytest.raises(
            RealMLXExperimentError,
            match="run ledger derivation identity does not match trusted source",
        ):
            assemble_aggregate(
                workspace,
                tmp_path / "aggregate-mismatch",
                **identity,
            )
    assert (
        assemble_aggregate(workspace, private, **derivation_identity)[
            "complete_replicates"
        ]
        == 6
    )
    assert verify_aggregate(private, **derivation_identity)["verified"] is True
    for mismatch in (
        {"expected_commit": "b" * 40},
        {"generator_package_digest": "sha256:" + "0" * 64},
    ):
        with pytest.raises(
            RealMLXExperimentError,
            match="run ledger derivation identity does not match trusted source",
        ):
            verify_aggregate(private, **{**derivation_identity, **mismatch})
    results = json.loads((private / "results.json").read_text())
    assert results["delta_convention"] == "treatment_minus_control"
    assert results["ratio_convention"] == "treatment_divided_by_control"
    assert all(
        comparison["independent_units"] == 6
        and len(comparison["samples"]) == 6
        and comparison["comparable_pair_count"]
        == sum(sample["paired_latency_comparable"] for sample in comparison["samples"])
        for comparison in results["comparisons"].values()
    )
    assert all(
        "control_verdict" in sample
        and "control_engine_cached_tokens" in sample
        and "control_policy_reusable_tokens" in sample
        for comparison in results["comparisons"].values()
        for sample in comparison["samples"]
    )
    with pytest.raises(cache_bundle.CacheAuditBundleError, match="bundle files differ"):
        cache_bundle.verify_bundle(private / "replicates" / "replicate-0" / "bundle")
    private_index = json.loads((private / "replicate-index.json").read_text())
    assert private_index["lane_request_counts"] == {"1k": 108, "4k": 108}
    assert private_index["evidence_binding"]["expected_commit"] == "a" * 40
    assert (
        private_index["evidence_binding"]["generator_package_digest"]
        == derivation_identity["generator_package_digest"]
    )
    assert set(private_index["lane_scenario_counts"]) == set(LANE_IDS)
    assert private_index["replicates"][-1]["status"] == "complete"
    private_claims = json.loads((private / "claim-matrix.json").read_text())
    assert private_claims["lane_request_observations"] == {
        "1k": 108,
        "4k": 108,
    }
    assert private_claims["constant_target_distinguishes_workload_arrays"] is False
    assert (
        private_claims[
            "constant_target_evidence_against_cross_prefix_or_namespace_contamination"
        ]
        is False
    )
    assert (
        "not_independent_reuse_corroboration" in private_claims["constant_target_role"]
    )
    assert (
        private_claims["allocator_peak_scope"]
        == "process_global_allocator_peak_from_cache_fetch_through_generation"
    )
    assert private_claims["stage_instrumentation_in_client_clocks"] is False
    comparisons = json.loads((private / "descriptive-summary.json").read_text())[
        "comparisons"
    ]
    assert len(comparisons) == 14
    assert all(
        sample["lane_id"] == name.split(":", 1)[0]
        for name, comparison in comparisons.items()
        for sample in comparison["samples"]
    )
    assert all(
        sample["control_generated_output_tokens"] == 3
        and sample["treatment_generated_output_tokens"] == 3
        and "process_rss_difference_bytes" not in sample
        and "system_swap_difference_bytes" not in sample
        and "system_memory_free_difference_percentage_points" not in sample
        and "allocator_active_difference_bytes" not in sample
        and "allocator_cache_difference_bytes" not in sample
        for comparison in comparisons.values()
        for sample in comparison["samples"]
    )
    assert all(
        sample["paired_latency_comparable"] is True
        and sample["client_ttft_ratio"] == 1.0
        and sample["client_ttft_difference_seconds"] == 0.0
        for name, comparison in comparisons.items()
        if not name.endswith(":capacity-eviction")
        for sample in comparison["samples"]
    )
    assert all(
        sample["paired_latency_comparable"] is False
        and sample["client_ttft_difference_seconds"] is None
        and sample["client_ttft_ratio"] is None
        and sample["total_difference_seconds"] is None
        and sample["total_ratio"] is None
        and sample["allocator_peak_difference_bytes"] is None
        for name, comparison in comparisons.items()
        if name.endswith(":capacity-eviction")
        for sample in comparison["samples"]
    )
    contract = json.loads((private / "experiment-contract.json").read_text())
    assert set(contract["lanes"]) == set(LANE_IDS)
    assert len(contract["combined_blocks"]) == 14
    assert set(contract["exact_block_schedules"]) == set(REPLICATE_IDS)
    assert contract["run_attempt"] == 1
    assert contract["canonical_run_id"] == real_mlx_module.CANONICAL_RUN_ID
    assert (
        contract["canonical_output_workspace_digest"]
        == real_mlx_module._workspace_digest()
    )
    assert contract["canonical_attempt_marker"] == {
        "identity": real_mlx_module.CANONICAL_ATTEMPT_MARKER.name,
        "started_digest": "sha256:" + "1" * 64,
    }
    assert contract["prior_invalidated_run_ledger_digests"] == []
    assert contract["evidence_binding"]["expected_commit"] == "a" * 40
    assert (
        contract["evidence_binding"]["generator_package_digest"]
        == derivation_identity["generator_package_digest"]
    )
    assert contract["evidence_binding"]["run_attempt"] == 1
    assert contract["evidence_binding"]["prior_invalidated_run_ledger_digests"] == []
    assert contract["replicate_eligibility"] == {
        "attempted_replicates": 6,
        "required_complete_replicates": 6,
        "maximum_failed_replicates": 0,
    }
    assert contract["minimum_complete"] == 6
    assert (
        contract["child_temporary_directory"] == "private_output_workspace_filesystem"
    )
    assert (
        contract["integrity_and_authenticity"]["sha256sums"] == "integrity_only_unkeyed"
    )
    assert any(
        "non-causal scoped levels" in limitation
        for limitation in contract["limitations"]
    )
    assert any(
        "2-second monitoring" in limitation for limitation in contract["limitations"]
    )
    assert any("not causal speedups" in item for item in contract["limitations"])
    assert (private / "results.json").is_file()
    assert (private / "run-ledger.jsonl").is_file()
    assert not (private / "private-artifacts").exists()
    assert not (private / "aggregate_verifier.py").exists()
    assert not list(private.rglob("*.py"))
    assert all(
        path.stat().st_mode & 0o111 == 0
        for path in private.rglob("*")
        if path.is_file()
    )

    ledger_tamper = tmp_path / "ledger-tamper"
    shutil.copytree(private, ledger_tamper)
    ledger_path = ledger_tamper / "run-ledger.jsonl"
    ledger_rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    next(
        row
        for row in ledger_rows
        if row.get("event") == "finalized" and row.get("replicate_id") == "replicate-0"
    )["attempt_digest"] = ("sha256:" + "0" * 64)
    ledger_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in ledger_rows
        )
    )
    _write_recursive_checksums(ledger_tamper)
    with pytest.raises(RealMLXExperimentError, match="does not match attempt"):
        verify_aggregate(ledger_tamper, **derivation_identity)

    lane_tamper = tmp_path / "lane-tamper"
    shutil.copytree(private, lane_tamper)
    attempt = lane_tamper / "replicates" / "replicate-0" / "attempt.json"
    attempt_value = json.loads(attempt.read_text())
    attempt_value["lane_request_counts"] = {"1k": 17, "4k": 19}
    _write_json(attempt, attempt_value)
    _write_recursive_checksums(lane_tamper)
    with pytest.raises(
        RealMLXExperimentError,
        match="run ledger finalization does not match attempt",
    ):
        verify_aggregate(lane_tamper, **derivation_identity)

    summary = private / "summary.json"
    value = json.loads(summary.read_text())
    value["request_count"] += 1
    _write_json(summary, value)
    _write_recursive_checksums(private)
    with pytest.raises(RealMLXExperimentError, match="derived aggregate"):
        verify_aggregate(private, **derivation_identity)

    shutil.rmtree(private)
    assemble_aggregate(workspace, private, **derivation_identity)
    public = tmp_path / "public"
    for mismatch in (
        {"expected_commit": "b" * 40},
        {"generator_package_digest": "sha256:" + "0" * 64},
    ):
        with pytest.raises(
            RealMLXExperimentError,
            match="run ledger derivation identity does not match trusted source",
        ):
            sanitize_aggregate(
                private,
                tmp_path / "sanitize-mismatch",
                **{**derivation_identity, **mismatch},
            )
    assert (
        sanitize_aggregate(private, public, **derivation_identity)["publication_mode"]
        == "public_redacted"
    )
    public_index = json.loads((public / "replicate-index.json").read_text())
    assert public_index["lane_request_counts"] == {"1k": 108, "4k": 108}
    assert public_index["evidence_binding"]["expected_commit"] == "a" * 40
    assert (
        public_index["evidence_binding"]["generator_package_digest"]
        == derivation_identity["generator_package_digest"]
    )
    assert public_index["evidence_binding"]["run_attempt"] == 1
    assert (
        public_index["evidence_binding"]["prior_invalidated_run_ledger_digests"] == []
    )
    assert public_index["evidence_binding"]["generator_package_digest"].startswith(
        "sha256:"
    )
    public_contract = json.loads((public / "experiment-contract.json").read_text())
    assert public_contract["evidence_binding"]["expected_commit"] == "a" * 40
    assert (
        public_contract["evidence_binding"]["generator_package_digest"]
        == derivation_identity["generator_package_digest"]
    )
    public_results = json.loads((public / "results.json").read_text())
    assert public_results["source"] == "verified_private_records_before_redaction"
    assert all(
        sample["control_output_token_identity"] is True
        and sample["treatment_output_token_identity"] is True
        and sample["control_deterministic_correctness"] is True
        and sample["treatment_deterministic_correctness"] is True
        and sample["verdict"] is not None
        for comparison in public_results["comparisons"].values()
        for sample in comparison["samples"]
    )
    public_ledger = [
        json.loads(line)
        for line in (public / "run-ledger.jsonl").read_text().splitlines()
    ]
    assert {row["event"] for row in public_ledger} == {
        "run-start",
        "planned",
        "preflight",
        "started",
        "postflight",
        "finalized",
        "run-finalized",
    }
    assert any("observation" in row for row in public_ledger)
    public_run_start = next(row for row in public_ledger if row["event"] == "run-start")
    assert public_run_start["run_attempt"] == 1
    assert public_run_start["canonical_run_id"] == real_mlx_module.CANONICAL_RUN_ID
    assert public_run_start["canonical_attempt_marker_digest"] == "sha256:" + "1" * 64
    assert public_run_start["prior_invalidated_run_ledger_digests"] == []
    public_workload_binding = json.loads(
        (public / "replicates" / "replicate-0" / "workload-binding.json").read_text()
    )
    assert public_workload_binding["run_attempt"] == 1
    assert public_workload_binding["prior_invalidated_run_ledger_digests"] == []
    assert not (public / "aggregate_verifier.py").exists()
    assert not list(public.rglob("*.py"))
    assert all(
        path.stat().st_mode & 0o111 == 0 for path in public.rglob("*") if path.is_file()
    )
    assert "run_instance_id" not in json.loads(
        (public / "environment.json").read_text()
    )
    assert all(
        "run_instance_id"
        not in json.loads(
            (public / "replicates" / replicate_id / "environment.json").read_text()
        )
        for replicate_id in REPLICATE_IDS[:5]
    )
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in public.rglob("*") if path.is_file()
    )
    assert '"input_token_ids":[' not in text
    assert 'request-0"' not in text

    results_tamper = tmp_path / "results-tamper"
    shutil.copytree(public, results_tamper)
    results_path = results_tamper / "results.json"
    results_value = json.loads(results_path.read_text())
    results_value["comparisons"]["1k:cold-exact"]["samples"][0][
        "treatment_deterministic_correctness"
    ] = False
    _write_json(results_path, results_value)
    _write_recursive_checksums(results_tamper)
    with pytest.raises(RealMLXExperimentError, match="public result"):
        verify_aggregate(results_tamper, **derivation_identity)

    sample_count_tamper = tmp_path / "sample-count-tamper"
    shutil.copytree(public, sample_count_tamper)
    sample_count_results = sample_count_tamper / "results.json"
    sample_count_value = json.loads(sample_count_results.read_text())
    sample_count_value["comparisons"]["1k:cold-exact"]["samples"].pop()
    sample_count_value["comparisons"]["1k:cold-exact"]["independent_units"] = 5
    _write_json(sample_count_results, sample_count_value)
    _write_recursive_checksums(sample_count_tamper)
    with pytest.raises(RealMLXExperimentError, match="sample count"):
        verify_aggregate(sample_count_tamper, **derivation_identity)

    memory_tamper = tmp_path / "memory-tamper"
    shutil.copytree(public, memory_tamper)
    memory_results = memory_tamper / "results.json"
    memory_value = json.loads(memory_results.read_text())
    memory_value["comparisons"]["1k:cold-exact"]["samples"][0][
        "treatment_system_memory"
    ]["free_percent"] = 101.0
    _write_json(memory_results, memory_value)
    _write_recursive_checksums(memory_tamper)
    with pytest.raises(RealMLXExperimentError, match="system-memory"):
        verify_aggregate(memory_tamper, **derivation_identity)

    control_memory_tamper = tmp_path / "control-memory-tamper"
    shutil.copytree(public, control_memory_tamper)
    control_memory_results = control_memory_tamper / "results.json"
    control_memory_value = json.loads(control_memory_results.read_text())
    control_memory_value["comparisons"]["1k:cold-exact"]["samples"][0][
        "control_system_memory"
    ]["free_percent"] = -1.0
    _write_json(control_memory_results, control_memory_value)
    _write_recursive_checksums(control_memory_tamper)
    with pytest.raises(RealMLXExperimentError, match="system-memory"):
        verify_aggregate(control_memory_tamper, **derivation_identity)

    nested_tamper = tmp_path / "nested-tamper"
    shutil.copytree(public, nested_tamper)
    nested_results = nested_tamper / "results.json"
    nested_value = json.loads(nested_results.read_text())
    nested_value["comparisons"]["1k:cold-exact"]["samples"][0]["control_process_rss"][
        "bytes"
    ] += 1
    _write_json(nested_results, nested_value)
    nested_digest = _digest_bytes(_json_bytes(nested_value))
    nested_ledger = nested_tamper / "run-ledger.jsonl"
    nested_ledger_rows = [
        json.loads(line) for line in nested_ledger.read_text().splitlines()
    ]
    nested_ledger_rows[-1]["results_digest"] = nested_digest
    nested_ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in nested_ledger_rows
        ),
        encoding="ascii",
    )
    nested_contract = nested_tamper / "experiment-contract.json"
    nested_contract_value = json.loads(nested_contract.read_text())
    nested_contract_value["results_digest"] = nested_digest
    _write_json(nested_contract, nested_contract_value)
    _write_recursive_checksums(nested_tamper)
    with pytest.raises(RealMLXExperimentError, match="retained nested evidence"):
        verify_aggregate(nested_tamper, **derivation_identity)

    script_tamper = tmp_path / "script-tamper"
    shutil.copytree(public, script_tamper)
    script = script_tamper / "replicates" / "replicate-0" / "bundle" / "unexpected.py"
    script.write_text("print('not data-only')\n", encoding="ascii")
    _write_recursive_checksums(script_tamper)
    with pytest.raises(RealMLXExperimentError, match="script-like"):
        verify_aggregate(script_tamper, **derivation_identity)

    executable_tamper = tmp_path / "executable-tamper"
    shutil.copytree(public, executable_tamper)
    executable = executable_tamper / "report.html"
    executable.chmod(executable.stat().st_mode | 0o100)
    with pytest.raises(RealMLXExperimentError, match="executable"):
        verify_aggregate(executable_tamper, **derivation_identity)

    stage_tamper = tmp_path / "stage-tamper"
    shutil.copytree(public, stage_tamper)
    stages = stage_tamper / "replicates" / "replicate-0" / "stages.jsonl"
    rows = stages.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    stages.write_text("\n".join(rows) + "\n", encoding="ascii")
    _write_recursive_checksums(stage_tamper)
    with pytest.raises(
        RealMLXExperimentError,
        match="run ledger finalization does not match attempt",
    ):
        verify_aggregate(stage_tamper, **derivation_identity)
    rows[0], rows[1] = rows[1], rows[0]
    stages.write_text("\n".join(rows) + "\n", encoding="ascii")
    real_mlx_module._write_sanitized_run_ledger(
        private / "run-ledger.jsonl",
        stage_tamper / "run-ledger.jsonl",
        attempts_dir=stage_tamper / "replicates",
    )
    _write_recursive_checksums(stage_tamper)

    descriptive = stage_tamper / "descriptive-summary.json"
    descriptive_value = json.loads(descriptive.read_text())
    descriptive_value["paired_comparisons"] = {}
    _write_json(descriptive, descriptive_value)
    _write_recursive_checksums(stage_tamper)
    with pytest.raises(RealMLXExperimentError, match="derived aggregate"):
        verify_aggregate(stage_tamper, **derivation_identity)


def test_run_ledger_rejects_minimal_finalization_only_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    rows = [
        row
        for row in (json.loads(line) for line in ledger.read_text().splitlines())
        if row["event"] in {"run-start", "planned", "finalized", "run-finalized"}
    ]
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )

    with pytest.raises(RealMLXExperimentError, match="incomplete"):
        real_mlx_module._verify_run_ledger(
            ledger,
            attempts_dir=workspace / "attempts",
        )


def test_run_ledger_rejects_lifecycle_schema_and_instance_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    original = ledger.read_text(encoding="ascii")

    rows = [json.loads(line) for line in original.splitlines()]
    started = [row for row in rows if row["event"] == "started"]
    started[1]["child_instance_id_digest"] = started[0]["child_instance_id_digest"]
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )
    with pytest.raises(RealMLXExperimentError, match="child instance binding"):
        real_mlx_module._verify_run_ledger(ledger)

    rows = [json.loads(line) for line in original.splitlines()]
    preflight = next(row for row in rows if row["event"] == "preflight")
    preflight["observation"]["unexpected"] = True
    ledger.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )
    with pytest.raises(RealMLXExperimentError, match="field allowlist"):
        real_mlx_module._verify_run_ledger(ledger)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_attempt", 2),
        ("prior_invalidated_run_ledger_digests", ["sha256:" + "0" * 64]),
    ],
)
def test_run_ledger_rejects_whole_run_retry_binding_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="ascii").splitlines()
    ]
    rows[0][field] = value
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )

    with pytest.raises(RealMLXExperimentError, match="run ledger binding"):
        real_mlx_module._verify_run_ledger(ledger)


def test_run_ledger_enforces_monitor_and_preserves_pre_postflight_terminal_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    original = [
        json.loads(line) for line in ledger.read_text(encoding="ascii").splitlines()
    ]

    rows = [dict(row) for row in original]
    postflight_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("replicate_id") == "replicate-0" and row["event"] == "postflight"
    )
    failed_observation = dict(rows[postflight_index]["observation"])
    failed_observation["vm_stat_available_ratio"] = 0.1
    rows.insert(
        postflight_index,
        {
            "schema_version": "2",
            "sequence": -1,
            "timestamp": rows[postflight_index]["timestamp"],
            "replicate_id": "replicate-0",
            "event": "monitor",
            "status": "failed",
            "reason": "runtime_memory_below_floor",
            "observation": failed_observation,
            "elapsed_seconds": 2.0,
        },
    )
    rows[postflight_index + 1]["elapsed_seconds"] = 2.5
    finalized = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "finalized"
    )
    finalized["status"] = "failed"
    finalized["reason"] = "child_exit_nonzero"
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )
    with pytest.raises(RealMLXExperimentError, match="contradicts failed policy"):
        real_mlx_module._verify_run_ledger(ledger)

    rows = [dict(row) for row in original]
    postflight = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "postflight"
    )
    postflight["observation"] = dict(postflight["observation"])
    postflight["observation"]["vm_stat_available_ratio"] = 0.1
    postflight["status"] = "failed"
    postflight["reason"] = "runtime_memory_below_floor"
    finalized = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "finalized"
    )
    finalized["status"] = "failed"
    finalized["reason"] = "child_exit_nonzero"
    run_finalized = rows[-1]
    run_finalized["status"] = "insufficient_complete_replicates"
    run_finalized["reason"] = "insufficient_complete_replicates"
    run_finalized["complete_replicates"] = 5
    run_finalized["results_digest"] = None
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )
    real_mlx_module._verify_run_ledger(ledger)


def test_run_ledger_requires_started_rows_and_abort_ordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    original = ledger.read_text(encoding="ascii")

    rows = [
        row
        for row in (json.loads(line) for line in original.splitlines())
        if not (row.get("replicate_id") == "replicate-0" and row["event"] == "started")
    ]
    postflight = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "postflight"
    )
    postflight["status"] = "not_started"
    postflight["reason"] = "child_timeout"
    postflight.pop("elapsed_seconds")
    finalized = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "finalized"
    )
    finalized["status"] = "failed"
    finalized["reason"] = "child_timeout"
    for sequence, row in enumerate(rows):
        row["sequence"] = sequence
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )
    with pytest.raises(RealMLXExperimentError, match="not-started terminal"):
        real_mlx_module._verify_run_ledger(ledger)

    rows = [json.loads(line) for line in original.splitlines()]
    finalized = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-0" and row["event"] == "finalized"
    )
    finalized["status"] = "failed"
    finalized["reason"] = "source_validation_failed"
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )
    with pytest.raises(RealMLXExperimentError, match="continued after an aborting"):
        real_mlx_module._verify_run_ledger(ledger)


def test_run_ledger_rejects_supervisor_abort_without_earlier_aborting_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    ledger = workspace / "run-ledger.jsonl"
    rows = [
        json.loads(line) for line in ledger.read_text(encoding="ascii").splitlines()
    ]
    for row in rows:
        if row.get("replicate_id") == "replicate-5" and row["event"] in {
            "preflight",
            "postflight",
            "finalized",
        }:
            row["reason"] = "supervisor_aborted_before_start"
    ledger.write_text(
        "".join(real_mlx_module._json_line(row) for row in rows),
        encoding="ascii",
    )

    with pytest.raises(
        RealMLXExperimentError,
        match="external preflight state|lacks an earlier aborting",
    ):
        real_mlx_module._verify_run_ledger(ledger)


def test_all_six_eligibility_rejects_preflight_and_started_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch, failed_last=True)
    ledger = workspace / "run-ledger.jsonl"
    original = [
        json.loads(line) for line in ledger.read_text(encoding="ascii").splitlines()
    ]

    def write_rows(rows: list[dict[str, Any]]) -> None:
        for sequence, row in enumerate(rows):
            row["sequence"] = sequence
        ledger.write_text(
            "".join(real_mlx_module._json_line(row) for row in rows),
            encoding="ascii",
        )

    non_preflight = [dict(row) for row in original[:-1]]
    preflight = next(
        row
        for row in non_preflight
        if row.get("replicate_id") == "replicate-5" and row["event"] == "preflight"
    )
    preflight["status"] = "passed"
    preflight["reason"] = None
    preflight["observation"] = {
        **preflight["observation"],
        "heavy_process_count": 0,
        "heavy_process_categories": [],
    }
    for row in non_preflight:
        if row.get("replicate_id") == "replicate-5" and row["event"] in {
            "postflight",
            "finalized",
        }:
            row["reason"] = "child_launch_failed"
    write_rows(non_preflight)
    real_mlx_module._finalize_run_ledger(
        ledger,
        sequence=len(non_preflight),
        complete_replicates=5,
        results_digest=None,
    )
    finalized = json.loads(ledger.read_text(encoding="ascii").splitlines()[-1])
    assert finalized["status"] == "insufficient_complete_replicates"
    assert finalized["reason"] == "insufficient_complete_replicates"
    real_mlx_module._verify_run_ledger(ledger)

    rows = [dict(row) for row in original[:-1]]
    preflight = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-5" and row["event"] == "preflight"
    )
    preflight["status"] = "passed"
    preflight["reason"] = None
    preflight["observation"] = {
        **preflight["observation"],
        "heavy_process_count": 0,
        "heavy_process_categories": [],
    }
    postflight_index = next(
        index
        for index, row in enumerate(rows)
        if row.get("replicate_id") == "replicate-5" and row["event"] == "postflight"
    )
    rows.insert(
        postflight_index,
        {
            "schema_version": "2",
            "sequence": -1,
            "timestamp": rows[postflight_index]["timestamp"],
            "replicate_id": "replicate-5",
            "event": "started",
            "status": "running",
            "reason": None,
            "child_instance_id_digest": "sha256:" + "f" * 64,
        },
    )
    postflight = rows[postflight_index + 1]
    postflight["status"] = "passed"
    postflight["reason"] = None
    postflight["elapsed_seconds"] = 0.5
    failed = next(
        row
        for row in rows
        if row.get("replicate_id") == "replicate-5" and row["event"] == "finalized"
    )
    failed["reason"] = "child_exit_nonzero"
    write_rows(rows)
    real_mlx_module._finalize_run_ledger(
        ledger,
        sequence=len(rows),
        complete_replicates=5,
        results_digest=None,
    )
    finalized = json.loads(ledger.read_text(encoding="ascii").splitlines()[-1])
    assert finalized["status"] == "insufficient_complete_replicates"
    real_mlx_module._verify_run_ledger(ledger)

    synthetic = [
        {
            "event": "finalized",
            "replicate_id": replicate_id,
            "status": "complete",
            "reason": None,
        }
        for replicate_id in REPLICATE_IDS[:5]
    ]
    synthetic.append(
        {
            "event": "finalized",
            "replicate_id": "replicate-5",
            "status": "failed",
            "reason": "supervisor_aborted_before_start",
        }
    )
    assert not real_mlx_module._aggregate_eligibility_from_rows(synthetic)


def test_aggregate_requires_all_six_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    attempts = workspace / "attempts"
    for replicate_id in ("replicate-3", "replicate-4"):
        shutil.rmtree(attempts / replicate_id)
        replicate = attempts / replicate_id
        replicate.mkdir()
        _write_json(
            replicate / "attempt.json",
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "failed",
                "replacement": False,
                "reason": "synthetic_failure",
                "failed_at": "2026-01-01T00:00:00.000000Z",
            },
        )
        _write_json(
            replicate / "teardown.json",
            {
                "schema_version": "1",
                "cache_lifecycles_released": None,
                "garbage_collection_completed": None,
                "mlx_allocator_cleanup": "externally_terminated",
                "allocator_cache_bytes_after": None,
                "scope": "current_replicate_child_process",
            },
        )
    with pytest.raises(RealMLXExperimentError, match="all 6"):
        assemble_aggregate(
            workspace,
            tmp_path / "aggregate",
            expected_commit="a" * 40,
            generator_package_digest=real_mlx_module.package_source_digest(),
        )

    (attempts / "replicate-5").rename(attempts / "replicate-6")
    with pytest.raises(RealMLXExperimentError, match="replicate-0..5 exactly"):
        assemble_aggregate(
            workspace,
            tmp_path / "replacement",
            expected_commit="a" * 40,
            generator_package_digest=real_mlx_module.package_source_digest(),
        )


@pytest.mark.parametrize("results_failure", [None, "derivation", "verification"])
def test_run_all_uses_fresh_offline_sandboxed_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    results_failure: str | None,
) -> None:
    launches: list[dict[str, Any]] = []
    observed_instances: set[str] = set()

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.pid = 10_000 + len(launches)
            self.returncode = 0
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            _write_json(output / "attempt.json", {"status": "complete"})
            instance = kwargs["env"][real_mlx_module.RUN_INSTANCE_ENV]
            assert instance not in observed_instances
            observed_instances.add(instance)
            launches.append({"command": command, **kwargs})

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    safe_observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion_summary = tmp_path / "summary.json"
    conversion_summary.write_text("{}")
    calibrated = _calibrated_workload()
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        calibrated.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in calibrated.lanes},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    source_checks = 0

    def validate_source(_: str) -> str:
        nonlocal source_checks
        source_checks += 1
        return "sha256:" + "b" * 64

    monkeypatch.setattr(real_mlx_module, "_validate_supervisor_source", validate_source)
    monkeypatch.setattr(
        real_mlx_module,
        "load_workload",
        lambda *_args, **_kwargs: calibrated,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "verify_model_contract",
        lambda *_args, **_kwargs: EXPECTED_MODEL_ARTIFACT_DIGEST,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_run_exact_sandbox_probe",
        lambda **_kwargs: _runtime_packages(),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_runtime_package_identity",
        lambda *_args, **_kwargs: _runtime_packages(),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: safe_observation,
    )
    monkeypatch.setattr(real_mlx_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(real_mlx_module, "_group_exists", lambda _: False)
    monkeypatch.setattr(real_mlx_module, "verify_replicate", lambda *_a, **_k: {})

    def derive_results(*_args: Any) -> dict[str, bool]:
        if results_failure == "derivation":
            raise RealMLXExperimentError("synthetic results derivation failure")
        return {"synthetic": True}

    monkeypatch.setattr(real_mlx_module, "_public_results_from_private", derive_results)

    def verify_results(*_args: Any, **_kwargs: Any) -> None:
        if results_failure == "verification":
            raise RealMLXExperimentError("synthetic results verification failure")

    monkeypatch.setattr(
        real_mlx_module,
        "_verify_public_results",
        verify_results,
    )

    workspace = tmp_path / "run"
    result = real_mlx_module.run_all_replicates(
        workload=workload_path,
        model_dir=model_dir,
        conversion_summary=conversion_summary,
        output_workspace=workspace,
        expected_commit="a" * 40,
        run_attempt=1,
    )

    assert result == {
        "run_all_complete": results_failure is None,
        "attempted_replicates": 6,
        "complete_replicates": 6,
        "failed_replicates": 0,
        "aggregate_created": False,
        "sanitized": False,
    }
    assert len(launches) == 6
    assert source_checks == 13
    assert len(observed_instances) == 6
    assert all(launch["start_new_session"] is True for launch in launches)
    assert all(launch["shell"] is False for launch in launches)
    assert all(launch["cwd"] == tmp_path for launch in launches)
    assert all("OPENAI_API_KEY" not in launch["env"] for launch in launches)
    assert all(launch["env"]["HF_HUB_OFFLINE"] == "1" for launch in launches)
    assert all(launch["env"]["PYTHONSAFEPATH"] == "1" for launch in launches)
    assert all("-I" in launch["command"] for launch in launches)
    assert all("-S" in launch["command"] for launch in launches)
    assert all(Path(launch["env"]["TMPDIR"]).parent == workspace for launch in launches)
    assert all(not Path(launch["env"]["TMPDIR"]).exists() for launch in launches)
    assert all(
        launch["env"][real_mlx_module.RUN_INSTANCE_ENV] in observed_instances
        for launch in launches
    )
    assert [
        launch["command"][launch["command"].index("--replicate-id") + 1]
        for launch in launches
    ] == list(REPLICATE_IDS)
    assert all(
        launch["command"][launch["command"].index("--expected-commit") + 1] == "a" * 40
        for launch in launches
    )
    assert all(
        Path(
            launch["command"][launch["command"].index("--output-dir") + 1]
        ).is_absolute()
        for launch in launches
    )
    assert all(
        Path(launch["command"][launch["command"].index(flag) + 1]).is_absolute()
        for launch in launches
        for flag in ("--workload", "--model-dir", "--conversion-summary")
    )
    assert {path.name for path in (workspace / "attempts").iterdir()} == set(
        REPLICATE_IDS
    )
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    assert ledger[0]["event"] == "run-start"
    assert ledger[0]["expected_commit"] == "a" * 40
    assert [row["replicate_id"] for row in ledger[1:7]] == list(REPLICATE_IDS)
    assert all(row["event"] == "planned" for row in ledger[1:7])
    assert [
        row["replicate_id"] for row in ledger if row["event"] == "finalized"
    ] == list(REPLICATE_IDS)
    assert not any("pid" in row or "path" in row for row in ledger)
    assert ledger[-1]["status"] == (
        "results_derivation_failed" if results_failure else "aggregate_eligible"
    )
    assert ledger[-1]["reason"] == (
        "results_derivation_failed" if results_failure else None
    )
    assert (ledger[-1]["results_digest"] is None) is (results_failure is not None)


def test_run_all_global_machine_failure_creates_no_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.24,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion_summary = tmp_path / "summary.json"
    conversion_summary.write_text("{}")
    calibrated = _calibrated_workload()
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        calibrated.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in calibrated.lanes},
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "load_workload",
        lambda *_args, **_kwargs: calibrated,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "verify_model_contract",
        lambda *_args, **_kwargs: EXPECTED_MODEL_ARTIFACT_DIGEST,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_run_exact_sandbox_probe",
        lambda **_kwargs: _runtime_packages(),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_runtime_package_identity",
        lambda *_args, **_kwargs: _runtime_packages(),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: blocked,
    )

    workspace = tmp_path / "blocked"
    monkeypatch.setattr(real_mlx_module, "CANONICAL_OUTPUT_WORKSPACE", workspace)
    with pytest.raises(
        RealMLXExperimentError,
        match="NEEDS_CLEAN_BOOT:preflight_memory_below_floor",
    ):
        real_mlx_module.run_all_replicates(
            workload=workload_path,
            model_dir=model_dir,
            conversion_summary=conversion_summary,
            output_workspace=workspace,
            expected_commit="a" * 40,
            run_attempt=1,
        )

    assert not workspace.exists()


def test_global_preflight_rejects_parent_and_sandbox_runtime_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion_summary = tmp_path / "summary.json"
    conversion_summary.write_text("{}")
    output_workspace = tmp_path / "output" / "run"
    output_workspace.parent.mkdir()
    monkeypatch.setattr(real_mlx_module, "CANONICAL_OUTPUT_WORKSPACE", output_workspace)
    parent_identity = _runtime_packages()
    sandbox_identity = _runtime_packages()
    sandbox_identity["mlx"]["tree_sha256"] = "sha256:" + "f" * 64

    monkeypatch.setattr(real_mlx_module, "_ensure_non_repository_cwd", lambda _p: None)
    monkeypatch.setattr(real_mlx_module, "_reject_import_shadows", lambda _p: None)
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _commit: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        real_mlx_module, "load_workload", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        real_mlx_module,
        "verify_model_contract",
        lambda *_args, **_kwargs: EXPECTED_MODEL_ARTIFACT_DIGEST,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_runtime_package_identity",
        lambda _output: parent_identity,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_run_exact_sandbox_probe",
        lambda **_kwargs: sandbox_identity,
    )

    with pytest.raises(RealMLXExperimentError, match="does not match parent"):
        real_mlx_module._run_global_preflight(
            workload=workload_path,
            model_dir=model_dir,
            conversion_summary=conversion_summary,
            output_workspace=output_workspace,
            expected_commit="a" * 40,
        )


def test_run_all_bundle_source_failure_finalizes_every_planned_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    summary = tmp_path / "summary.json"
    summary.write_text("{}")
    workspace = tmp_path / "run"
    workload = _calibrated_workload()
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        workload.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes},
    )
    safe_observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }
    monkeypatch.setattr(
        real_mlx_module,
        "_run_global_preflight",
        lambda **_kwargs: real_mlx_module._RunPreflight(
            workload_path=workload_path,
            model_dir=model_dir,
            conversion_summary=summary,
            output_workspace=workspace,
            package_digest="sha256:" + "b" * 64,
            workload=workload,
            runtime_packages=_runtime_packages(),
            machine_observation=safe_observation,
        ),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: safe_observation,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _commit: (_ for _ in ()).throw(
            cache_bundle.CacheAuditBundleError("digest failed")
        ),
    )

    result = real_mlx_module.run_all_replicates(
        workload=workload_path,
        model_dir=model_dir,
        conversion_summary=summary,
        output_workspace=workspace,
        expected_commit="a" * 40,
        run_attempt=1,
    )

    assert result["complete_replicates"] == 0
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    finalized = [row for row in ledger if row["event"] == "finalized"]
    assert len(finalized) == len(REPLICATE_IDS)
    assert finalized[0]["reason"] == "source_validation_failed"
    assert all(
        row["reason"] == "supervisor_aborted_before_start" for row in finalized[1:]
    )


def test_run_all_preserves_terminal_reason_when_postflight_policy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion_summary = tmp_path / "summary.json"
    conversion_summary.write_text("{}")
    workspace = tmp_path / "run"
    workload = _calibrated_workload()
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        workload.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes},
    )
    safe_observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }
    failed_observation = {**safe_observation, "vm_stat_available_ratio": 0.1}
    monkeypatch.setattr(
        real_mlx_module,
        "_run_global_preflight",
        lambda **_kwargs: real_mlx_module._RunPreflight(
            workload_path=workload_path,
            model_dir=model_dir,
            conversion_summary=conversion_summary,
            output_workspace=workspace,
            package_digest="sha256:" + "b" * 64,
            workload=workload,
            runtime_packages=_runtime_packages(),
            machine_observation=safe_observation,
        ),
    )
    observations = 0

    def observe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal observations
        observations += 1
        return safe_observation if observations % 2 == 1 else failed_observation

    monkeypatch.setattr(real_mlx_module, "_machine_observation", observe)
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _commit: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        real_mlx_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )

    result = real_mlx_module.run_all_replicates(
        workload=workload_path,
        model_dir=model_dir,
        conversion_summary=conversion_summary,
        output_workspace=workspace,
        expected_commit="a" * 40,
        run_attempt=1,
    )

    assert result["run_all_complete"] is False
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    postflight = [row for row in ledger if row["event"] == "postflight"]
    finalized = [row for row in ledger if row["event"] == "finalized"]
    assert all(
        row["status"] == "failed" and row["reason"] == "runtime_memory_below_floor"
        for row in postflight
    )
    assert all(row["reason"] == "child_launch_failed" for row in finalized)


def test_run_all_cli_exits_nonzero_when_results_derivation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        real_mlx_module,
        "_run_cli",
        lambda _args: {
            "run_all_complete": False,
            "complete_replicates": 6,
        },
    )

    with pytest.raises(SystemExit) as raised:
        real_mlx_module.main(
            [
                "run-all",
                "--workload",
                "workload.json",
                "--model-dir",
                "model",
                "--output-workspace",
                "run",
                "--expected-commit",
                "a" * 40,
                "--run-attempt",
                "1",
            ]
        )

    assert raised.value.code == 1


@pytest.mark.parametrize(
    ("arguments", "target"),
    [
        (
            [
                "aggregate",
                "--run-workspace",
                "run",
                "--output-dir",
                "aggregate",
                "--expected-commit",
                "a" * 40,
            ],
            "assemble_aggregate",
        ),
        (
            [
                "sanitize",
                "aggregate",
                "--output-dir",
                "public",
                "--expected-commit",
                "a" * 40,
            ],
            "sanitize_aggregate",
        ),
        (
            ["verify", "aggregate", "--expected-commit", "a" * 40],
            "verify_aggregate",
        ),
    ],
)
def test_derivation_cli_dispatch_preserves_trusted_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    target: str,
) -> None:
    package_digest = "sha256:" + "b" * 64
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        real_mlx_module,
        "_require_trusted_canonical_dispatch",
        lambda expected_commit: (
            package_digest if expected_commit == "a" * 40 else pytest.fail()
        ),
    )
    monkeypatch.setattr(
        real_mlx_module,
        target,
        lambda *_args, **kwargs: calls.append(kwargs) or {"verified": True},
    )

    real_mlx_module._run_cli(real_mlx_module._parser().parse_args(arguments))

    assert calls == [
        {
            "expected_commit": "a" * 40,
            "generator_package_digest": package_digest,
        }
    ]


@pytest.mark.parametrize("command", ["preflight", "run-all"])
@pytest.mark.parametrize("attempt", [None, "0", "2", "1.0"])
def test_canonical_entrypoints_accept_only_explicit_integer_attempt_one(
    command: str,
    attempt: str | None,
) -> None:
    arguments = [
        command,
        "--workload",
        "workload.json",
        "--model-dir",
        "model",
        "--output-workspace",
        "run",
        "--expected-commit",
        "a" * 40,
    ]
    if command == "preflight":
        arguments.extend(("--output", "preflight.json"))
    if attempt is not None:
        arguments.extend(("--run-attempt", attempt))

    with pytest.raises(SystemExit) as raised:
        real_mlx_module._parser().parse_args(arguments)

    assert raised.value.code == 2


@pytest.mark.parametrize("attempt", [2, 1.0, True])
def test_direct_preflight_rejects_noncanonical_attempt_before_writing(
    tmp_path: Path, attempt: object
) -> None:
    receipt = tmp_path / "preflight.json"
    with pytest.raises(RealMLXExperimentError, match="accepts only run attempt 1"):
        real_mlx_module.run_preflight(
            workload=tmp_path / "workload.json",
            model_dir=tmp_path / "model",
            conversion_summary=tmp_path / "summary.json",
            output_workspace=tmp_path / "run",
            expected_commit="a" * 40,
            output=receipt,
            run_attempt=attempt,  # type: ignore[arg-type]
        )

    assert not receipt.exists()


def test_canonical_workspace_is_exact_and_consumed_marker_is_durable(
    tmp_path: Path,
) -> None:
    workload = tmp_path / "workload.json"
    model = tmp_path / "model"
    summary = tmp_path / "summary.json"
    workload.write_text("{}")
    model.mkdir()
    summary.write_text("{}")
    with pytest.raises(RealMLXExperimentError, match="exact canonical"):
        real_mlx_module._resolve_run_all_paths(
            workload=workload,
            model_dir=model,
            conversion_summary=summary,
            output_workspace=tmp_path / "wrong-run",
        )

    started, digest = real_mlx_module._consume_canonical_attempt(
        expected_commit="a" * 40,
        workload=_calibrated_workload(),
    )
    marker = real_mlx_module.CANONICAL_ATTEMPT_MARKER
    inode = marker.stat().st_ino
    with pytest.raises(RealMLXExperimentError, match="already consumed"):
        real_mlx_module._consume_canonical_attempt(
            expected_commit="a" * 40,
            workload=_calibrated_workload(),
        )
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"status":"complete"}\n', encoding="ascii")
    real_mlx_module._finalize_canonical_attempt_marker(
        started,
        digest,
        status="completed",
        ledger=ledger,
    )
    terminal = json.loads(marker.read_text())
    assert marker.stat().st_ino == inode
    assert terminal["status"] == "completed"
    assert terminal["started_marker_digest"] == digest
    assert terminal["run_ledger_digest"] == _digest_bytes(ledger.read_bytes())


def test_preflight_does_not_consume_marker_and_run_consumes_after_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = _calibrated_workload()
    state = real_mlx_module._RunPreflight(
        workload_path=tmp_path / "workload.json",
        model_dir=tmp_path / "model",
        conversion_summary=tmp_path / "summary.json",
        output_workspace=real_mlx_module.CANONICAL_OUTPUT_WORKSPACE,
        package_digest="sha256:" + "b" * 64,
        workload=workload,
        runtime_packages=_runtime_packages(),
        machine_observation={},
    )
    monkeypatch.setattr(real_mlx_module, "_run_global_preflight", lambda **_: state)
    receipt = tmp_path / "preflight.json"
    result = real_mlx_module.run_preflight(
        workload=state.workload_path,
        model_dir=state.model_dir,
        conversion_summary=state.conversion_summary,
        output_workspace=state.output_workspace,
        expected_commit="a" * 40,
        output=receipt,
        run_attempt=1,
    )
    assert result["preflight_passed"] is True
    assert not real_mlx_module.CANONICAL_ATTEMPT_MARKER.exists()

    def execute(**_: Any) -> dict[str, Any]:
        assert real_mlx_module.CANONICAL_ATTEMPT_MARKER.is_file()
        assert not state.output_workspace.exists()
        return {"run_all_complete": False}

    monkeypatch.setattr(
        real_mlx_module, "_run_all_replicates_after_consumption", execute
    )
    real_mlx_module.run_all_replicates(
        workload=state.workload_path,
        model_dir=state.model_dir,
        conversion_summary=state.conversion_summary,
        output_workspace=state.output_workspace,
        expected_commit="a" * 40,
        run_attempt=1,
    )
    marker = json.loads(real_mlx_module.CANONICAL_ATTEMPT_MARKER.read_text())
    assert marker["status"] == "failed"
    assert marker["run_ledger_digest"] is None


def test_run_all_marker_finalization_failure_does_not_mask_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = _calibrated_workload()
    state = real_mlx_module._RunPreflight(
        workload_path=tmp_path / "workload.json",
        model_dir=tmp_path / "model",
        conversion_summary=tmp_path / "summary.json",
        output_workspace=real_mlx_module.CANONICAL_OUTPUT_WORKSPACE,
        package_digest="sha256:" + "b" * 64,
        workload=workload,
        runtime_packages=_runtime_packages(),
        machine_observation={},
    )
    marker = tmp_path / "attempt-marker.json"
    monkeypatch.setattr(real_mlx_module, "_run_global_preflight", lambda **_: state)
    monkeypatch.setattr(
        real_mlx_module,
        "_consume_canonical_attempt",
        lambda **_: (marker, "sha256:" + "c" * 64),
    )

    class OriginalFailure(RuntimeError):
        pass

    def execute(**_: Any) -> dict[str, Any]:
        raise OriginalFailure("original run failure")

    def fail_finalization(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("marker write failure")

    monkeypatch.setattr(
        real_mlx_module, "_run_all_replicates_after_consumption", execute
    )
    monkeypatch.setattr(
        real_mlx_module, "_finalize_canonical_attempt_marker", fail_finalization
    )

    with pytest.raises(OriginalFailure, match="original run failure") as caught:
        real_mlx_module.run_all_replicates(
            workload=state.workload_path,
            model_dir=state.model_dir,
            conversion_summary=state.conversion_summary,
            output_workspace=state.output_workspace,
            expected_commit="a" * 40,
            run_attempt=1,
        )

    assert getattr(caught.value, "__notes__", []) == [
        "canonical attempt marker finalization also failed: "
        "OSError: marker write failure"
    ]


def test_public_results_allow_measured_output_failure_and_nonevicted_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _make_attempts(tmp_path, monkeypatch)
    results = real_mlx_module._public_results_from_private(workspace / "attempts")
    comparison = results["comparisons"]["1k:capacity-eviction"]
    sample = comparison["samples"][0]
    sample["control_output_token_identity"] = False
    sample["treatment_output_token_identity"] = False
    sample["control_deterministic_correctness"] = False
    sample["treatment_deterministic_correctness"] = False
    sample["verdict"] = Verdict.VERIFIED_MISS.value
    comparison["output_identity_true_pair_count"] -= 1
    comparison["correctness_true_pair_count"] -= 1
    real_mlx_module._verify_public_results(results)


def test_public_results_preserve_recomputation_attestation_mismatch_and_missing_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_request = "1k:interior-mutation:variant"
    control_request = "1k:interior-mutation:seed"
    fully_unsupported_request = "4k:suffix-only-change:variant"

    def transform(record: RequestEvidence) -> RequestEvidence:
        if record.spec.request_id == fully_unsupported_request:
            return replace(record, terminal_state=TerminalState.REFUSED)
        if (
            record.spec.request_id == control_request
            and record.spec.replicate_id == "replicate-3"
        ):
            return replace(
                record,
                reuse=replace(
                    record.reuse,
                    engine_cached_tokens=replace(
                        record.reuse.engine_cached_tokens,
                        value=1,
                    ),
                    engine_created_tokens=replace(
                        record.reuse.engine_created_tokens,
                        value=record.spec.input_token_count - 1,
                    ),
                ),
            )
        if (
            record.spec.request_id == control_request
            and record.spec.replicate_id == "replicate-4"
        ):
            required = record.reuse.policy_required_prompt_tokens.value
            assert isinstance(required, int)
            return replace(
                record,
                reuse=replace(
                    record.reuse,
                    observed_prompt_tokens=replace(
                        record.reuse.observed_prompt_tokens,
                        value=required - 1,
                    ),
                ),
            )
        if record.spec.request_id != target_request:
            return record
        if record.spec.replicate_id == "replicate-0":
            required = record.reuse.policy_required_prompt_tokens.value
            assert isinstance(required, int)
            return replace(
                record,
                reuse=replace(
                    record.reuse,
                    observed_prompt_tokens=replace(
                        record.reuse.observed_prompt_tokens,
                        value=required + 1,
                    ),
                    unexpected_recomputed_tokens=replace(
                        record.reuse.unexpected_recomputed_tokens,
                        value=1,
                    ),
                ),
            )
        if record.spec.replicate_id == "replicate-1":
            policy = record.reuse.policy_reusable_tokens.value
            assert isinstance(policy, int) and policy > 0
            cached = policy - 1
            return replace(
                record,
                reuse=replace(
                    record.reuse,
                    engine_cached_tokens=replace(
                        record.reuse.engine_cached_tokens,
                        value=cached,
                    ),
                    engine_created_tokens=replace(
                        record.reuse.engine_created_tokens,
                        value=record.spec.input_token_count - cached,
                    ),
                ),
            )
        if record.spec.replicate_id == "replicate-2":
            return replace(record, terminal_state=TerminalState.REFUSED)
        return record

    workspace = _make_attempts(
        tmp_path,
        monkeypatch,
        record_transform=transform,
    )
    results = real_mlx_module._public_results_from_private(workspace / "attempts")
    real_mlx_module._verify_public_results(
        results,
        complete_replicate_ids=set(REPLICATE_IDS),
    )

    comparison = results["comparisons"]["1k:interior-mutation"]
    assert comparison["independent_units"] == 5
    assert comparison["missing_pair_count"] == 1
    assert comparison["missing_pair_reasons"] == [
        {
            "replicate_id": "replicate-2",
            "reason": "treatment_refused_unsupported",
        }
    ]
    recomputed = next(
        sample
        for sample in comparison["samples"]
        if sample["replicate_id"] == "replicate-0"
    )
    assert recomputed["verdict"] == Verdict.RECOMPUTED.value
    assert recomputed["control_engine_attestation_matches_policy"] is True
    assert recomputed["control_observed_matches_policy_required"] is True
    assert recomputed["engine_attestation_matches_policy"] is True
    assert recomputed["observed_matches_policy_required"] is False
    assert recomputed["unexpected_recomputed_tokens"] == 1
    assert recomputed["paired_latency_comparable"] is False
    assert recomputed["client_ttft_difference_seconds"] is None
    assert recomputed["total_ratio"] is None

    mismatch = next(
        sample
        for sample in comparison["samples"]
        if sample["replicate_id"] == "replicate-1"
    )
    assert mismatch["verdict"] == Verdict.INVALID.value
    assert mismatch["engine_attestation_matches_policy"] is False
    assert mismatch["observed_matches_policy_required"] is True
    assert mismatch["paired_latency_comparable"] is False
    assert mismatch["client_ttft_ratio"] is None
    assert mismatch["allocator_peak_difference_bytes"] is None
    control_mismatch = next(
        sample
        for sample in comparison["samples"]
        if sample["replicate_id"] == "replicate-3"
    )
    assert control_mismatch["control_verdict"] == Verdict.INVALID.value
    assert control_mismatch["control_engine_attestation_matches_policy"] is False
    assert control_mismatch["paired_latency_comparable"] is False
    assert control_mismatch["total_difference_seconds"] is None
    control_observed_mismatch = next(
        sample
        for sample in comparison["samples"]
        if sample["replicate_id"] == "replicate-4"
    )
    assert control_observed_mismatch["control_unexpected_recomputed_tokens"] == 0
    assert (
        control_observed_mismatch["control_observed_matches_policy_required"] is False
    )
    assert control_observed_mismatch["paired_latency_comparable"] is False
    assert comparison["statistics"]["unexpected_recomputed_tokens"]["values"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]

    tampered = json.loads(json.dumps(results))
    tampered_sample = next(
        sample
        for sample in tampered["comparisons"]["1k:interior-mutation"]["samples"]
        if sample["replicate_id"] == "replicate-1"
    )
    tampered_sample["engine_attestation_matches_policy"] = True
    with pytest.raises(RealMLXExperimentError, match="agreement boolean"):
        real_mlx_module._verify_public_results(tampered)

    unsupported = results["comparisons"]["4k:suffix-only-change"]
    assert unsupported["independent_units"] == 0
    assert unsupported["missing_pair_count"] == 6
    assert unsupported["samples"] == []
    assert {item["reason"] for item in unsupported["missing_pair_reasons"]} == {
        "treatment_refused_unsupported"
    }
    assert unsupported["statistics"]["unexpected_recomputed_tokens"] == {
        "count": 0,
        "values": [],
        "median": None,
        "minimum": None,
        "maximum": None,
    }


def test_run_all_rejects_symlinked_input_before_creating_workspace(
    tmp_path: Path,
) -> None:
    target = tmp_path / "workload-target.json"
    target.write_text("{}")
    workload = tmp_path / "workload.json"
    workload.symlink_to(target)
    output = tmp_path / "run"

    with pytest.raises(RealMLXExperimentError, match="must not be a symlink"):
        real_mlx_module.run_all_replicates(
            workload=workload,
            model_dir=tmp_path / "model",
            conversion_summary=tmp_path / "summary.json",
            output_workspace=output,
            expected_commit="a" * 40,
            run_attempt=1,
        )

    assert not output.exists()


def test_replicate_child_uses_current_module_and_expected_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sandbox = Path("/usr/bin/sandbox-exec")
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if self == sandbox else original_is_file(self),
    )
    monkeypatch.setattr(real_mlx_module.os, "access", lambda *_args: True)

    command = real_mlx_module._replicate_child_command(
        workload=(tmp_path / "workload.json").resolve(),
        model_dir=(tmp_path / "model").resolve(),
        conversion_summary=(tmp_path / "summary.json").resolve(),
        replicate_id="replicate-0",
        output_dir=(tmp_path / "attempt").resolve(),
        expected_commit="a" * 40,
        runtime_packages_digest="sha256:" + "b" * 64,
    )

    bootstrap_index = command.index(str(real_mlx_module._TRUSTED_BOOTSTRAP))
    assert command[bootstrap_index - 3 : bootstrap_index + 4] == [
        real_mlx_module.sys.executable,
        "-I",
        "-S",
        str(real_mlx_module._TRUSTED_BOOTSTRAP),
        "--trusted-repo-root",
        str(real_mlx_module._PROJECT_ROOT),
        "replicate",
    ]
    assert command[command.index("--expected-commit") + 1] == "a" * 40
    assert command[command.index("--runtime-packages-digest") + 1] == (
        "sha256:" + "b" * 64
    )
    assert real_mlx_module.SANDBOX_POLICY in command


def test_sandbox_probe_is_isolated_and_uses_non_repository_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sandbox = Path("/usr/bin/sandbox-exec")
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if self == sandbox else original_is_file(self),
    )
    monkeypatch.setattr(real_mlx_module.os, "access", lambda *_args: True)
    calls: list[dict[str, Any]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "sandbox_probe_passed": True,
                    "runtime_packages": _runtime_packages(),
                }
            ),
            "",
        )

    monkeypatch.setattr(real_mlx_module.subprocess, "run", fake_run)
    output_workspace = tmp_path / "run"
    identity = real_mlx_module._run_exact_sandbox_probe(
        output_workspace=output_workspace,
        expected_commit="a" * 40,
        expected_package_digest="sha256:" + "b" * 64,
    )

    assert identity == _runtime_packages()
    assert len(calls) == 1
    assert calls[0]["cwd"] == tmp_path
    command = calls[0]["command"]
    bootstrap_index = command.index(str(real_mlx_module._TRUSTED_BOOTSTRAP))
    assert command[bootstrap_index - 3 : bootstrap_index + 4] == [
        real_mlx_module.sys.executable,
        "-I",
        "-S",
        str(real_mlx_module._TRUSTED_BOOTSTRAP),
        "--trusted-repo-root",
        str(real_mlx_module._PROJECT_ROOT),
        "sandbox-probe",
    ]
    assert real_mlx_module.SANDBOX_POLICY in command


def test_sandbox_probe_refuses_leftover_temp_without_deleting_it(
    tmp_path: Path,
) -> None:
    output_workspace = tmp_path / "run"
    leftover = tmp_path / ".run.preflight-tmp"
    leftover.mkdir()
    marker = leftover / "preserve.txt"
    marker.write_text("preserve", encoding="ascii")

    with pytest.raises(
        RealMLXExperimentError,
        match="PREFLIGHT_VALIDATION_FAILED.*already exists",
    ):
        real_mlx_module._run_exact_sandbox_probe(
            output_workspace=output_workspace,
            expected_commit="a" * 40,
            expected_package_digest="sha256:" + "b" * 64,
        )

    assert marker.read_text(encoding="ascii") == "preserve"


def test_preflight_refusal_writes_only_explicit_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        real_mlx_module,
        "_run_global_preflight",
        lambda **_kwargs: (_ for _ in ()).throw(
            RealMLXExperimentError("NEEDS_CLEAN_BOOT:preflight_heavy_process_present")
        ),
    )
    output_workspace = tmp_path / "run"
    receipt = tmp_path / "preflight.json"
    result = real_mlx_module.run_preflight(
        workload=tmp_path / "workload.json",
        model_dir=tmp_path / "model",
        conversion_summary=tmp_path / "summary.json",
        output_workspace=output_workspace,
        expected_commit="a" * 40,
        output=receipt,
        run_attempt=1,
    )

    assert result == {
        "preflight_passed": False,
        "reason": "NEEDS_CLEAN_BOOT:preflight_heavy_process_present",
    }
    assert json.loads(receipt.read_text()) == {
        "schema_version": "1",
        "canonical_execution": False,
        "canonical_run_id": real_mlx_module.CANONICAL_RUN_ID,
        "canonical_output_workspace": str(real_mlx_module.CANONICAL_OUTPUT_WORKSPACE),
        "canonical_attempt_marker_id": real_mlx_module.CANONICAL_ATTEMPT_MARKER.name,
        "run_attempt": 1,
        "prior_invalidated_run_ledger_digests": [],
        "status": "refused",
        "reason": "NEEDS_CLEAN_BOOT:preflight_heavy_process_present",
    }
    assert not output_workspace.exists()


def test_child_environment_is_a_minimal_explicit_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HOME",
        "PYTHONPATH",
        "DYLD_LIBRARY_PATH",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AZURE_FEDERATED_TOKEN_FILE",
        "UNRECOGNIZED_FUTURE_SECRET",
    ):
        monkeypatch.setenv(name, f"sensitive-{name}")

    instance_id = "a" * 32
    environment = real_mlx_module._offline_child_environment(instance_id)

    assert environment == {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_DISABLED": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        real_mlx_module.RUN_INSTANCE_ENV: instance_id,
    }


def test_cleanup_failure_preserves_partial_evidence_and_aborts_remaining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    conversion_summary = tmp_path / "summary.json"
    conversion_summary.write_text("{}")
    calibrated = _calibrated_workload()
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        calibrated.to_dict()["workload_digest"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in calibrated.lanes},
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_validate_supervisor_source",
        lambda _: "sha256:" + "b" * 64,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "load_workload",
        lambda *_args, **_kwargs: calibrated,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "verify_model_contract",
        lambda *_args, **_kwargs: EXPECTED_MODEL_ARTIFACT_DIGEST,
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_run_exact_sandbox_probe",
        lambda **_kwargs: _runtime_packages(),
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_runtime_package_identity",
        lambda *_args, **_kwargs: _runtime_packages(),
    )
    safe_observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: safe_observation,
    )
    launches = 0

    class OrphanedProcess:
        def __init__(self, command: list[str], **_kwargs: Any) -> None:
            nonlocal launches
            launches += 1
            self.pid = 20_000
            self.returncode = 0
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            for name in ("bundle",):
                (output / name).mkdir()
            for name in ("stages.jsonl", "workload.json", "environment.json"):
                (output / name).write_text("partial")
            _write_json(output / "attempt.json", {"status": "complete"})

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(real_mlx_module.subprocess, "Popen", OrphanedProcess)
    monkeypatch.setattr(real_mlx_module, "_group_exists", lambda _pid: True)
    monkeypatch.setattr(
        real_mlx_module,
        "_terminate_process_group",
        lambda *_args, **_kwargs: False,
    )

    workspace = tmp_path / "run"
    result = real_mlx_module.run_all_replicates(
        workload=workload_path,
        model_dir=model_dir,
        conversion_summary=conversion_summary,
        output_workspace=workspace,
        expected_commit="a" * 40,
        run_attempt=1,
    )

    assert result["complete_replicates"] == 0
    assert launches == 1
    partial = workspace / "private-artifacts" / "replicate-0" / "partial-output"
    assert {
        "bundle",
        "stages.jsonl",
        "workload.json",
        "environment.json",
        "stale-complete-attempt.json",
    } <= {path.name for path in partial.iterdir()}
    first = json.loads(
        (workspace / "attempts" / "replicate-0" / "attempt.json").read_text()
    )
    assert first["reason"] == "process_cleanup_failed"
    for replicate_id in REPLICATE_IDS[1:]:
        attempt = json.loads(
            (workspace / "attempts" / replicate_id / "attempt.json").read_text()
        )
        assert attempt["reason"] == "supervisor_aborted_before_start"
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    assert sum(row["event"] == "finalized" for row in ledger) == 6
