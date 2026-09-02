"""Offline tests for the Qwen3-8B M5 Pro control: manifests, namespace
separation, the subprocess-isolated benchmark runner, reports, and the
CLI. No test in this module loads real model weights or spawns a real
MLX runtime; a monkeypatched fake stands in for the checkpoint so the
resume, safety-gate, tokenizer-count, and subprocess-isolation logic
can be verified deterministically and offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.lab.core import HostSnapshot, LabError, SafetyDecision
from llmtracefx.optimizer.lab.manifest import LabManifest
from llmtracefx.optimizer.lab.qwen3_8b import benchmark as bm
from llmtracefx.optimizer.lab.qwen3_8b import cli as qcli
from llmtracefx.optimizer.lab.qwen3_8b.control_manifest import (
    ControlManifestError,
    ControlManifestTemplate,
    bind_control_manifest,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion import (
    ConversionReceipt,
    conversion_manifest_hash,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion_manifest import ConversionManifest
from llmtracefx.optimizer.lab.qwen3_8b.report import (
    build_control_report,
    write_control_reports,
)
from llmtracefx.optimizer.manifest import EnvironmentManifest
from llmtracefx.optimizer.schema import PlatformInfo
from llmtracefx.optimizer.workloads.schema import ContextTier
from llmtracefx.optimizer.workloads.verify import (
    RowResult,
    RowStatus,
    RowVerification,
)

TWENTY_SEVEN_B_MANIFEST_PATH = Path(
    "llmtracefx/optimizer/lab/data/lab-manifest-v1.json"
)
FRONTIER_MANIFEST_PATH = Path(
    "llmtracefx/optimizer/lab/data/fit-frontier-manifest-v1.json"
)
AUTOPSY_MANIFEST_PATH = Path("llmtracefx/optimizer/lab/data/autopsy-manifest-v1.json")
CONTROL_TEMPLATE_PATH = Path(
    "llmtracefx/optimizer/lab/qwen3_8b/data/qwen3-8b-control-manifest-template-v1.json"
)
CONVERSION_MANIFEST_PATH = Path(
    "llmtracefx/optimizer/lab/qwen3_8b/data/qwen3-8b-conversion-manifest-v1.json"
)


@pytest.fixture(autouse=True)
def stable_m5_platform(monkeypatch):
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.mlx.record_platform",
        lambda accelerator: PlatformInfo(
            os_name="Darwin",
            os_version="25.6.0",
            architecture="arm64",
            cpu_cores=18,
            total_memory_gb=24.0,
            accelerator=accelerator,
        ),
    )
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.mlx.collect_environment_manifest",
        lambda **kwargs: EnvironmentManifest(
            schema_version="1",
            collected_at="2026-08-31T00:00:00.000000Z",
            os_name="Darwin",
            os_release="25.6.0",
            architecture="arm64",
            python_implementation="CPython",
            python_version="3.13.15",
            cpu_count=18,
            total_memory_gb=24.0,
            package_versions={
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "transformers": "5.16.1",
            },
        ),
    )


def _conversion_manifest() -> ConversionManifest:
    return ConversionManifest.read_json(CONVERSION_MANIFEST_PATH)


def _fabricated_receipt(**overrides) -> ConversionReceipt:
    """A receipt whose identity fields match the packaged conversion spec
    exactly by default, so ``bind_control_manifest``'s provenance
    cross-check succeeds unless a test deliberately overrides a field to
    exercise a mismatch refusal."""
    values = {
        "schema_version": "1",
        "conversion_id": "qwen3-8b-mlx-q4g64-self-convert-v1",
        "conversion_manifest_hash": conversion_manifest_hash(_conversion_manifest()),
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T01:00:00Z",
        "source": {
            "official_id": "Qwen/Qwen3-8B",
            "official_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "license": "Apache-2.0",
        },
        "converter": {
            "package": "mlx-lm",
            "version": "0.31.3",
            "git_revision": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
        },
        "parameters": {
            "quantize": True,
            "q_group_size": 64,
            "q_bits": 4,
            "q_mode": "affine",
            "dtype": None,
            "quant_predicate": None,
            "dequantize": False,
            "trust_remote_code": False,
            "upload_repo": None,
        },
        "argv": ("mlx_lm", "convert"),
        "output_files": (
            {"path": "config.json", "size_bytes": 10, "sha256": "a" * 64},
            {"path": "model.safetensors", "size_bytes": 20, "sha256": "b" * 64},
        ),
        "output_total_bytes": 30,
        "host": {"os_name": "Darwin"},
    }
    values.update(overrides)
    return ConversionReceipt(**values)


def _bound_manifest(
    tmp_path: Path, *, receipt: ConversionReceipt | None = None
) -> tuple[LabManifest, Path]:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    receipt = receipt or _fabricated_receipt()
    bound = bind_control_manifest(
        template, receipt, conversion_manifest=_conversion_manifest()
    )
    payload = template.to_dict()
    payload["model"] = {
        **payload["model"],
        "files": [
            {"path": pin.path, "size_bytes": pin.size_bytes, "sha256": pin.sha256}
            for pin in bound.model.files
        ],
        "expected_download_bytes": bound.model.expected_download_bytes,
        "revision": bound.model.revision,
    }
    manifest_path = tmp_path / "control-manifest.bound.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LabManifest.read_json(manifest_path), manifest_path


def _safe_decision() -> SafetyDecision:
    return SafetyDecision(
        safe=True,
        blockers=(),
        snapshot=HostSnapshot(
            collected_at="2026-09-01T00:00:00.000000Z",
            os_name="Darwin",
            os_release="25.6.0",
            architecture="arm64",
            python_implementation="CPython",
            python_version="3.13.15",
            cpu_count=18,
            chip="Apple M5 Pro",
            total_memory_bytes=24 * 1024**3,
            memory_free_percent=50.0,
            swap_used_bytes=0,
            disk_free_bytes=500 * 1024**3,
            package_versions={
                "mlx": "0.32.2",
                "mlx-lm": "0.31.3",
                "transformers": "5.16.1",
            },
        ),
    )


@dataclass
class FakeResponse:
    text: str
    from_draft: bool = False
    prompt_tokens: int = 5
    generation_tokens: int = 1
    finish_reason: str | None = None


class FakeChatTokenizer:
    bos_token = None

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    ):
        assert enable_thinking is False
        assert tokenize is True
        assert add_generation_prompt is True
        return [1, 2, 3, 4, 5]


class FakeRuntime:
    mlx_version = "0.32.2"
    mlx_lm_version = "0.31.3"
    runtime_name = "mlx-lm"
    runtime_version = "0.31.3"
    generate_calls = 0
    peak_bytes = 1024**3
    fail_generation = False
    wrong_answer = False

    def __init__(self, **kwargs):
        self.prompt = ""
        self._kwargs = kwargs

    def load_model(self, path):
        return object(), FakeChatTokenizer()

    def encode(self, tokenizer, prompt):
        self.prompt = prompt
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self._kwargs.get("enable_thinking", False),
        )

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return MLXMemorySnapshot(
            active_bytes=512 * 1024**2,
            cache_bytes=128 * 1024**2,
            peak_bytes=type(self).peak_bytes,
        )

    def accelerator_name(self):
        return "Apple M5 Pro"

    def stream_generate(
        self,
        model,
        tokenizer,
        prompt_tokens,
        *,
        max_tokens,
        draft_model,
        num_draft_tokens,
    ):
        type(self).generate_calls += 1
        if type(self).fail_generation:
            raise RuntimeError("simulated failure")
        if type(self).wrong_answer:
            text = "this response satisfies neither evaluator's expected format"
        else:
            text = (
                '{"name":"Priya","age":34,"is_active":true}'
                if "Priya Nakamura" in self.prompt
                else "3 hours. The combined closing speed is 70 miles per hour."
            )
        yield FakeResponse(text=text, prompt_tokens=len(prompt_tokens))


def _in_process_launcher(manifest: LabManifest):
    """A ``launcher`` that runs the exact child-row code path in-process
    (bypassing a real subprocess spawn) so tests can assert on the tier
    loop/safety-gate/resume logic without any process overhead."""

    def launch(
        *,
        manifest_path,
        workload_id,
        tier,
        repetition_index,
        warmup,
        model_path,
        output_dir,
        resume,
        timeout_seconds,
        cleanup_grace_seconds,
    ):
        bm.run_child_row(
            manifest,
            workload_id=workload_id,
            tier=tier,
            repetition_index=repetition_index,
            warmup=warmup,
            model_path=model_path,
            output_dir=output_dir,
            manifest_dir=manifest_path.parent,
            resume=resume,
            hardware_fingerprint="fixed-test-fingerprint",
        )
        return bm.ChildLaunchResult(
            exit_code=0, timed_out=False, descendants_cleaned=True
        )

    return launch


def _prepare(monkeypatch):
    monkeypatch.setattr(bm, "verify_model", lambda *args, **kwargs: {"verified": True})
    monkeypatch.setattr(bm, "assess_safety", lambda *args, **kwargs: _safe_decision())
    monkeypatch.setattr(bm, "Qwen3ChatMLXLMRuntime", FakeRuntime)
    FakeRuntime.generate_calls = 0
    FakeRuntime.peak_bytes = 1024**3
    FakeRuntime.fail_generation = False
    FakeRuntime.wrong_answer = False


# ---------------------------------------------------------------------------
# Control manifest template / bind
# ---------------------------------------------------------------------------


def test_control_template_parses_without_precommitted_model_identity() -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    raw = template.to_dict()
    assert "files" not in raw["model"]
    assert "expected_download_bytes" not in raw["model"]
    assert "revision" not in raw["model"]
    assert raw["model"]["model_family"] == "qwen3"
    assert raw["generation"]["enable_thinking"] is False
    assert raw["repetitions"]["warmup_per_tier"] == 1
    assert raw["repetitions"]["measured_per_workload"] == 2


def test_control_template_rejects_precommitted_files_field() -> None:
    payload = json.loads(CONTROL_TEMPLATE_PATH.read_text(encoding="utf-8"))
    payload["model"]["files"] = [{"path": "x", "size_bytes": 1, "sha256": "0" * 64}]
    with pytest.raises(ControlManifestError, match="must not pre-commit"):
        ControlManifestTemplate.from_dict(payload)


def test_bind_control_manifest_is_deterministic_and_never_fabricated(tmp_path) -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt()
    first = bind_control_manifest(
        template, receipt, conversion_manifest=conversion_manifest
    )
    second = bind_control_manifest(
        template, receipt, conversion_manifest=conversion_manifest
    )
    assert first.model.revision == second.model.revision
    assert len(first.model.revision) == 40
    assert all(c in "0123456789abcdef" for c in first.model.revision)
    assert first.model.expected_download_bytes == 30
    assert first.model.model_family == "qwen3"
    # Changing the receipt's output hashes must change the binding.
    different = bind_control_manifest(
        template,
        _fabricated_receipt(
            output_files=(
                {"path": "config.json", "size_bytes": 10, "sha256": "c" * 64},
            ),
            output_total_bytes=10,
        ),
        conversion_manifest=conversion_manifest,
    )
    assert different.model.revision != first.model.revision


def test_bind_refuses_a_non_completed_receipt() -> None:
    from dataclasses import replace

    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    completed = bind_control_manifest(
        template, _fabricated_receipt(), conversion_manifest=conversion_manifest
    )
    assert completed.model.model_family == "qwen3"
    # bind_control_manifest's own guard rejects a hand-built non-completed
    # receipt (``ConversionReceipt.from_dict`` already enforces this on any
    # receipt read back from disk; this exercises the belt-and-braces
    # in-memory guard directly).
    stale_receipt = replace(_fabricated_receipt(), status="failed")
    with pytest.raises(ControlManifestError, match="non-completed"):
        bind_control_manifest(
            template, stale_receipt, conversion_manifest=conversion_manifest
        )


@pytest.mark.parametrize(
    ("override_key", "override_value", "expected_substring"),
    [
        ("official_id", "Qwen/Qwen3.8-27B", "source.official_id"),
        ("official_revision", "0" * 40, "source.official_revision"),
        ("license", "MIT", "source.license"),
    ],
)
def test_bind_refuses_receipt_with_mismatched_source_identity(
    override_key, override_value, expected_substring
) -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt(
        source={**_fabricated_receipt().source, override_key: override_value}
    )
    with pytest.raises(ControlManifestError, match=expected_substring):
        bind_control_manifest(
            template, receipt, conversion_manifest=conversion_manifest
        )


@pytest.mark.parametrize(
    ("override_key", "override_value", "expected_substring"),
    [
        ("package", "mlx-vlm", "converter.package"),
        ("version", "0.0.0", "converter.version"),
        ("git_revision", "1" * 40, "converter.git_revision"),
    ],
)
def test_bind_refuses_receipt_with_mismatched_converter_identity(
    override_key, override_value, expected_substring
) -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt(
        converter={**_fabricated_receipt().converter, override_key: override_value}
    )
    with pytest.raises(ControlManifestError, match=expected_substring):
        bind_control_manifest(
            template, receipt, conversion_manifest=conversion_manifest
        )


def test_bind_refuses_receipt_with_mismatched_conversion_id() -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt(conversion_id="some-other-conversion")
    with pytest.raises(ControlManifestError, match="conversion_id"):
        bind_control_manifest(
            template, receipt, conversion_manifest=conversion_manifest
        )


def test_bind_refuses_receipt_with_mismatched_quantization_parameters() -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt(
        parameters={**_fabricated_receipt().parameters, "q_bits": 8}
    )
    with pytest.raises(ControlManifestError, match="parameters"):
        bind_control_manifest(
            template, receipt, conversion_manifest=conversion_manifest
        )


def test_bind_refuses_receipt_with_mismatched_manifest_hash() -> None:
    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    conversion_manifest = _conversion_manifest()
    receipt = _fabricated_receipt(conversion_manifest_hash="sha256:" + "f" * 64)
    with pytest.raises(ControlManifestError, match="conversion_manifest_hash"):
        bind_control_manifest(
            template, receipt, conversion_manifest=conversion_manifest
        )


# ---------------------------------------------------------------------------
# Namespace separation from the packaged 27B lab
# ---------------------------------------------------------------------------


def test_namespace_fully_separate_from_27b_lab_paths() -> None:
    control_template = json.loads(CONTROL_TEMPLATE_PATH.read_text(encoding="utf-8"))
    conversion_manifest = json.loads(
        CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    lab_27b = json.loads(TWENTY_SEVEN_B_MANIFEST_PATH.read_text(encoding="utf-8"))
    frontier_27b = json.loads(FRONTIER_MANIFEST_PATH.read_text(encoding="utf-8"))
    autopsy_27b = json.loads(AUTOPSY_MANIFEST_PATH.read_text(encoding="utf-8"))

    control_paths = {
        control_template["artifacts"]["workspace"],
        control_template["artifacts"]["model_cache"],
        control_template["artifacts"]["shareable_example_dir"],
        conversion_manifest["artifacts"]["source_cache"],
        conversion_manifest["artifacts"]["output_cache"],
        conversion_manifest["artifacts"]["workspace"],
    }
    existing_27b_paths = {
        lab_27b["artifacts"]["workspace"],
        lab_27b["artifacts"]["model_cache"],
        lab_27b["artifacts"]["shareable_example_dir"],
        frontier_27b["artifacts"]["workspace"],
        frontier_27b["artifacts"]["shareable_example_dir"],
        autopsy_27b["artifacts"]["workspace"],
    }
    assert control_paths.isdisjoint(existing_27b_paths)
    # None of the new paths is a parent/child directory of an existing one.
    for new_path in control_paths:
        for old_path in existing_27b_paths:
            new_parts = Path(new_path).parts
            old_parts = Path(old_path).parts
            shorter, longer = sorted((new_parts, old_parts), key=len)
            assert longer[: len(shorter)] != shorter, (new_path, old_path)
    assert control_template["lab_id"] != lab_27b["lab_id"]
    assert (
        control_template["model"]["repository_id"] != lab_27b["model"]["repository_id"]
    )


def test_control_manifest_leaves_27b_manifests_byte_identical_on_disk() -> None:
    # Guards against any accidental in-place edit of the packaged 27B
    # artifacts while adding this control (only additive fields with
    # backward-compatible defaults were introduced in manifest.py/core.py).
    for path, expected_lab_id in (
        (TWENTY_SEVEN_B_MANIFEST_PATH, "m5-pro-qwen3.8-27b-v1"),
        (FRONTIER_MANIFEST_PATH, None),
        (AUTOPSY_MANIFEST_PATH, None),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if expected_lab_id is not None:
            assert payload["lab_id"] == expected_lab_id
        assert "model_family" not in payload.get("model", {})


def test_prompt_hashes_reused_from_catalog_are_model_independent() -> None:
    """Prompt materialization never depends on which model executes it,
    so the control manifest can (and does) pin the exact same prompt
    hashes the 27B lab pins for the same (workload, tier) pair."""
    from llmtracefx.optimizer.lab.core import verify_catalog

    template = ControlManifestTemplate.read_json(CONTROL_TEMPLATE_PATH)
    bound_in_memory = bind_control_manifest(
        template, _fabricated_receipt(), conversion_manifest=_conversion_manifest()
    )
    verify_catalog(bound_in_memory)


# ---------------------------------------------------------------------------
# Benchmark runner: run mode, subprocess isolation, safety, resume
# ---------------------------------------------------------------------------


def test_run_mode_defaults_exploratory_and_publication_requires_clean_boot(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    with pytest.raises(LabError, match="operator assertion"):
        bm.run_benchmark(
            manifest,
            manifest_path=manifest_path,
            workspace=tmp_path / "ws1",
            model_path=tmp_path / "model",
            max_tier="2k",
            run_mode="publication",
            clean_boot_confirmed=False,
            launcher=_in_process_launcher(manifest),
        )
    with pytest.raises(LabError, match="only valid in publication mode"):
        bm.run_benchmark(
            manifest,
            manifest_path=manifest_path,
            workspace=tmp_path / "ws2",
            model_path=tmp_path / "model",
            max_tier="2k",
            run_mode="exploratory",
            clean_boot_confirmed=True,
            launcher=_in_process_launcher(manifest),
        )
    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "ws3",
        model_path=tmp_path / "model",
        max_tier="2k",
    )
    assert state["run_mode"] == "exploratory"
    assert state["clean_boot_confirmed"] is False


def test_subprocess_launcher_uses_new_session_and_no_shell(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    captured = {}

    class FakeProcess:
        pid = 12345
        returncode = 0

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(bm.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(bm, "_clean_process_group", lambda *args: True)
    result = bm.launch_row_subprocess(
        manifest_path=manifest_path,
        workload_id="structured-json-profile-extraction",
        tier=ContextTier.TIER_2K,
        repetition_index=0,
        warmup=False,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        resume=True,
        timeout_seconds=10,
        cleanup_grace_seconds=1,
    )
    assert captured["argv"][1:4] == [
        "-m",
        "llmtracefx.optimizer.lab.qwen3_8b.benchmark",
    ][: len(["-m", "llmtracefx.optimizer.lab.qwen3_8b.benchmark"])] or captured["argv"][
        1:4
    ] == [
        "-m",
        "llmtracefx.optimizer.lab.qwen3_8b.benchmark",
        "_child",
    ]
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False
    assert result.descendants_cleaned is True


def test_row_timeout_escalates_before_declaring_cleaned(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)

    class FakeProcess:
        pid = 999
        returncode = -15

        def __init__(self, argv, **kwargs):
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                raise bm.subprocess.TimeoutExpired("child", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

    cleaned_polled_at = []
    monkeypatch.setattr(bm.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        bm,
        "_clean_process_group",
        lambda process, group, grace: cleaned_polled_at.append(process.poll()) or True,
    )
    result = bm.launch_row_subprocess(
        manifest_path=manifest_path,
        workload_id="structured-json-profile-extraction",
        tier=ContextTier.TIER_2K,
        repetition_index=0,
        warmup=False,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        resume=True,
        timeout_seconds=0.01,
        cleanup_grace_seconds=0.01,
    )
    assert result.timed_out is True
    assert result.descendants_cleaned is True
    assert cleaned_polled_at == [-15]


def test_stop_on_first_failed_row_preserves_lower_tier_evidence(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)

    def launcher(**kwargs):
        if kwargs["tier"].value == "8k" and not kwargs["warmup"]:
            FakeRuntime.fail_generation = True
        else:
            FakeRuntime.fail_generation = False
        return _in_process_launcher(manifest)(**kwargs)

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="16k",
        launcher=launcher,
    )
    statuses = [tier["status"] for tier in state["tiers"]]
    assert statuses[0] == "passed"
    assert "16k" not in [tier["tier"] for tier in state["tiers"]]
    assert state["unattempted_tiers"] == ["16k"]
    assert state["status"] == "stopped"
    # Tier 2k's evidence remains on disk, unaffected by the later failure.
    assert (
        tmp_path
        / "workspace"
        / "results"
        / "runs"
        / "structured-json-profile-extraction-2k-rep-01"
        / "verification.json"
    ).is_file()


def test_evaluator_failure_stops_immediately_in_measured_rows(
    tmp_path, monkeypatch
) -> None:
    """RowStatus.COMPLETED with outcome_success=False is an evaluator
    failure (the model executed cleanly but got the task wrong); it must
    stop immediately, exactly like an infrastructure failure, and never
    let the tier or a later tier pass."""
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)

    def launcher(**kwargs):
        FakeRuntime.wrong_answer = not kwargs["warmup"]
        return _in_process_launcher(manifest)(**kwargs)

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="8k",
        launcher=launcher,
    )
    assert state["status"] == "stopped"
    assert state["tiers"][0]["status"] == "failed"
    assert any(
        "evaluator reported outcome_success=False" in reason
        for reason in state["stop_reasons"]
    )
    assert "8k" not in [tier["tier"] for tier in state["tiers"]]
    assert state["unattempted_tiers"] == ["8k"]
    # The row itself really did complete (collector succeeded); only the
    # evaluator's task-quality verdict failed.
    run_dir = (
        tmp_path
        / "workspace"
        / "results"
        / "runs"
        / "structured-json-profile-extraction-2k-rep-01"
    )
    verification = RowVerification.read_json(run_dir / "verification.json")
    assert verification.status == RowStatus.COMPLETED
    assert verification.outcome_success is False


def test_evaluator_failure_stops_immediately_in_warmup(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    FakeRuntime.wrong_answer = True

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=_in_process_launcher(manifest),
    )
    assert state["status"] == "stopped"
    assert state["tiers"][0]["status"] == "failed_warmup"
    assert any(
        "evaluator reported outcome_success=False" in reason
        for reason in state["stop_reasons"]
    )
    # No measured row ever ran once the warmup's evaluator failed.
    assert not (tmp_path / "workspace" / "results").exists()


def test_parent_never_trusts_disk_artifact_after_nonzero_child_exit(
    tmp_path, monkeypatch
) -> None:
    """A non-zero child exit code must never be overridden by whatever
    verification.json happens to already exist at that row's path (e.g.
    stale evidence from an unrelated earlier attempt)."""
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"

    # Plant a stale, otherwise well-formed "completed" verification.json at
    # the exact path the first warmup row would use.
    stale_run_dir = (
        workspace
        / "warmups"
        / "2k"
        / "runs"
        / "structured-json-profile-extraction-2k-warmup-01"
    )
    stale_run_dir.mkdir(parents=True, exist_ok=True)
    stale_verification = RowVerification(
        schema_version="2",
        run_id="structured-json-profile-extraction-2k-warmup-01",
        workload_id="structured-json-profile-extraction",
        workload_version="1",
        category="structured_json",
        context_tier="2k",
        decode_mode="autoregressive",
        status=RowStatus.COMPLETED,
        reason=None,
        recorded_prompt_hash="sha256:" + "0" * 64,
        verified_prompt_hash="sha256:" + "0" * 64,
        run_binding_hash="sha256:" + "1" * 64,
        resumed=False,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1.0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        final_record_path=None,
        collection_dir=None,
    )
    (stale_run_dir / "verification.json").write_text(
        stale_verification.to_json(), encoding="utf-8"
    )

    def crashing_launcher(**kwargs):
        return bm.ChildLaunchResult(
            exit_code=1, timed_out=False, descendants_cleaned=True
        )

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=crashing_launcher,
    )
    assert state["status"] == "stopped"
    assert state["tiers"][0]["status"] == "failed_warmup"
    assert any(
        "never trusted after a crash" in reason for reason in state["stop_reasons"]
    )


def test_parent_never_advances_on_mismatched_child_artifact(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    wrong_verification = RowVerification(
        schema_version="2",
        run_id="wrong-run-id",
        workload_id="structured-json-profile-extraction",
        workload_version="1",
        category="structured_json",
        context_tier="2k",
        decode_mode="autoregressive",
        status=RowStatus.COMPLETED,
        reason=None,
        recorded_prompt_hash="sha256:" + "0" * 64,
        verified_prompt_hash="sha256:" + "0" * 64,
        run_binding_hash="sha256:" + "1" * 64,
        resumed=False,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1.0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        final_record_path=None,
        collection_dir=None,
    )
    monkeypatch.setattr(
        bm,
        "_row_result_from_disk",
        lambda *args, **kwargs: RowResult(
            entry=object(), verification=wrong_verification, final_record=None
        ),
    )

    result = bm._run_row_isolated(
        manifest,
        manifest_path=manifest_path,
        workload_id="structured-json-profile-extraction",
        tier=ContextTier.TIER_2K,
        repetition_index=0,
        warmup=True,
        model_path=tmp_path / "model",
        output_dir=tmp_path / "output",
        resume=False,
        timeout_seconds=1,
        cleanup_grace_seconds=1,
        launcher=lambda **kwargs: bm.ChildLaunchResult(
            exit_code=0, timed_out=False, descendants_cleaned=True
        ),
    )

    assert result.verification.status is RowStatus.FAILED
    assert "artifact identity is stale or mismatched" in (
        result.verification.reason or ""
    )


def test_missing_supported_evidence_after_exit_synthesizes_failure_and_stops(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)

    def launcher(**kwargs):
        # Exits "successfully" without ever writing verification.json.
        return bm.ChildLaunchResult(
            exit_code=0, timed_out=False, descendants_cleaned=True
        )

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=launcher,
    )
    assert state["status"] == "stopped"
    assert state["tiers"][0]["status"] == "failed_warmup"
    assert any(
        "without writing supported evidence" in reason
        for reason in state["stop_reasons"]
    )


def test_timeout_and_cleanup_failure_stop_without_advancing(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)

    def timeout_launcher(**kwargs):
        return bm.ChildLaunchResult(
            exit_code=None, timed_out=True, descendants_cleaned=True
        )

    state = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "ws-timeout",
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=timeout_launcher,
    )
    assert state["status"] == "stopped"
    assert any("timeout" in reason for reason in state["stop_reasons"])

    def cleanup_failure_launcher(**kwargs):
        return bm.ChildLaunchResult(
            exit_code=1, timed_out=False, descendants_cleaned=False
        )

    state2 = bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=tmp_path / "ws-cleanup",
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=cleanup_failure_launcher,
    )
    assert state2["status"] == "stopped"
    assert any("cleanup failed" in reason for reason in state2["stop_reasons"])


def test_resume_reuses_only_verified_matching_evidence(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    launcher = _in_process_launcher(manifest)

    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        resume=True,
        launcher=launcher,
    )
    first_calls = FakeRuntime.generate_calls
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        resume=True,
        launcher=launcher,
    )
    assert FakeRuntime.generate_calls == first_calls + 1  # only the warmup reran

    statuses = [
        RowVerification.read_json(path).status
        for path in (workspace / "results" / "runs").glob("*/verification.json")
    ]
    assert statuses and set(statuses) == {RowStatus.SKIPPED}


def test_stale_evidence_is_never_trusted_on_resume(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    launcher = _in_process_launcher(manifest)
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        resume=True,
        launcher=launcher,
    )
    record_path = next((workspace / "results" / "runs").glob("*/final_record.json"))
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["command"]["config_hash"] = "sha256:" + "0" * 64
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    from llmtracefx.optimizer.lab.qwen3_8b.report import verify_control_evidence

    result = verify_control_evidence(manifest, workspace=workspace)
    assert result["verified"] is False
    assert any(
        "current manifest binding mismatch" in failure for failure in result["failures"]
    )


def test_tokenizer_requested_and_actual_token_counts_are_reported_separately(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=_in_process_launcher(manifest),
    )
    report = build_control_report(manifest, workspace=workspace)
    tier = report["tiers"][0]
    assert tier["requested_tokens"] == 2048
    assert tier["mean_actual_input_tokens"] == 5.0
    assert tier["requested_tokens"] != tier["mean_actual_input_tokens"]


# ---------------------------------------------------------------------------
# Reports: differs-from-27B language, determinism, privacy/integrity
# ---------------------------------------------------------------------------


def test_report_states_it_differs_from_27b_and_never_repeats_its_disclaimer(
    tmp_path, monkeypatch
) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=_in_process_launcher(manifest),
    )
    report = build_control_report(manifest, workspace=workspace)
    joined_limitations = " ".join(report["limitations"])
    assert "not the packaged Qwen3.8-27B lab" in joined_limitations
    assert "never directly comparable" in joined_limitations
    # The 27B report's inaccurate-for-us disclaimer must never appear here:
    # this control *does* cryptographically bind its source revision.
    assert "does not cryptographically bind" not in joined_limitations
    assert report["self_conversion"]["official_revision"] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )


def test_report_is_deterministic_sanitized_json_and_html(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "private-user-name" / "workspace"
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=_in_process_launcher(manifest),
    )
    report = build_control_report(manifest, workspace=workspace)
    from llmtracefx.optimizer.lab.core import render_lab_report_html

    first = render_lab_report_html(report)
    second = render_lab_report_html(report)
    assert first == second
    assert "<h1>M5 Pro × Qwen3-8B</h1>" in first
    assert "Runtime: mlx-lm 0.31.3" in first
    assert "Qwen3.8-27B</h1>" not in first
    assert "Requested tokens" in first
    assert "Mean actual tokens" in first
    assert "Observed MLX active memory" in first
    assert "Observed MLX cache memory" in first
    assert str(tmp_path) not in json.dumps(report)
    assert str(tmp_path) not in first

    shareable = tmp_path / "shareable"
    written = write_control_reports(
        manifest, workspace=workspace, shareable_dir=shareable
    )
    assert written == report
    assert (workspace / "reports" / "tune-report.json").is_file()
    assert (workspace / "reports" / "compare-report.json").is_file()
    assert (shareable / "evidence-summary.json").is_file()
    assert (shareable / "report.html").read_text(encoding="utf-8") == first


def test_write_control_reports_refuses_tampered_evidence(tmp_path, monkeypatch) -> None:
    manifest, manifest_path = _bound_manifest(tmp_path)
    _prepare(monkeypatch)
    (tmp_path / "model").mkdir(exist_ok=True)
    workspace = tmp_path / "workspace"
    bm.run_benchmark(
        manifest,
        manifest_path=manifest_path,
        workspace=workspace,
        model_path=tmp_path / "model",
        max_tier="2k",
        launcher=_in_process_launcher(manifest),
    )
    record_path = next((workspace / "results" / "runs").glob("*/final_record.json"))
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["command"]["config_hash"] = "sha256:" + "1" * 64
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="unverified current-run evidence"):
        write_control_reports(manifest, workspace=workspace)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_default_action_is_no_download_plan(monkeypatch, capsys) -> None:
    from llmtracefx.optimizer.lab.qwen3_8b.conversion import ConversionSafetyDecision

    monkeypatch.setattr(
        qcli,
        "assess_conversion_safety",
        lambda *args, **kwargs: ConversionSafetyDecision(
            safe=True,
            blockers=(),
            snapshot=_safe_decision().snapshot,
            required_free_disk_bytes=128395428637,
        ),
    )
    assert qcli.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["no_spend"] is True
    assert payload["downloads_performed"] is False
    assert payload["conversion_manifest"].startswith("package:")
    assert payload["required_free_disk_bytes"] == 128395428637


def test_cli_bind_materializes_manifest_from_receipt(tmp_path) -> None:
    receipt = _fabricated_receipt()
    receipt_path = tmp_path / "conversion-receipt.json"
    receipt_path.write_text(json.dumps(receipt.to_dict()), encoding="utf-8")
    output_path = tmp_path / "control-manifest.bound.json"
    exit_code = qcli.main(
        ["bind", "--receipt", str(receipt_path), "--output", str(output_path)]
    )
    assert exit_code == 0
    bound = LabManifest.read_json(output_path)
    assert bound.model.model_family == "qwen3"
    assert len(bound.model.revision) == 40


def test_cli_run_requires_manifest_or_receipt(tmp_path) -> None:
    exit_code = qcli.main(["run", "--workspace", str(tmp_path / "ws")])
    assert exit_code == 1


def test_packaged_resources_load_from_external_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    conversion, conversion_source = qcli._load_conversion_manifest(None)
    template, template_source = qcli._load_control_template(None)
    assert conversion_source.startswith("package:llmtracefx.optimizer.lab.qwen3_8b/")
    assert template_source.startswith("package:llmtracefx.optimizer.lab.qwen3_8b/")
    assert conversion.source.official_id == "Qwen/Qwen3-8B"
    assert template.to_dict()["model"]["model_family"] == "qwen3"
