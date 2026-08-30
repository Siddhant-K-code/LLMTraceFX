"""Tests for the workload-matrix verification pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.schema import ExperimentRecord
from llmtracefx.optimizer.workloads.catalog import (
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    MatrixEntry,
    MatrixManifest,
    generate_matrix,
    write_matrix,
)
from llmtracefx.optimizer.workloads.schema import ContextTier
from llmtracefx.optimizer.workloads.verify import (
    RowSelection,
    RowStatus,
    RunBinding,
    VerifyError,
    execute_row,
    plan_row,
    plan_selected_rows,
    run_selected_rows,
    select_entries,
)

GOOD_RESPONSE = '{"name": "Priya", "age": 34, "is_active": true}'
BAD_RESPONSE = "not json at all"


@dataclass
class FakeResponse:
    text: str
    from_draft: bool = False
    prompt_tokens: int = 3
    generation_tokens: int = 1
    finish_reason: str | None = None


class FakeTokenizer:
    bos_token = None


class FakeMLXRuntime:
    """Minimal injectable MLX runtime; no Apple hardware or MLX import needed."""

    mlx_version = "0.32.0"
    mlx_lm_version = "0.31.3"

    def __init__(self, response_text: str = GOOD_RESPONSE, *, fail: bool = False):
        self.response_text = response_text
        self.fail = fail
        self.load_calls: list[Path] = []
        self.generate_calls: list[dict] = []

    def load_model(self, path):
        self.load_calls.append(path)
        if self.fail:
            raise RuntimeError("simulated model load failure")
        return object(), FakeTokenizer()

    def encode(self, tokenizer, prompt):
        return [1, 2, 3]

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return MLXMemorySnapshot(active_bytes=1024, cache_bytes=256, peak_bytes=2048)

    def accelerator_name(self):
        return "Apple M5 Pro (test)"

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
        self.generate_calls.append(
            {"draft_model": draft_model, "num_draft_tokens": num_draft_tokens}
        )
        yield FakeResponse(self.response_text, generation_tokens=1)


def build_manifest(
    tmp_path: Path,
    *,
    target_model_path: Path | None = None,
    context_tiers=(ContextTier.TIER_2K,),
) -> tuple[MatrixManifest, Path]:
    """Generate+write a small (single workload) matrix for fast tests."""
    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        target_model_path=(str(target_model_path) if target_model_path else None),
        workloads=(STRUCTURED_JSON_PROFILE_EXTRACTION,),
        context_tiers=context_tiers,
        mtp_depths=(2,),
    )
    write_matrix(manifest)
    reloaded = MatrixManifest.read_json(output_dir / "manifest.json")
    return reloaded, output_dir


def make_target_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "model"
    model_path.mkdir()
    return model_path


# --- Row selection -----------------------------------------------------------


def test_select_entries_filters_by_decode_mode(tmp_path):
    manifest, _ = build_manifest(tmp_path)
    selection = RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE}))
    selected = select_entries(manifest, selection)
    assert selected
    assert all(e.decode_mode == DECODE_MODE_AUTOREGRESSIVE for e in selected)


def test_select_entries_filters_by_run_id(tmp_path):
    manifest, _ = build_manifest(tmp_path)
    target = manifest.entries[0].run_id
    selected = select_entries(manifest, RowSelection(run_ids=frozenset({target})))
    assert len(selected) == 1
    assert selected[0].run_id == target


def test_select_entries_filters_by_category_and_context_tier(tmp_path):
    manifest, _ = build_manifest(tmp_path)
    selection = RowSelection(
        categories=frozenset({"structured_json"}), context_tiers=frozenset({"2k"})
    )
    selected = select_entries(manifest, selection)
    assert selected
    assert all(
        e.category == "structured_json" and e.context_tier == "2k" for e in selected
    )


def test_select_entries_empty_selection_matches_everything(tmp_path):
    manifest, _ = build_manifest(tmp_path)
    assert select_entries(manifest, RowSelection()) == manifest.entries


# --- Path binding / no-download invariants -----------------------------------


def test_run_binding_rejects_missing_target_model_path(tmp_path):
    with pytest.raises(VerifyError, match="never downloaded"):
        RunBinding(target_model_path=tmp_path / "missing-model")


def test_run_binding_rejects_missing_draft_model_path(tmp_path):
    target = make_target_model(tmp_path)
    with pytest.raises(VerifyError, match="never downloaded"):
        RunBinding(
            target_model_path=target, draft_model_path=tmp_path / "missing-draft"
        )


def test_run_binding_accepts_existing_paths(tmp_path):
    target = make_target_model(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    binding = RunBinding(target_model_path=target, draft_model_path=draft)
    assert binding.target_model_path == target
    assert binding.draft_model_path == draft


@pytest.mark.parametrize("value", [True, 0, -1])
def test_run_binding_rejects_invalid_num_draft_tokens(tmp_path, value):
    target = make_target_model(tmp_path)
    with pytest.raises(VerifyError, match="positive integer"):
        RunBinding(target_model_path=target, num_draft_tokens=value)


# --- Unsupported (native-mtp) rows -------------------------------------------


def test_native_mtp_rows_are_rejected_explicitly_never_executed(tmp_path):
    manifest, output_dir = build_manifest(tmp_path, target_model_path=None)
    target = make_target_model(tmp_path)
    entry = next(e for e in manifest.entries if e.decode_mode == DECODE_MODE_NATIVE_MTP)
    runtime = FakeMLXRuntime()

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.UNSUPPORTED
    assert result.verification.reason
    assert result.final_record is None
    assert runtime.load_calls == []  # never touched the model


def test_native_mtp_only_selection_never_constructs_a_runtime(tmp_path):
    """A batch with only unsupported rows must never invoke the runtime
    factory at all -- not even to construct it -- so an environment
    without MLX installed can still report native-mtp rows as
    unsupported instead of crashing."""
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)

    def _factory_that_must_not_be_called():
        raise AssertionError("runtime factory must not be called for unsupported rows")

    results = run_selected_rows(
        manifest,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_NATIVE_MTP})),
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=_factory_that_must_not_be_called,
    )

    assert results
    assert all(r.verification.status == RowStatus.UNSUPPORTED for r in results)


def test_unavailable_mlx_runtime_fails_the_row_cleanly(tmp_path):
    """If the runtime factory itself raises (e.g. MLX not installed), the
    row must fail cleanly with an explicit reason rather than crashing
    the whole batch."""
    from llmtracefx.optimizer.collectors.mlx import MLXCollectorError

    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )

    def _unavailable_runtime():
        raise MLXCollectorError("MLX collection requires Apple Silicon running macOS")

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=_unavailable_runtime,
    )

    assert result.verification.status == RowStatus.FAILED
    assert "MLX runtime is unavailable" in result.verification.reason


def test_native_mtp_rows_not_downgraded_even_with_draft_model_path(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    draft = tmp_path / "draft"
    draft.mkdir()
    entry = next(e for e in manifest.entries if e.decode_mode == DECODE_MODE_NATIVE_MTP)
    runtime = FakeMLXRuntime()

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target, draft_model_path=draft),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.UNSUPPORTED
    assert runtime.generate_calls == []


# --- Prompt hash verification -------------------------------------------------


def test_local_run_id_cannot_escape_output_directory(tmp_path):
    manifest, manifest_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    original = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    payload = original.to_dict()
    payload["run_id"] = "../escaped"
    entry = MatrixEntry.from_dict(payload)
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)

    plan = plan_row(
        entry,
        manifest_dir=manifest_dir,
        output_dir=results_dir,
        binding=binding,
    )
    runtime = FakeMLXRuntime()
    result = execute_row(
        entry,
        manifest_dir=manifest_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert not plan.ready
    assert "unsafe artifact path" in plan.blockers[0]
    assert result.verification.status is RowStatus.FAILED
    assert "unsafe artifact path" in (result.verification.reason or "")
    assert not results_dir.exists()
    assert not (tmp_path / "escaped").exists()
    assert runtime.load_calls == []


@pytest.mark.parametrize("corruption", ["invalid-path", "oversized", "symlink"])
def test_local_prompt_must_be_a_bounded_regular_file(tmp_path, monkeypatch, corruption):
    manifest, manifest_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    prompt_path = Path(entry.prompt_path)
    if corruption == "invalid-path":
        payload = entry.to_dict()
        payload["prompt_path"] = "\0"
        entry = MatrixEntry.from_dict(payload)
    elif corruption == "oversized":
        monkeypatch.setattr(
            "llmtracefx.optimizer.workloads.verify.MAX_EVIDENCE_ARTIFACT_BYTES", 8
        )
    else:
        target_path = tmp_path / "prompt-target.txt"
        target_path.write_text(
            prompt_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        prompt_path.unlink()
        prompt_path.symlink_to(target_path)
    binding = RunBinding(target_model_path=target)

    plan = plan_row(
        entry,
        manifest_dir=manifest_dir,
        output_dir=tmp_path / "results",
        binding=binding,
    )
    runtime = FakeMLXRuntime()
    result = execute_row(
        entry,
        manifest_dir=manifest_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert not plan.ready
    assert "prompt file unreadable" in plan.blockers[0]
    assert result.verification.status is RowStatus.FAILED
    assert "prompt file unreadable" in (result.verification.reason or "")
    assert runtime.load_calls == []


def test_prompt_hash_mismatch_fails_the_row_without_executing(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    Path(entry.prompt_path).write_text(
        "this text was edited after matrix generation", encoding="utf-8"
    )
    runtime = FakeMLXRuntime()

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.FAILED
    assert "prompt hash mismatch" in result.verification.reason
    assert runtime.load_calls == []


def test_prompt_hash_match_allows_execution(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(GOOD_RESPONSE)

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


# --- Evaluator success/failure ------------------------------------------------


def test_completed_row_reflects_correct_evaluator_outcome(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(GOOD_RESPONSE)

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert result.final_record.outcome.success is True
    assert result.final_record.outcome.quality_score == 1.0
    assert result.verification.outcome_success is True


def test_completed_row_reflects_incorrect_evaluator_outcome(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(BAD_RESPONSE)

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert result.final_record.outcome.success is False
    assert result.final_record.outcome.quality_score == 0.0


# --- Runtime failure preservation ---------------------------------------------


def test_runtime_failure_is_preserved_and_not_overwritten_by_evaluator(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(fail=True)

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.FAILED
    assert result.final_record is not None
    assert result.final_record.outcome.success is False
    assert result.final_record.error is not None
    assert result.final_record.error.category == "RuntimeError"
    assert "simulated model load failure" in result.verification.reason
    final_record_path = Path(result.verification.final_record_path)
    assert final_record_path.exists()
    persisted = ExperimentRecord.read_json(final_record_path)
    assert persisted.error is not None
    assert persisted.error.category == "RuntimeError"
    verification_path = (
        tmp_path / "results" / "runs" / entry.run_id / "verification.json"
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["final_record_path"] == str(final_record_path)


# --- Inconclusive evaluator errors --------------------------------------------


def test_evaluator_error_produces_inconclusive_status(tmp_path, monkeypatch):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(GOOD_RESPONSE)

    def _boom(workload, response_text):
        raise OSError("evaluator subprocess could not start")

    monkeypatch.setattr(
        "llmtracefx.optimizer.workloads.verify.evaluate_workload", _boom
    )

    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.INCONCLUSIVE
    assert result.final_record.outcome.quality_score is None
    assert result.final_record.outcome.success is True  # runtime itself succeeded
    assert "inconclusive" in result.final_record.outcome.notes


# --- Atomic artifacts ----------------------------------------------------------


def test_completed_row_persists_all_expected_artifacts(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    results_dir = tmp_path / "results"

    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: runtime,
    )

    run_dir = results_dir / "runs" / entry.run_id
    assert (run_dir / "collection" / "record.json").exists()
    assert (run_dir / "collection" / "response.txt").exists()
    assert (run_dir / "collection" / "environment.json").exists()
    assert (run_dir / "final_record.json").exists()
    assert (run_dir / "verification.json").exists()

    final_record = ExperimentRecord.read_json(run_dir / "final_record.json")
    assert final_record.outcome.success is True

    verification_payload = json.loads((run_dir / "verification.json").read_text())
    assert verification_payload["status"] == "completed"
    assert verification_payload["recorded_prompt_hash"] == entry.prompt.prompt_hash


# --- Resume / staleness detection ----------------------------------------------


def test_resume_trusts_hash_matching_completed_artifact(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)

    first_runtime = FakeMLXRuntime(GOOD_RESPONSE)
    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: first_runtime,
    )
    assert first_runtime.load_calls == [target]

    second_runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: second_runtime,
    )

    assert result.verification.status == RowStatus.SKIPPED
    assert result.verification.resumed is True
    assert second_runtime.load_calls == []  # not re-executed
    assert result.final_record.outcome.success is True


@pytest.mark.parametrize(
    "corruption",
    ["invalid-utf8", "truncated", "recursion", "oversized", "symlink"],
)
def test_resume_reruns_when_prior_verification_is_unreadable(
    tmp_path, corruption, monkeypatch
):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)
    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )
    verification_path = results_dir / "runs" / entry.run_id / "verification.json"
    if corruption == "invalid-utf8":
        verification_path.write_bytes(b"\xff")
    elif corruption == "truncated":
        verification_path.write_text("{", encoding="utf-8")
    elif corruption == "recursion":
        verification_path.write_text(
            "[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8"
        )
    elif corruption == "oversized":
        monkeypatch.setattr(
            "llmtracefx.optimizer.workloads.verify.MAX_METADATA_ARTIFACT_BYTES", 16
        )
        verification_path.write_bytes(b"x" * 17)
    else:
        target_path = verification_path.with_name("verification-target.json")
        target_path.write_text("{}", encoding="utf-8")
        verification_path.unlink()
        verification_path.symlink_to(target_path)

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


@pytest.mark.parametrize(
    "corruption",
    ["invalid-utf8", "truncated", "recursion", "oversized", "symlink"],
)
def test_resume_reruns_when_final_record_is_unreadable(
    tmp_path, corruption, monkeypatch
):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)
    first = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )
    final_record_path = Path(first.verification.final_record_path)
    if corruption == "invalid-utf8":
        final_record_path.write_bytes(b"\xff")
    elif corruption == "truncated":
        final_record_path.write_text("{", encoding="utf-8")
    elif corruption == "recursion":
        final_record_path.write_text(
            "[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8"
        )
    elif corruption == "oversized":
        monkeypatch.setattr(
            "llmtracefx.optimizer.schema.MAX_EVIDENCE_ARTIFACT_BYTES", 16
        )
        final_record_path.write_bytes(b"x" * 17)
    else:
        target_path = final_record_path.with_name("record-target.json")
        target_path.write_text("{}", encoding="utf-8")
        final_record_path.unlink()
        final_record_path.symlink_to(target_path)

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


@pytest.mark.parametrize("artifact", ["verification", "final-record"])
def test_resume_reruns_when_prior_artifact_has_another_run_id(tmp_path, artifact):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)
    first = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )
    if artifact == "verification":
        path = results_dir / "runs" / entry.run_id / "verification.json"
    else:
        path = Path(first.verification.final_record_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_id"] = "copied-from-another-row"
    path.write_text(json.dumps(payload), encoding="utf-8")

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


def test_resume_reruns_when_final_record_has_another_model_id(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)
    first = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )
    path = Path(first.verification.final_record_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["model_id"] = "copied-from-another-model"
    path.write_text(json.dumps(payload), encoding="utf-8")

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


def test_resume_reruns_when_final_record_has_non_finite_metric(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)
    first = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )
    path = Path(first.verification.final_record_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["timing"]["total"]["value"] = "NON_FINITE"
    path.write_text(
        json.dumps(payload).replace('"NON_FINITE"', "1e400"), encoding="utf-8"
    )

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


def test_resume_reruns_when_prompt_content_changes(tmp_path):
    """A stale hash-mismatched artifact must be re-run, not blindly trusted."""
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)

    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )

    # Regenerate the matrix with a different workload so the prompt content
    # (and its recorded hash) legitimately changes, simulating an edited
    # catalog/matrix; force the run_id to collide with the prior artifact
    # directory to exercise the resume-trust path against stale data.
    import dataclasses

    from llmtracefx.optimizer.workloads.catalog import CODE_COMPLETION_PALINDROME

    regenerated = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        workloads=(CODE_COMPLETION_PALINDROME,),
        context_tiers=(ContextTier.TIER_2K,),
        mtp_depths=(2,),
    )
    write_matrix(regenerated)
    reloaded = MatrixManifest.read_json(output_dir / "manifest.json")
    new_entry = next(
        e for e in reloaded.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    assert new_entry.prompt.prompt_hash != entry.prompt.prompt_hash
    colliding_entry = dataclasses.replace(new_entry, run_id=entry.run_id)

    runtime = FakeMLXRuntime("def is_palindrome(text): return True")
    execute_row(
        colliding_entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert runtime.load_calls == [target]  # re-executed, not trusted as stale


def test_resume_reruns_when_run_binding_changes(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"

    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target, seed=0),
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=RunBinding(target_model_path=target, seed=7),  # different seed
        resume=True,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]  # re-executed, binding changed


def test_no_resume_forces_reexecution(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    results_dir = tmp_path / "results"
    binding = RunBinding(target_model_path=target)

    execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )

    runtime = FakeMLXRuntime(GOOD_RESPONSE)
    result = execute_row(
        entry,
        manifest_dir=output_dir,
        output_dir=results_dir,
        model_id=manifest.model_id,
        binding=binding,
        resume=False,
        runtime_factory=lambda: runtime,
    )

    assert result.verification.status == RowStatus.COMPLETED
    assert runtime.load_calls == [target]


# --- Dry run planning ------------------------------------------------------


def test_plan_selected_rows_reports_ready_rows_without_touching_model(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)
    target = make_target_model(tmp_path)
    binding = RunBinding(target_model_path=target)

    plans = plan_selected_rows(
        manifest,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=binding,
    )

    assert len(plans) == 1
    assert plans[0].ready is True
    assert plans[0].unsupported is False
    assert plans[0].blockers == ()


def test_plan_selected_rows_flags_native_mtp_as_unsupported(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)

    plans = plan_selected_rows(
        manifest,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_NATIVE_MTP})),
        binding=None,
    )

    assert len(plans) == 1
    assert plans[0].unsupported is True
    assert plans[0].unsupported_reason


def test_plan_selected_rows_flags_missing_binding_as_blocker(tmp_path):
    manifest, output_dir = build_manifest(tmp_path)

    plans = plan_selected_rows(
        manifest,
        manifest_dir=output_dir,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=None,
    )

    assert len(plans) == 1
    assert plans[0].ready is False
    assert any("--model-path" in blocker for blocker in plans[0].blockers)


# --- End-to-end fake matrix run -------------------------------------------------


def test_run_selected_rows_end_to_end_over_full_matrix(tmp_path):
    manifest, output_dir = build_manifest(
        tmp_path, context_tiers=(ContextTier.TIER_2K, ContextTier.TIER_8K)
    )
    target = make_target_model(tmp_path)
    results_dir = tmp_path / "results"

    results = run_selected_rows(
        manifest,
        manifest_dir=output_dir,
        output_dir=results_dir,
        selection=RowSelection(),
        binding=RunBinding(target_model_path=target),
        resume=True,
        runtime_factory=lambda: FakeMLXRuntime(GOOD_RESPONSE),
    )

    assert len(results) == len(manifest.entries)
    statuses = {r.verification.status for r in results}
    assert RowStatus.COMPLETED in statuses
    assert RowStatus.UNSUPPORTED in statuses
    for result in results:
        assert (
            results_dir / "runs" / result.entry.run_id / "verification.json"
        ).exists()
