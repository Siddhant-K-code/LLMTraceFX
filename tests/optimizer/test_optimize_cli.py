"""End-to-end tests for the `llmtracefx-optimizer optimize` CLI subcommand.

`optimize` composes `workloads run`, `tune`, and `tune-report` exactly as
they already behave: every test here exercises the real
``run_selected_rows``/``tune``/``render_tune_report_html`` code paths
(never mocking the top-level `_cmd_optimize` function itself), faking only
the outermost MLX runtime boundary the same way `test_workloads_cli.py`
already does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from _tune_fixtures import write_run

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.optimize_summary import OptimizeSummary
from llmtracefx.optimizer.tune.report import TuneReport
from llmtracefx.optimizer.workloads.catalog import (
    CODE_COMPLETION_PALINDROME,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.matrix import generate_matrix, write_matrix
from llmtracefx.optimizer.workloads.schema import ContextTier


@dataclass
class _FakeResponse:
    text: str = '{"name": "Priya", "age": 34, "is_active": true}'
    from_draft: bool = False
    prompt_tokens: int = 3
    generation_tokens: int = 1
    finish_reason: str | None = None


class _FakeTokenizer:
    bos_token = None


class _FakeMLXRuntime:
    mlx_version = "0.32.0"
    mlx_lm_version = "0.31.3"

    def __init__(self, response_text: str | None = None) -> None:
        self.response_text = response_text or _FakeResponse.text

    def load_model(self, path):
        return object(), _FakeTokenizer()

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
        yield _FakeResponse(self.response_text)


class _FailingRuntime(_FakeMLXRuntime):
    def load_model(self, path):
        raise RuntimeError("boom")


def _sequenced_runtime_factory(*runtimes):
    iterator = iter(runtimes)

    def factory():
        return next(iterator)

    return factory


def _never_call_runtime_factory():
    def factory():
        raise AssertionError(
            "MLXLMRuntime must never be constructed for this invocation"
        )

    return factory


def _never_call_phase(*args, **kwargs):
    raise AssertionError("tune and render phases must not run during dry-run")


def _build_matrix(tmp_path, workloads, *, context_tiers=(ContextTier.TIER_2K,)):
    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        workloads=workloads,
        context_tiers=context_tiers,
        mtp_depths=(2,),
    )
    write_matrix(manifest)
    return output_dir / "manifest.json"


def _write_policy(tmp_path, payload, name="policy.json"):
    policy_path = tmp_path / name
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    return policy_path


def _run_optimize(argv):
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _base_argv(
    *,
    matrix_path,
    results_dir,
    policy_path,
    report_json,
    model_path=None,
    report_html=None,
    extra=(),
):
    argv = [
        "optimize",
        "--matrix",
        str(matrix_path),
        "--results",
        str(results_dir),
        "--policy",
        str(policy_path),
    ]
    if model_path is not None:
        argv += ["--model-path", str(model_path)]
    if report_json is not None:
        argv += ["--report-json", str(report_json)]
    if report_html is not None:
        argv += ["--report-html", str(report_html)]
    return argv + list(extra)


# --- Full success ------------------------------------------------------


def test_optimize_full_success_writes_json_html_and_summary(
    tmp_path, monkeypatch, capsys
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    report_json = tmp_path / "report.json"
    report_html = tmp_path / "report.html"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=report_json,
            model_path=model_path,
            report_html=report_html,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 0
    assert report_json.exists()
    assert report_html.exists()

    tune_report = TuneReport.read_json(report_json)
    assert tune_report.has_recommendation

    summary_path = results_dir / "optimize_summary.json"
    assert summary_path.exists()
    summary = OptimizeSummary.read_json(summary_path)
    assert summary.overall_status.value == "success"
    assert summary.exit_code == 0
    assert [phase.status.value for phase in summary.phases] == [
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    assert summary.row_counts.completed == 1
    assert len(summary.recommendations) == 1

    out = capsys.readouterr().out
    assert "Tune report JSON written" in out
    assert "Tune report HTML written" in out
    assert "Orchestration summary written" in out


def test_optimize_default_redacts_paths_html_include_paths_opt_in(
    tmp_path, monkeypatch
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    redacted_results = tmp_path / "results-redacted"
    redacted_html = tmp_path / "redacted.html"
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=redacted_results,
            policy_path=policy_path,
            report_json=tmp_path / "redacted.json",
            model_path=model_path,
            report_html=redacted_html,
            extra=["--mode", "autoregressive"],
        )
    )
    assert exit_code == 0
    redacted_text = redacted_html.read_text(encoding="utf-8")
    assert str(model_path) not in redacted_text
    assert str(redacted_results) not in redacted_text

    included_results = tmp_path / "results-included"
    included_html = tmp_path / "included.html"
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=included_results,
            policy_path=policy_path,
            report_json=tmp_path / "included.json",
            model_path=model_path,
            report_html=included_html,
            extra=["--mode", "autoregressive", "--include-paths"],
        )
    )
    assert exit_code == 0
    included_text = included_html.read_text(encoding="utf-8")
    assert str(included_results) in included_text


# --- Dry run -------------------------------------------------------------


def test_optimize_dry_run_never_constructs_runtime_or_writes_reports(
    tmp_path, monkeypatch
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    report_json = tmp_path / "report.json"
    report_html = tmp_path / "report.html"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())
    monkeypatch.setattr(cli, "tune", _never_call_phase)
    monkeypatch.setattr(cli, "render_tune_report_html", _never_call_phase)

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=report_json,
            model_path=model_path,
            report_html=report_html,
            extra=["--mode", "autoregressive", "--dry-run"],
        )
    )

    assert exit_code == 0  # ready: valid --model-path given, nothing blocked
    assert not (results_dir / "runs").exists()
    assert not report_json.exists()
    assert not report_html.exists()
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.dry_run is True
    assert summary.exit_code == 0
    assert summary.row_counts.total == 1
    assert summary.row_counts.ready == 1
    assert summary.row_counts.blocked == 0
    assert summary.row_counts.unsupported == 0
    assert summary.phase(cli.PhaseName.PLANNED).status.value == "ok"
    for phase_name in (
        cli.PhaseName.EXECUTED,
        cli.PhaseName.VERIFIED,
        cli.PhaseName.TUNED,
        cli.PhaseName.RENDERED,
    ):
        assert summary.phase(phase_name).status.value == "not_run"


def test_optimize_dry_run_reports_blockers_without_model_path(tmp_path, capsys):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            extra=["--mode", "autoregressive", "--dry-run"],
        )
    )

    assert exit_code == 2
    out = capsys.readouterr().out
    assert "BLOCKED" in out
    assert "no model was loaded, no results were tuned" not in out  # printed once
    assert "no tune report was written" in out
    summary = OptimizeSummary.read_json(tmp_path / "results" / "optimize_summary.json")
    assert summary.dry_run is True
    assert summary.exit_code == 2
    assert summary.overall_status.value == "inconclusive"
    assert summary.row_counts.total == 1
    assert summary.row_counts.ready == 0
    assert summary.row_counts.blocked == 1
    assert summary.row_counts.unsupported == 0


def test_optimize_dry_run_reports_unsupported_rows(tmp_path, capsys):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            extra=["--mode", "native-mtp", "--dry-run"],
        )
    )

    assert exit_code == 0  # unsupported rows never block
    out = capsys.readouterr().out
    assert "UNSUPPORTED" in out
    summary = OptimizeSummary.read_json(tmp_path / "results" / "optimize_summary.json")
    assert summary.row_counts.total == 1
    assert summary.row_counts.ready == 0
    assert summary.row_counts.blocked == 0
    assert summary.row_counts.unsupported == 1
    assert summary.phase(cli.PhaseName.EXECUTED).status.value == "not_run"


def test_optimize_dry_run_writes_custom_summary_path(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    summary_path = tmp_path / "custom-summary.json"

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=[
                "--mode",
                "autoregressive",
                "--dry-run",
                "--summary-json",
                str(summary_path),
            ],
        )
    )

    assert exit_code == 0
    assert OptimizeSummary.read_json(summary_path).dry_run is True


def test_optimize_dry_run_rejects_invalid_num_draft_tokens(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=[
                "--mode",
                "autoregressive",
                "--dry-run",
                "--num-draft-tokens",
                "0",
            ],
        )
    )

    assert exit_code == 2
    summary = OptimizeSummary.read_json(tmp_path / "results" / "optimize_summary.json")
    assert summary.row_counts.ready == 0
    assert summary.row_counts.blocked == 1


# --- Execution failure precedence ----------------------------------------


def test_optimize_all_rows_failed_exits_1(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FailingRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 1
    summary = OptimizeSummary.read_json(tmp_path / "results" / "optimize_summary.json")
    assert summary.exit_code == 1
    assert summary.overall_status.value == "failed"
    assert summary.phase(cli.PhaseName.EXECUTED).status.value == "failed"
    assert summary.row_counts.failed == 1


def test_optimize_partial_failure_with_surviving_recommendation_retains_failure_exit(
    tmp_path, monkeypatch
):
    # Two independent workloads (=> two independent tune groups): the first
    # fails at execution, the second succeeds and is recommended. The
    # overall command must still fail even though a recommendation exists.
    matrix_path = _build_matrix(
        tmp_path, (CODE_COMPLETION_PALINDROME, STRUCTURED_JSON_PROFILE_EXTRACTION)
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    report_json = tmp_path / "report.json"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(
        cli,
        "MLXLMRuntime",
        _sequenced_runtime_factory(_FailingRuntime(), _FakeMLXRuntime()),
    )

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=report_json,
            model_path=model_path,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 1
    tune_report = TuneReport.read_json(report_json)
    assert tune_report.has_recommendation  # the surviving group did recommend

    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.exit_code == 1
    assert summary.overall_status.value == "failed"
    assert summary.row_counts.failed == 1
    assert summary.row_counts.completed == 1
    assert len(summary.recommendations) == 1  # tuned eligible successful evidence


def test_optimize_all_rows_unsupported_exits_1_consistently_with_tune(
    tmp_path, monkeypatch
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "native-mtp"],
        )
    )

    # Execution itself is a clean no-op (never touches MLX, no row failed),
    # but there is zero tunable evidence: `tune` itself would report zero
    # comparable groups (exit 1), so `optimize` mirrors that exactly.
    assert exit_code == 1
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.phase(cli.PhaseName.EXECUTED).status.value == "ok"
    assert summary.phase(cli.PhaseName.TUNED).status.value == "failed"
    assert summary.row_counts.unsupported == 1
    assert summary.row_counts.failed == 0


def test_optimize_ignores_stale_unselected_primary_results(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    write_run(results_dir, "stale-successful-run")
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "native-mtp"],
        )
    )

    assert exit_code == 1
    report = TuneReport.read_json(tmp_path / "report.json")
    assert report.groups == ()
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.recommendations == ()


def test_optimize_all_rows_inconclusive_exits_2(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    def _raising_evaluate_workload(workload, response_text):
        raise RuntimeError("evaluator exploded")

    import llmtracefx.optimizer.workloads.verify as verify_module

    monkeypatch.setattr(verify_module, "evaluate_workload", _raising_evaluate_workload)
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 2
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.overall_status.value == "inconclusive"
    assert summary.phase(cli.PhaseName.EXECUTED).status.value == "inconclusive"
    assert summary.phase(cli.PhaseName.TUNED).status.value == "inconclusive"
    assert summary.row_counts.inconclusive == 1


# --- Setup/config errors ---------------------------------------------------


def _assert_optimize_path_collision(
    tmp_path,
    monkeypatch,
    *,
    collision_flag,
    collision_path,
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    report_json = tmp_path / "report.json"
    report_html = tmp_path / "report.html"
    summary_json = tmp_path / "summary.json"
    paths = {
        "--report-json": report_json,
        "--report-html": report_html,
        "--summary-json": summary_json,
    }
    paths[collision_flag] = collision_path(
        matrix_path=matrix_path,
        policy_path=policy_path,
        report_json=report_json,
        report_html=report_html,
        summary_json=summary_json,
    )
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())
    policy_before = policy_path.read_bytes()
    matrix_before = matrix_path.read_bytes()

    argv = _base_argv(
        matrix_path=matrix_path,
        results_dir=results_dir,
        policy_path=policy_path,
        report_json=paths["--report-json"],
        model_path=model_path,
        report_html=paths["--report-html"],
        extra=["--summary-json", str(paths["--summary-json"])],
    )
    exit_code = _run_optimize(argv)

    assert exit_code == 1
    assert policy_path.read_bytes() == policy_before
    assert matrix_path.read_bytes() == matrix_before
    assert not results_dir.exists()


def test_optimize_rejects_report_json_report_html_collision(tmp_path, monkeypatch):
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--report-html",
        collision_path=lambda **paths: paths["report_json"],
    )


def test_optimize_rejects_report_json_summary_collision(tmp_path, monkeypatch):
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--summary-json",
        collision_path=lambda **paths: paths["report_json"],
    )


def test_optimize_rejects_report_json_default_summary_collision(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=results_dir / "optimize_summary.json",
            model_path=model_path,
        )
    )

    assert exit_code == 1
    assert not results_dir.exists()


def test_optimize_rejects_policy_report_collision(tmp_path, monkeypatch):
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--report-json",
        collision_path=lambda **paths: paths["policy_path"],
    )


def test_optimize_rejects_matrix_summary_collision(tmp_path, monkeypatch):
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--summary-json",
        collision_path=lambda **paths: paths["matrix_path"],
    )


def test_optimize_rejects_relative_path_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--report-html",
        collision_path=lambda **paths: tmp_path
        / "subdir"
        / ".."
        / paths["report_json"].name,
    )


def test_optimize_rejects_symlink_path_aliases(tmp_path, monkeypatch):
    report_target = tmp_path / "report.json"
    report_alias = tmp_path / "report-alias.json"
    try:
        report_alias.symlink_to(report_target)
    except OSError:
        pytest.skip("symlinks are not available")
    _assert_optimize_path_collision(
        tmp_path,
        monkeypatch,
        collision_flag="--report-html",
        collision_path=lambda **paths: report_alias,
    )


def test_optimize_rejects_case_insensitive_path_aliases(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    if not cli._filesystem_is_case_insensitive(matrix_path):
        pytest.skip("filesystem is case-sensitive")
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=matrix_path.with_name(matrix_path.name.swapcase()),
            model_path=model_path,
        )
    )

    assert exit_code == 1


def test_optimize_rejects_selected_run_artifact_collision(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    run_id = "structured-json-profile-extraction-2k-autoregressive"
    final_record_path = results_dir / "runs" / run_id / "final_record.json"
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--summary-json", str(final_record_path)],
        )
    )

    assert exit_code == 1
    assert not results_dir.exists()


def test_optimize_invalid_policy_fails_before_any_execution(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
        )
    )

    assert exit_code == 1
    assert not results_dir.exists()


def test_optimize_missing_policy_fails_without_traceback(tmp_path, monkeypatch, capsys):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=tmp_path / "missing-policy.json",
            report_json=tmp_path / "report.json",
            model_path=model_path,
        )
    )

    assert exit_code == 1
    assert "Invalid tune policy" in capsys.readouterr().err
    assert not results_dir.exists()


def test_optimize_invalid_matrix_manifest_exits_1(tmp_path):
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=tmp_path / "missing-manifest.json",
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=tmp_path,
        )
    )
    assert exit_code == 1


def test_optimize_non_utf8_matrix_manifest_exits_1(tmp_path, capsys):
    matrix_path = tmp_path / "manifest.json"
    matrix_path.write_bytes(b"\xff\xfe not utf-8")
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=tmp_path,
        )
    )

    assert exit_code == 1
    assert "Failed to load matrix manifest" in capsys.readouterr().err


def test_optimize_no_rows_selected_exits_1(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--run-id", "does-not-exist"],
        )
    )
    assert exit_code == 1


def test_optimize_requires_model_path_unless_dry_run(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
        )
    )
    assert exit_code == 1


def test_optimize_requires_report_json_unless_dry_run(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=None,
            model_path=model_path,
        )
    )
    assert exit_code == 1


def test_optimize_invalid_model_path_binding_exits_1(tmp_path):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=tmp_path / "results",
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=tmp_path / "does-not-exist",
        )
    )
    assert exit_code == 1


# --- Report write/read failures -------------------------------------------


def test_optimize_report_json_write_failure_exits_1(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    # A directory at the report-json path makes the atomic write fail with
    # an explicit OSError instead of silently succeeding.
    report_json = tmp_path / "report-json-is-a-dir"
    report_json.mkdir()
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=report_json,
            model_path=model_path,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 1
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.phase(cli.PhaseName.RENDERED).status.value == "failed"


def test_optimize_report_html_write_failure_exits_1(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    report_html = tmp_path / "report-html-is-a-dir"
    report_html.mkdir()
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            report_html=report_html,
            extra=["--mode", "autoregressive"],
        )
    )

    assert exit_code == 1
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.phase(cli.PhaseName.RENDERED).status.value == "failed"
    # The JSON report itself was still written and validated successfully.
    assert (tmp_path / "report.json").exists()


def test_optimize_summary_write_failure_forces_exit_1(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    summary_path = tmp_path / "summary-is-a-dir"
    summary_path.mkdir()

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "autoregressive", "--summary-json", str(summary_path)],
        )
    )

    # Even though execution/tuning/rendering all otherwise succeeded, an
    # untrustworthy orchestration summary must force a hard failure.
    assert exit_code == 1


def test_optimize_invalid_tuning_evidence_conflict_exits_1(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    # An --extra-results directory that conflicts (same run_id, different
    # content) with what this invocation is about to produce.
    conflicting_dir = tmp_path / "conflicting"
    write_run(
        conflicting_dir,
        "structured-json-profile-extraction-2k-autoregressive",
        total_ms=99999.0,
    )

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=[
                "--mode",
                "autoregressive",
                "--extra-results",
                str(conflicting_dir),
            ],
        )
    )

    assert exit_code == 1
    assert not (tmp_path / "report.json").exists()
    summary = OptimizeSummary.read_json(results_dir / "optimize_summary.json")
    assert summary.phase(cli.PhaseName.TUNED).status.value == "failed"
    assert summary.phase(cli.PhaseName.RENDERED).status.value == "skipped"


# --- Opt-in extra-results (requirement 5) ---------------------------------


def test_optimize_only_tunes_primary_results_dir_by_default(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    # An unrelated historical results directory with a *different* run_id
    # so it never conflicts; only used to prove it's excluded by default.
    unrelated_dir = tmp_path / "unrelated-history"
    write_run(unrelated_dir, "some-other-historical-run")

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "autoregressive"],
        )
    )
    assert exit_code == 0
    report = TuneReport.read_json(tmp_path / "report.json")
    assert str(unrelated_dir) not in report.results_dirs


def test_optimize_extra_results_opt_in_includes_additional_directory(
    tmp_path, monkeypatch
):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    extra_dir = tmp_path / "extra-history"
    write_run(extra_dir, "some-other-historical-run")

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=[
                "--mode",
                "autoregressive",
                "--extra-results",
                str(extra_dir),
            ],
        )
    )
    assert exit_code == 0
    report = TuneReport.read_json(tmp_path / "report.json")
    assert str(extra_dir) in report.results_dirs


# --- Resume (requirement 4) ------------------------------------------------


def test_optimize_resume_reuses_pr6_hash_matching_artifacts(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    first_argv = _base_argv(
        matrix_path=matrix_path,
        results_dir=results_dir,
        policy_path=policy_path,
        report_json=tmp_path / "report-1.json",
        model_path=model_path,
        extra=["--mode", "autoregressive"],
    )
    assert _run_optimize(first_argv) == 0

    run_dir = next((results_dir / "runs").iterdir())
    first_verification = json.loads((run_dir / "verification.json").read_text())
    assert first_verification["status"] == "completed"

    # A runtime that would fail if it were ever invoked again: resume must
    # trust the hash-matching prior artifact instead of re-executing.
    monkeypatch.setattr(cli, "MLXLMRuntime", _never_call_runtime_factory())
    second_argv = _base_argv(
        matrix_path=matrix_path,
        results_dir=results_dir,
        policy_path=policy_path,
        report_json=tmp_path / "report-2.json",
        model_path=model_path,
        extra=["--mode", "autoregressive"],
    )
    assert _run_optimize(second_argv) == 0
    second_verification = json.loads((run_dir / "verification.json").read_text())
    assert second_verification["status"] == "skipped"
    assert second_verification["resumed"] is True


def test_optimize_no_resume_forces_re_execution(tmp_path, monkeypatch):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    argv = _base_argv(
        matrix_path=matrix_path,
        results_dir=results_dir,
        policy_path=policy_path,
        report_json=tmp_path / "report-1.json",
        model_path=model_path,
        extra=["--mode", "autoregressive"],
    )
    assert _run_optimize(argv) == 0

    calls = {"count": 0}

    def _counting_factory():
        calls["count"] += 1
        return _FakeMLXRuntime()

    monkeypatch.setattr(cli, "MLXLMRuntime", _counting_factory)
    no_resume_argv = _base_argv(
        matrix_path=matrix_path,
        results_dir=results_dir,
        policy_path=policy_path,
        report_json=tmp_path / "report-2.json",
        model_path=model_path,
        extra=["--mode", "autoregressive", "--no-resume"],
    )
    assert _run_optimize(no_resume_argv) == 0
    assert calls["count"] == 1


# --- Explain / row status printing ----------------------------------------


def test_optimize_explain_flag_forwards_to_tune_explain(tmp_path, monkeypatch, capsys):
    matrix_path = _build_matrix(tmp_path, (STRUCTURED_JSON_PROFILE_EXTRACTION,))
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    policy_path = _write_policy(
        tmp_path,
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {"max_total_latency_ms": 0.0001},
        },
    )
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    exit_code = _run_optimize(
        _base_argv(
            matrix_path=matrix_path,
            results_dir=results_dir,
            policy_path=policy_path,
            report_json=tmp_path / "report.json",
            model_path=model_path,
            extra=["--mode", "autoregressive", "--explain"],
        )
    )
    assert exit_code == 2
    out = capsys.readouterr().out
    assert "total latency" in out
