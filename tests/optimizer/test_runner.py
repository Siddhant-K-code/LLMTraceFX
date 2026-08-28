"""Tests for the reproducible experiment runner primitive."""

import json
import sys
import textwrap

import pytest

from llmtracefx.optimizer.runner import (
    ExperimentRunner,
    RepetitionOutcome,
    RunnerConfig,
    RunnerConfigError,
)


def _echo_command(
    stdout_text: str = "hello", stderr_text: str = "", exit_code: int = 0
):
    script = textwrap.dedent(
        f"""
        import sys
        sys.stdout.write({stdout_text!r})
        sys.stderr.write({stderr_text!r})
        sys.exit({exit_code})
        """
    )
    return [sys.executable, "-c", script]


def _counting_command(counter_path):
    """A command that increments a counter file and fails until count >= 2."""
    script = textwrap.dedent(
        """
        import sys
        path = sys.argv[1]
        try:
            with open(path) as fh:
                count = int(fh.read().strip() or "0")
        except FileNotFoundError:
            count = 0
        count += 1
        with open(path, "w") as fh:
            fh.write(str(count))
        sys.exit(0 if count >= 2 else 1)
        """
    )
    return [sys.executable, "-c", script, str(counter_path)]


def _always_incrementing_command(counter_path):
    script = textwrap.dedent(
        """
        import sys
        path = sys.argv[1]
        try:
            with open(path) as fh:
                count = int(fh.read().strip() or "0")
        except FileNotFoundError:
            count = 0
        count += 1
        with open(path, "w") as fh:
            fh.write(str(count))
        sys.exit(0)
        """
    )
    return [sys.executable, "-c", script, str(counter_path)]


def test_runner_executes_warmup_and_measured_repetitions(tmp_path):
    config = RunnerConfig(
        run_id="demo",
        command=tuple(_echo_command("out", "")),
        results_dir=tmp_path / "results",
        warmup_repetitions=1,
        measured_repetitions=2,
    )
    runner = ExperimentRunner(config)
    results = runner.run()

    assert len(results) == 2
    assert all(result.outcome == RepetitionOutcome.COMPLETED for result in results)
    assert all(result.returncode == 0 for result in results)
    assert (config.results_dir / "warmup-000" / "meta.json").exists()
    assert (config.results_dir / "measured-000" / "meta.json").exists()
    assert (config.results_dir / "measured-001" / "meta.json").exists()

    summary_lines = (
        (config.results_dir / "summary.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(summary_lines) == 2
    for line in summary_lines:
        json.loads(line)  # must be valid JSON per line


def test_runner_captures_stdout_and_stderr(tmp_path):
    config = RunnerConfig(
        run_id="demo",
        command=tuple(_echo_command("stdout-content", "stderr-content")),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
    )
    results = ExperimentRunner(config).run()

    stdout_text = (config.results_dir / "measured-000" / "stdout.txt").read_text(
        encoding="utf-8"
    )
    stderr_text = (config.results_dir / "measured-000" / "stderr.txt").read_text(
        encoding="utf-8"
    )
    assert stdout_text == "stdout-content"
    assert stderr_text == "stderr-content"
    assert results[0].stdout_path == str(
        config.results_dir / "measured-000" / "stdout.txt"
    )


def test_runner_reports_failure_without_success_shaped_fallback(tmp_path):
    config = RunnerConfig(
        run_id="demo",
        command=tuple(_echo_command("", "", exit_code=17)),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
    )
    results = ExperimentRunner(config).run()

    assert results[0].outcome == RepetitionOutcome.COMPLETED
    assert results[0].returncode == 17
    assert results[0].succeeded is False
    assert results[0].error_message == "non-zero exit code"


def test_runner_marks_timeout_explicitly(tmp_path):
    sleep_script = textwrap.dedent(
        """
        import time
        time.sleep(5)
        """
    )
    config = RunnerConfig(
        run_id="demo",
        command=(sys.executable, "-c", sleep_script),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
        timeout_seconds=0.2,
    )
    results = ExperimentRunner(config).run()

    assert results[0].outcome == RepetitionOutcome.TIMED_OUT
    assert results[0].returncode is None
    assert results[0].succeeded is False
    assert "timed out" in results[0].error_message


def test_runner_reports_failed_to_start_for_missing_binary(tmp_path):
    config = RunnerConfig(
        run_id="demo",
        command=("definitely-not-a-real-binary-xyz",),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
    )
    results = ExperimentRunner(config).run()

    assert results[0].outcome == RepetitionOutcome.FAILED_TO_START
    assert results[0].succeeded is False
    assert results[0].error_message


def test_runner_resume_skips_completed_and_reruns_failed(tmp_path):
    counter_path = tmp_path / "counter.txt"
    config = RunnerConfig(
        run_id="demo",
        command=tuple(_counting_command(counter_path)),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
    )
    runner = ExperimentRunner(config)

    first = runner.run(resume=True)
    assert first[0].succeeded is False  # first attempt: count becomes 1, exits 1
    assert counter_path.read_text() == "1"

    second = runner.run(resume=True)
    assert (
        second[0].succeeded is True
    )  # resume reruns the failed rep: count becomes 2, exits 0
    assert counter_path.read_text() == "2"

    third = runner.run(resume=True)
    assert third[0].succeeded is True  # resume skips the now-completed rep
    assert counter_path.read_text() == "2"  # command was not invoked again


def test_runner_no_resume_always_reruns(tmp_path):
    counter_path = tmp_path / "counter.txt"
    config = RunnerConfig(
        run_id="demo",
        command=tuple(_always_incrementing_command(counter_path)),
        results_dir=tmp_path / "results",
        measured_repetitions=1,
    )
    runner = ExperimentRunner(config)

    runner.run(resume=True)
    assert counter_path.read_text() == "1"

    runner.run(resume=False)
    assert counter_path.read_text() == "2"


def test_runner_config_rejects_empty_run_id(tmp_path):
    with pytest.raises(RunnerConfigError, match="run_id"):
        RunnerConfig(run_id="", command=("true",), results_dir=tmp_path)


def test_runner_config_rejects_empty_command(tmp_path):
    with pytest.raises(RunnerConfigError, match="command"):
        RunnerConfig(run_id="demo", command=(), results_dir=tmp_path)


def test_runner_config_rejects_non_positive_timeout(tmp_path):
    with pytest.raises(RunnerConfigError, match="timeout_seconds"):
        RunnerConfig(
            run_id="demo", command=("true",), results_dir=tmp_path, timeout_seconds=0
        )


def test_runner_config_from_dict_requires_command_list():
    with pytest.raises(RunnerConfigError, match="command must be a list"):
        RunnerConfig.from_dict(
            {"run_id": "demo", "results_dir": "out", "command": "not-a-list"}
        )


def test_runner_config_from_file_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "demo",
                "command": ["true"],
                "results_dir": "results",
                "warmup_repetitions": 1,
                "measured_repetitions": 3,
            }
        ),
        encoding="utf-8",
    )
    config = RunnerConfig.from_file(config_path)
    assert config.run_id == "demo"
    assert config.warmup_repetitions == 1
    assert config.measured_repetitions == 3
    # Relative results_dir resolves against the config file's directory.
    assert config.results_dir == tmp_path / "results"


def test_runner_config_from_file_rejects_unsupported_extension(tmp_path):
    config_path = tmp_path / "config.txt"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RunnerConfigError, match="unsupported config extension"):
        RunnerConfig.from_file(config_path)


def test_runner_config_from_file_rejects_invalid_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RunnerConfigError, match="invalid JSON"):
        RunnerConfig.from_file(config_path)
