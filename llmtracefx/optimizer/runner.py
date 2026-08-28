"""Reproducible experiment runner primitive.

Executes a supplied command (argv list — never a shell string) for a
configured number of warmup and measured repetitions, capturing stdout,
stderr, and timing metadata as atomically-written JSON artifacts. Supports
timeouts and resuming a partially-completed run by skipping repetitions
that already completed successfully.

This module does not parse command output into the canonical
``ExperimentRecord`` schema — that is the job of format-specific
collectors (see ``llmtracefx.optimizer.parsers``). The runner's job is
only to execute safely and record what happened, without ever reporting
success for a run that failed, timed out, or could not be started.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import utc_now_iso


class RunnerConfigError(ValueError):
    """Raised for invalid or unsupported runner configuration."""


class RepetitionOutcome(str, Enum):
    """What happened when a single repetition was executed."""

    COMPLETED = "completed"
    """Process started and exited; ``returncode`` indicates success/failure."""

    TIMED_OUT = "timed_out"
    """Process did not finish within ``timeout_seconds`` and was killed."""

    FAILED_TO_START = "failed_to_start"
    """The command could not be launched at all (e.g. binary not found)."""


def _config_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerConfigError(f"config.{key} must be an integer")
    return int(value)


def _config_timeout(data: dict[str, Any]) -> float | None:
    value = data.get("timeout_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerConfigError("config.timeout_seconds must be a number or null")
    timeout = float(value)
    if not math.isfinite(timeout):
        raise RunnerConfigError("config.timeout_seconds must be finite")
    return timeout


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for one experiment run, loaded from JSON (or YAML)."""

    run_id: str
    command: tuple[str, ...]
    results_dir: Path
    warmup_repetitions: int = 0
    measured_repetitions: int = 1
    timeout_seconds: float | None = None
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise RunnerConfigError("run_id must be non-empty")
        if not self.command:
            raise RunnerConfigError("command must contain at least one argument")
        if self.warmup_repetitions < 0:
            raise RunnerConfigError("warmup_repetitions must be >= 0")
        if self.measured_repetitions < 1:
            raise RunnerConfigError("measured_repetitions must be >= 1")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise RunnerConfigError("timeout_seconds must be > 0 when set")

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, base_dir: Path | None = None
    ) -> RunnerConfig:
        try:
            command = data["command"]
        except KeyError as exc:
            raise RunnerConfigError(
                "config is missing required field: command"
            ) from exc
        if not isinstance(command, list) or not all(
            isinstance(part, str) and part for part in command
        ):
            raise RunnerConfigError(
                "config.command must be a list of non-empty strings"
            )

        try:
            run_id = data["run_id"]
            results_dir_raw = data["results_dir"]
        except KeyError as exc:
            raise RunnerConfigError(f"config is missing required field: {exc}") from exc
        if not isinstance(run_id, str) or not run_id:
            raise RunnerConfigError("config.run_id must be a non-empty string")
        if not isinstance(results_dir_raw, str) or not results_dir_raw:
            raise RunnerConfigError("config.results_dir must be a non-empty string")

        cwd = data.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise RunnerConfigError("config.cwd must be a string or null")

        env_raw = data.get("env", {})
        if not isinstance(env_raw, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in env_raw.items()
        ):
            raise RunnerConfigError(
                "config.env must be an object with non-empty string keys "
                "and string values"
            )

        results_dir = Path(results_dir_raw)
        if not results_dir.is_absolute() and base_dir is not None:
            results_dir = base_dir / results_dir

        return cls(
            run_id=run_id,
            command=tuple(command),
            results_dir=results_dir,
            warmup_repetitions=_config_int(data, "warmup_repetitions", 0),
            measured_repetitions=_config_int(data, "measured_repetitions", 1),
            timeout_seconds=_config_timeout(data),
            cwd=cwd,
            env=dict(env_raw),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RunnerConfig:
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RunnerConfigError(
                    "YAML config requires PyYAML to be installed (`uv add pyyaml`); "
                    "use a .json config instead if it is unavailable"
                ) from exc
            data = yaml.safe_load(text)
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RunnerConfigError(
                    f"invalid JSON in {config_path}: {exc}"
                ) from exc
        else:
            raise RunnerConfigError(
                f"unsupported config extension '{suffix}' (use .json or .yaml)"
            )

        if not isinstance(data, dict):
            raise RunnerConfigError(
                f"config in {config_path} must be a JSON/YAML object"
            )
        return cls.from_dict(data, base_dir=config_path.parent)


@dataclass(frozen=True)
class RepetitionResult:
    """Outcome and artifact locations for one repetition."""

    kind: str
    """'warmup' or 'measured'."""
    index: int
    outcome: RepetitionOutcome
    returncode: int | None
    elapsed_seconds: float
    started_at: str
    ended_at: str
    stdout_path: str
    stderr_path: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepetitionResult:
        return cls(
            kind=data["kind"],
            index=data["index"],
            outcome=RepetitionOutcome(data["outcome"]),
            returncode=data.get("returncode"),
            elapsed_seconds=data["elapsed_seconds"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            stdout_path=data["stdout_path"],
            stderr_path=data["stderr_path"],
            error_message=data.get("error_message"),
        )

    @property
    def succeeded(self) -> bool:
        return self.outcome == RepetitionOutcome.COMPLETED and self.returncode == 0


def _to_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


class ExperimentRunner:
    """Executes an ``RunnerConfig`` and records artifacts on disk."""

    def __init__(self, config: RunnerConfig) -> None:
        self.config = config

    def _repetition_dir(self, kind: str, index: int) -> Path:
        return self.config.results_dir / f"{kind}-{index:03d}"

    def _load_existing_result(self, kind: str, index: int) -> RepetitionResult | None:
        meta_path = self._repetition_dir(kind, index) / "meta.json"
        if not meta_path.exists():
            return None
        try:
            return RepetitionResult.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # A corrupt/partial artifact from an interrupted previous run
            # must not be mistaken for a completed repetition.
            return None

    def _run_once(self, kind: str, index: int) -> RepetitionResult:
        rep_dir = self._repetition_dir(kind, index)
        rep_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = rep_dir / "stdout.txt"
        stderr_path = rep_dir / "stderr.txt"

        merged_env = {**os.environ, **self.config.env}
        started_at = utc_now_iso()
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                list(self.config.command),
                cwd=self.config.cwd,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start
            ended_at = utc_now_iso()
            _atomic_write_text(stdout_path, _to_text(exc.stdout))
            _atomic_write_text(stderr_path, _to_text(exc.stderr))
            result = RepetitionResult(
                kind=kind,
                index=index,
                outcome=RepetitionOutcome.TIMED_OUT,
                returncode=None,
                elapsed_seconds=elapsed,
                started_at=started_at,
                ended_at=ended_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error_message=f"timed out after {self.config.timeout_seconds}s",
            )
        except (OSError, ValueError) as exc:
            elapsed = time.perf_counter() - start
            ended_at = utc_now_iso()
            _atomic_write_text(stdout_path, "")
            _atomic_write_text(stderr_path, "")
            result = RepetitionResult(
                kind=kind,
                index=index,
                outcome=RepetitionOutcome.FAILED_TO_START,
                returncode=None,
                elapsed_seconds=elapsed,
                started_at=started_at,
                ended_at=ended_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error_message=str(exc),
            )
        else:
            elapsed = time.perf_counter() - start
            ended_at = utc_now_iso()
            _atomic_write_text(stdout_path, completed.stdout or "")
            _atomic_write_text(stderr_path, completed.stderr or "")
            result = RepetitionResult(
                kind=kind,
                index=index,
                outcome=RepetitionOutcome.COMPLETED,
                returncode=completed.returncode,
                elapsed_seconds=elapsed,
                started_at=started_at,
                ended_at=ended_at,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                error_message=(
                    None if completed.returncode == 0 else "non-zero exit code"
                ),
            )

        _atomic_write_json(rep_dir / "meta.json", result.to_dict())
        return result

    def _run_repetition(
        self, kind: str, index: int, *, resume: bool
    ) -> RepetitionResult:
        if resume:
            existing = self._load_existing_result(kind, index)
            if existing is not None and existing.succeeded:
                return existing
        return self._run_once(kind, index)

    def run(self, *, resume: bool = True) -> list[RepetitionResult]:
        """Execute all warmup and measured repetitions.

        Returns the measured (non-warmup) results in order. When
        ``resume`` is true, repetitions whose ``meta.json`` already
        records a successful completion are skipped and re-reported
        as-is; failed/timed-out/missing repetitions are always re-run.
        """
        for index in range(self.config.warmup_repetitions):
            self._run_repetition("warmup", index, resume=resume)

        measured_results = [
            self._run_repetition("measured", index, resume=resume)
            for index in range(self.config.measured_repetitions)
        ]

        summary_path = self.config.results_dir / "summary.jsonl"
        _atomic_write_text(
            summary_path,
            "\n".join(
                json.dumps(result.to_dict(), sort_keys=False)
                for result in measured_results
            )
            + "\n",
        )
        return measured_results
