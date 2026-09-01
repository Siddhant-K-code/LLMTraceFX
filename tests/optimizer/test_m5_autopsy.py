"""Contract tests for the process-isolated M5 MLX OOM autopsy."""

from __future__ import annotations

import csv
import io
import json
import time
import types
from pathlib import Path

import pytest

from llmtracefx.optimizer.lab import autopsy
from llmtracefx.optimizer.lab.autopsy import (
    AutopsyJournal,
    ChildProcessResult,
    autopsy_manifest_hash,
    build_autopsy_csv,
    build_autopsy_report,
    execute_autopsy_child,
    host_process_max_rss_bytes,
    host_process_rss_bytes,
    host_swap_bytes,
    load_autopsy_manifest,
    load_bound_frontier,
    probe_mlx_counters,
    probe_mlx_reset_peak_memory_api,
    render_autopsy_chart_svg,
    render_autopsy_report_html,
    run_autopsy,
    verify_autopsy_evidence,
)
from llmtracefx.optimizer.lab.core import HostSnapshot, LabError, SafetyDecision
from llmtracefx.optimizer.lab.frontier import frontier_manifest_hash

AUTOPSY_MANIFEST_PATH = Path("llmtracefx/optimizer/lab/data/autopsy-manifest-v1.json")


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
                "mlx-vlm": "0.6.8",
                "transformers": "5.16.1",
            },
        ),
    )


def _load():
    manifest, _ = load_autopsy_manifest(AUTOPSY_MANIFEST_PATH)
    frontier_manifest, base = load_bound_frontier(manifest, frontier_manifest_path=None)
    return manifest, frontier_manifest, base


def _prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(autopsy, "verify_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        autopsy, "assess_safety", lambda *args, **kwargs: _safe_decision()
    )


def _checkpoints_for(*, terminal: str | None = "completed") -> list[dict]:
    """Build a schema-valid checkpoint sequence for the given terminal outcome.

    Uses the real ``build_checkpoint`` so every generated fixture already
    satisfies the stricter shape/sequence/terminal-semantics validators
    instead of relying on a hand-maintained parallel structure.
    """
    started = time.monotonic()
    checkpoints: list[dict] = []

    def add(stage: str, *, extra: dict | None = None) -> None:
        checkpoints.append(
            autopsy.build_checkpoint(
                stage,
                len(checkpoints),
                mlx_module=None,
                started_monotonic=started,
                extra=extra,
            )
        )

    add("child_start")
    if terminal is None:
        return checkpoints
    if terminal == "completed":
        for stage in autopsy.MAIN_STAGE_SEQUENCE[1:]:
            add(stage)
        add("cleanup")
    elif terminal == "oom":
        for stage in (
            "before_model_load",
            "after_model_load",
            "after_prompt_tokenization",
        ):
            add(stage)
        add("caught_oom")
        add("cleanup")
    elif terminal in ("timeout", "failed"):
        add("before_model_load")
        add("after_model_load")
        add("cleanup")
    elif terminal == "signal":
        add("before_model_load")
        add("signal_received", extra={"signal_received": "SIGTERM"})
        add("cleanup")
    else:
        raise ValueError(f"unsupported terminal fixture {terminal!r}")
    return checkpoints


def _valid_journal_payload(
    manifest,
    frontier_manifest,
    *,
    run_mode: str = "exploratory",
    clean_boot_confirmed: bool = False,
    complete: bool | None = None,
    terminal: str | None = "completed",
    checkpoints: list[dict] | None = None,
) -> dict:
    if complete is None:
        complete = (
            terminal is not None and terminal not in autopsy._INCOMPLETE_TERMINALS
        )
    if checkpoints is None:
        checkpoints = _checkpoints_for(terminal=terminal)
    body = {
        "schema_version": autopsy.AUTOPSY_JOURNAL_SCHEMA_VERSION,
        "autopsy_id": manifest.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(manifest),
        "frontier_manifest_hash": frontier_manifest_hash(frontier_manifest),
        "model": {
            "repository_id": frontier_manifest.model_repository_id,
            "revision": frontier_manifest.model_revision,
        },
        "tier": manifest.tier_name,
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "synthetic": False,
        "started_at": "2026-09-01T00:00:00.000000Z",
        "sampling": {
            "periodic_sampling_enabled": False,
            "note": "only discrete checkpoints",
        },
        "peak_memory_reset": {
            "applied": False,
            "api": None,
            "note": "peak memory is never reset after model load",
        },
        "checkpoints": checkpoints,
        "complete": complete,
        "terminal": terminal,
    }
    body["envelope_sha256"] = autopsy.config_hash(body)
    return body


def _valid_result_payload(
    manifest,
    frontier_manifest,
    *,
    status: str = "completed",
    run_mode: str = "exploratory",
    clean_boot_confirmed: bool = False,
    journal_sha256: str | None = None,
    journal_complete: bool | None = None,
    journal_terminal: str | None = None,
) -> dict:
    if journal_complete is None:
        journal_complete = status in autopsy._COMPLETE_TERMINALS
    if journal_terminal is None:
        journal_terminal = status
    return {
        "schema_version": autopsy.AUTOPSY_RESULT_SCHEMA_VERSION,
        "autopsy_id": manifest.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(manifest),
        "frontier_manifest_hash": frontier_manifest_hash(frontier_manifest),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": manifest.tier_name,
        "requested_tokens": manifest.tier_requested_tokens,
        "actual_tokens": manifest.tier_requested_tokens - 1,
        "status": status,
        "reason": (
            None if status == "completed" else "MLX/Metal reported insufficient memory"
        ),
        "journal_sha256": journal_sha256 or ("a" * 64),
        "journal_complete": journal_complete,
        "journal_terminal": journal_terminal,
        "synthetic": False,
    }


def _launcher(manifest, frontier_manifest, *, status="completed", calls=None):
    calls = calls if calls is not None else []

    def launch(**kwargs):
        calls.append(kwargs["run_mode"])
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = _valid_journal_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            terminal=status,
        )
        (output_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        journal_digest = autopsy._file_sha256(output_dir / "journal.json")
        result = _valid_result_payload(
            manifest,
            frontier_manifest,
            status=status,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            journal_sha256=journal_digest,
            journal_complete=journal["complete"],
            journal_terminal=journal["terminal"],
        )
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return ChildProcessResult(
            exit_code=0 if status == "completed" else 2,
            timed_out=False,
            descendants_cleaned=True,
        )

    return launch


class _FakeGenerationResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.from_draft = False
        self.prompt_tokens = 0
        self.generation_tokens = 1
        self.finish_reason = None


class _FakeRuntime:
    """Injectable runtime double: never touches real weights or MLX."""

    def __init__(
        self, *, mode: str = "completed", tokens_before_failure: int = 1
    ) -> None:
        self.mode = mode
        self.tokens_before_failure = tokens_before_failure
        self.seed_calls: list[int] = []
        self.synchronize_calls: int = 0
        self.reset_peak_memory_calls: int = 0

    def load_model(self, path: Path):
        return object(), object()

    def encode(self, processor, text: str) -> list[int]:
        return list(range(len(text) // 4))

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def reset_peak_memory(self) -> None:
        self.reset_peak_memory_calls += 1

    def stream_generate(
        self,
        model,
        processor,
        prompt_tokens,
        *,
        max_tokens: int,
        draft_model,
        num_draft_tokens: int,
    ):
        produced = 0
        while produced < max_tokens:
            if self.mode == "oom" and produced == self.tokens_before_failure:
                raise MemoryError("Metal reported insufficient memory")
            if self.mode == "timeout" and produced == self.tokens_before_failure:
                raise TimeoutError("child exceeded its allotted generation window")
            if self.mode == "failed" and produced == self.tokens_before_failure:
                raise RuntimeError("unclassified runtime failure")
            produced += 1
            yield _FakeGenerationResponse(f"token-{produced}")
            if self.mode == "short":
                return


# ---------------------------------------------------------------------------
# Manifest binding.
# ---------------------------------------------------------------------------


def test_manifest_binds_by_hash_and_identity_to_the_packaged_frontier() -> None:
    manifest, frontier_manifest, base = _load()
    assert manifest.autopsy_id == "m5-pro-qwen3.8-27b-oom-autopsy-v1"
    assert manifest.frontier_id == frontier_manifest.frontier_id
    assert manifest.model_repository_id == frontier_manifest.model_repository_id
    assert manifest.model_revision == "3e6447f082e89cc7f0bc6e5441afd38dfce760ff"
    assert manifest.model_expected_download_bytes == 16081490933
    assert manifest.tier_name == "t256"
    assert manifest.tier_requested_tokens == 256
    assert base.model.revision == manifest.model_revision


def test_frontier_binding_hash_drift_is_rejected(tmp_path: Path) -> None:
    manifest, _, _ = _load()
    tampered = tmp_path / "frontier.json"
    payload = json.loads(
        Path("llmtracefx/optimizer/lab/data/fit-frontier-manifest-v1.json").read_text(
            encoding="utf-8"
        )
    )
    payload["frontier_id"] = "tampered"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(autopsy.AutopsyManifestError, match="does not match"):
        load_bound_frontier(manifest, frontier_manifest_path=tampered)


def test_autopsy_manifest_hash_is_deterministic_and_prefixed() -> None:
    manifest, _, _ = _load()
    first = autopsy_manifest_hash(manifest)
    second = autopsy_manifest_hash(manifest)
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_autopsy_manifest_hash_changes_when_own_field_drifts_but_frontier_does_not(
    tmp_path: Path,
) -> None:
    import dataclasses

    manifest, frontier_manifest, _ = _load()
    baseline_hash = autopsy_manifest_hash(manifest)
    tampered = dataclasses.replace(
        manifest, child_timeout_seconds=manifest.child_timeout_seconds + 1
    )
    tampered_hash = autopsy_manifest_hash(tampered)
    assert tampered_hash != baseline_hash
    # The frontier binding fields are untouched by this manifest-local drift.
    assert tampered.frontier_manifest_sha256 == manifest.frontier_manifest_sha256
    assert frontier_manifest_hash(frontier_manifest) == frontier_manifest_hash(
        frontier_manifest
    )


def test_stale_autopsy_manifest_hash_journal_is_rejected_not_silently_absent(
    tmp_path: Path,
) -> None:
    manifest, frontier_manifest, _ = _load()
    journal_path = tmp_path / "journal.json"
    payload = _valid_journal_payload(manifest, frontier_manifest)
    payload["autopsy_manifest_hash"] = "sha256:" + "0" * 64
    payload["envelope_sha256"] = autopsy.config_hash(
        {k: v for k, v in payload.items() if k != "envelope_sha256"}
    )
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    result = autopsy._load_journal_if_valid(
        journal_path,
        manifest,
        frontier_manifest,
        run_mode="exploratory",
        clean_boot_confirmed=False,
    )
    assert result.status == "invalid"
    assert "autopsy_manifest_hash" in (result.reason or "")


def test_stale_autopsy_manifest_hash_result_is_rejected(tmp_path: Path) -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["autopsy_manifest_hash"] = "sha256:" + "0" * 64
    with pytest.raises(LabError, match="autopsy_manifest_hash"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_claimed_journal_outcome_without_digest() -> (
    None
):
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["journal_sha256"] = None
    with pytest.raises(LabError, match="claimed journal outcome requires a digest"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_completed_without_complete_terminal_journal() -> (
    None
):
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["journal_complete"] = False
    with pytest.raises(LabError, match="completed status requires"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_bool_as_actual_tokens() -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["actual_tokens"] = True
    with pytest.raises(LabError, match="non-negative integer or null"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_shortfall_beyond_frontier_maximum() -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    max_shortfall = frontier_manifest.maximum_token_shortfall
    result["actual_tokens"] = manifest.tier_requested_tokens - max_shortfall - 1
    with pytest.raises(LabError, match="shortfall exceeds"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_tokens_above_requested_tier() -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["actual_tokens"] = manifest.tier_requested_tokens + 1
    with pytest.raises(LabError, match="exceeds the requested tier"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_completed_status_with_a_reason() -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest)
    result["reason"] = "should not be present for a completed result"
    with pytest.raises(LabError, match="completed result must have no reason"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_validate_autopsy_result_rejects_noncompleted_status_missing_a_reason() -> None:
    manifest, frontier_manifest, _ = _load()
    result = _valid_result_payload(manifest, frontier_manifest, status="failed")
    result["reason"] = None
    with pytest.raises(LabError, match="must carry a reason"):
        autopsy._validate_autopsy_result(
            result,
            manifest,
            frontier_manifest,
            run_mode="exploratory",
            clean_boot_confirmed=False,
        )


def test_run_autopsy_state_and_report_bind_autopsy_manifest_hash(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest),
    )
    assert state["autopsy_manifest_hash"] == autopsy_manifest_hash(manifest)
    assert state["result"]["autopsy_manifest_hash"] == autopsy_manifest_hash(manifest)
    report = build_autopsy_report(
        manifest, frontier_manifest, workspace=workspace, run_mode="exploratory"
    )
    assert report["bindings"]["autopsy_manifest_hash"] == autopsy_manifest_hash(
        manifest
    )
    assert report["bindings"]["frontier_manifest_hash"] == frontier_manifest_hash(
        frontier_manifest
    )


def test_journal_max_bytes_over_metadata_artifact_limit_is_rejected() -> None:
    raw = json.loads(AUTOPSY_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["journal"]["max_bytes"] = autopsy.MAX_METADATA_ARTIFACT_BYTES + 1
    with pytest.raises(autopsy.AutopsyManifestError, match="max_bytes"):
        autopsy.AutopsyManifest.from_dict(raw)


def test_journal_max_bytes_at_metadata_artifact_limit_is_accepted() -> None:
    raw = json.loads(AUTOPSY_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw["journal"]["max_bytes"] = autopsy.MAX_METADATA_ARTIFACT_BYTES
    manifest = autopsy.AutopsyManifest.from_dict(raw)
    assert manifest.journal_max_bytes == autopsy.MAX_METADATA_ARTIFACT_BYTES


def test_packaged_manifest_loads_from_external_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest, source = load_autopsy_manifest(None)
    frontier_manifest, base = load_bound_frontier(manifest, frontier_manifest_path=None)
    assert source.startswith("package:llmtracefx.optimizer.lab/")
    assert base.model.revision == frontier_manifest.model_revision


def test_plan_does_not_construct_runtime_or_load_weights(monkeypatch, capsys) -> None:
    _prepare(monkeypatch)
    monkeypatch.setattr(autopsy, "model_files_present", lambda *args: True)
    monkeypatch.setattr(
        autopsy,
        "MLXVLMRuntime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plan loaded weights")
        ),
    )
    assert autopsy.main(["plan", "--manifest", str(AUTOPSY_MANIFEST_PATH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["weights_loaded"] is False
    assert payload["downloads_performed"] is False
    assert payload["manifest"] == str(AUTOPSY_MANIFEST_PATH)
    assert payload["sampling"]["periodic_sampling_enabled"] is False


# ---------------------------------------------------------------------------
# Direct MLX counter probing.
# ---------------------------------------------------------------------------


def test_probe_mlx_counters_records_exact_qualified_names_and_values() -> None:
    def get_active_memory() -> int:
        return 111

    def get_cache_memory() -> int:
        return 222

    def get_peak_memory() -> int:
        return 333

    fake_mlx = types.SimpleNamespace(
        get_active_memory=get_active_memory,
        get_cache_memory=get_cache_memory,
        get_peak_memory=get_peak_memory,
    )
    result = probe_mlx_counters(fake_mlx)
    assert result["active_bytes"] == {
        "api": autopsy._qualified_name(get_active_memory, "get_active_memory"),
        "value": 111,
        "unit": "bytes",
        "error_category": None,
    }
    assert result["cache_bytes"]["value"] == 222
    assert result["peak_bytes"]["value"] == 333
    assert all(entry["error_category"] is None for entry in result.values())


def test_probe_mlx_counters_missing_api_is_null_never_zero() -> None:
    fake_mlx = types.SimpleNamespace(get_active_memory=lambda: 5)
    result = probe_mlx_counters(fake_mlx)
    assert result["active_bytes"]["value"] == 5
    assert result["cache_bytes"] == {
        "api": None,
        "value": None,
        "unit": "bytes",
        "error_category": None,
    }
    assert result["peak_bytes"] == {
        "api": None,
        "value": None,
        "unit": "bytes",
        "error_category": None,
    }


def test_probe_mlx_counters_absent_module_is_entirely_null() -> None:
    result = probe_mlx_counters(None)
    for entry in result.values():
        assert entry["api"] is None
        assert entry["value"] is None
        assert entry["error_category"] is None


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError])
def test_probe_mlx_counters_swallows_callable_errors_as_null(error_type) -> None:
    def broken() -> int:
        raise error_type("device unavailable, serial ABC123 detached")

    fake_mlx = types.SimpleNamespace(
        get_active_memory=broken, get_cache_memory=broken, get_peak_memory=broken
    )
    result = probe_mlx_counters(fake_mlx)
    assert all(entry["value"] is None for entry in result.values())
    assert all(entry["api"] is not None for entry in result.values())
    assert all(
        entry["error_category"] == error_type.__name__ for entry in result.values()
    )
    # The raw exception message (which could carry private detail) is never
    # recorded -- only the exception's type name is.
    serialized = json.dumps(result)
    assert "device unavailable" not in serialized
    assert "ABC123" not in serialized


def test_probe_mlx_reset_peak_memory_api_reports_qualified_name_or_null() -> None:
    def reset_peak_memory() -> None:
        return None

    assert probe_mlx_reset_peak_memory_api(
        types.SimpleNamespace(reset_peak_memory=reset_peak_memory)
    ) == autopsy._qualified_name(reset_peak_memory, "reset_peak_memory")
    assert probe_mlx_reset_peak_memory_api(types.SimpleNamespace()) is None
    assert probe_mlx_reset_peak_memory_api(None) is None


def test_probe_mlx_counters_against_real_installed_mlx() -> None:
    mlx_core = pytest.importorskip("mlx.core")
    result = probe_mlx_counters(mlx_core)
    assert result["active_bytes"]["api"] == "mlx.core.get_active_memory"
    assert result["cache_bytes"]["api"] == "mlx.core.get_cache_memory"
    assert result["peak_bytes"]["api"] == "mlx.core.get_peak_memory"
    for entry in result.values():
        assert isinstance(entry["value"], int)


# ---------------------------------------------------------------------------
# Host process/system probing (macOS and Linux code paths).
# ---------------------------------------------------------------------------


def test_host_process_rss_bytes_macos_parses_ps_kibibytes() -> None:
    value, provenance = host_process_rss_bytes(
        system="Darwin", run_text=lambda argv: "2048"
    )
    assert value == 2048 * 1024
    assert "ps -o rss=" in provenance


def test_host_process_rss_bytes_macos_null_when_ps_unavailable() -> None:
    value, provenance = host_process_rss_bytes(
        system="Darwin", run_text=lambda argv: None
    )
    assert value is None
    assert provenance is None


def test_host_process_rss_bytes_linux_parses_vmrss_kb(tmp_path: Path) -> None:
    status_path = tmp_path / "status"
    status_path.write_text("Name:\tpython\nVmRSS:\t   4096 kB\n", encoding="utf-8")
    value, provenance = host_process_rss_bytes(
        system="Linux", linux_status_path=status_path
    )
    assert value == 4096 * 1024
    assert "VmRSS" in provenance


def test_host_process_rss_bytes_linux_null_when_proc_missing(tmp_path: Path) -> None:
    value, provenance = host_process_rss_bytes(
        system="Linux", linux_status_path=tmp_path / "missing"
    )
    assert value is None
    assert provenance is None


def test_host_process_rss_bytes_unknown_platform_is_null() -> None:
    assert host_process_rss_bytes(system="Windows") == (None, None)


def test_host_process_max_rss_bytes_macos_is_already_bytes() -> None:
    usage = types.SimpleNamespace(ru_maxrss=31_000_000)
    value, provenance = host_process_max_rss_bytes(
        system="Darwin", getrusage=lambda: usage
    )
    assert value == 31_000_000
    assert "bytes" in provenance


def test_host_process_max_rss_bytes_linux_converts_kib_to_bytes() -> None:
    usage = types.SimpleNamespace(ru_maxrss=31_000)
    value, provenance = host_process_max_rss_bytes(
        system="Linux", getrusage=lambda: usage
    )
    assert value == 31_000 * 1024
    assert "KiB" in provenance


def test_host_process_max_rss_bytes_null_on_getrusage_failure() -> None:
    def broken():
        raise OSError("unsupported")

    assert host_process_max_rss_bytes(system="Darwin", getrusage=broken) == (None, None)


def test_host_swap_bytes_macos_parses_sysctl_swapusage() -> None:
    text = "total = 12288.00M  used = 10744.75M  free = 1543.25M  (encrypted)"
    total, used, provenance = host_swap_bytes(
        system="Darwin", run_text=lambda argv: text
    )
    assert total == int(12288.00 * 1024 * 1024)
    assert used == int(10744.75 * 1024 * 1024)
    assert "sysctl" in provenance


def test_host_swap_bytes_macos_null_when_sysctl_unavailable() -> None:
    assert host_swap_bytes(system="Darwin", run_text=lambda argv: None) == (
        None,
        None,
        None,
    )


def test_host_swap_bytes_linux_parses_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:  1000 kB\nSwapTotal:  2048 kB\nSwapFree:   512 kB\n",
        encoding="utf-8",
    )
    total, used, provenance = host_swap_bytes(
        system="Linux", linux_meminfo_path=meminfo
    )
    assert total == 2048 * 1024
    assert used == (2048 - 512) * 1024
    assert "meminfo" in provenance


def test_host_swap_bytes_unknown_platform_is_null() -> None:
    assert host_swap_bytes(system="Solaris") == (None, None, None)


# ---------------------------------------------------------------------------
# Bounded, atomic journal persistence.
# ---------------------------------------------------------------------------


def _new_journal(tmp_path: Path, *, max_checkpoints: int = 16, max_bytes: int = 65536):
    manifest, frontier_manifest, _base = _load()
    return AutopsyJournal(
        path=tmp_path / "journal.json",
        autopsy_id=manifest.autopsy_id,
        autopsy_manifest_hash_value=autopsy_manifest_hash(manifest),
        frontier_manifest_hash_value=frontier_manifest_hash(frontier_manifest),
        model_repository_id=frontier_manifest.model_repository_id,
        model_revision=frontier_manifest.model_revision,
        tier=manifest.tier_name,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        max_checkpoints=max_checkpoints,
        max_bytes=max_bytes,
    )


def test_journal_checkpoint_persists_atomically_with_matching_envelope_hash(
    tmp_path: Path,
) -> None:
    journal = _new_journal(tmp_path)
    journal.checkpoint("child_start", None)
    journal.checkpoint("before_model_load", None)
    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    assert [entry["stage"] for entry in on_disk["checkpoints"]] == [
        "child_start",
        "before_model_load",
    ]
    canonical = {k: v for k, v in on_disk.items() if k != "envelope_sha256"}
    assert autopsy.config_hash(canonical) == on_disk["envelope_sha256"]
    assert on_disk["sampling"]["periodic_sampling_enabled"] is False
    assert on_disk["complete"] is False


def test_journal_finalize_marks_complete_and_terminal(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path)
    journal.checkpoint("child_start", None)
    journal.finalize(complete=True, terminal="completed")
    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is True
    assert on_disk["terminal"] == "completed"


def test_journal_checkpoint_count_bound_is_enforced(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path, max_checkpoints=2)
    journal.checkpoint("child_start", None)
    journal.checkpoint("before_model_load", None)
    with pytest.raises(LabError, match="maximum checkpoint count"):
        journal.checkpoint("after_model_load", None)


def test_journal_byte_size_bound_is_enforced(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path, max_bytes=16)
    with pytest.raises(LabError, match="maximum artifact size"):
        journal.checkpoint("child_start", None)


def test_journal_never_records_pids_paths_or_hostnames(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path)
    journal.checkpoint("child_start", None)
    journal.finalize(complete=True, terminal="completed")
    serialized = journal.path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    import os

    assert str(os.getpid()) not in serialized


def test_journal_records_peak_memory_is_never_reset_with_explicit_provenance(
    tmp_path: Path,
) -> None:
    journal = _new_journal(tmp_path)
    journal.checkpoint("child_start", None)
    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    # Peak memory must never be silently reset after model load; the
    # journal explicitly records that fact rather than omitting the field.
    assert on_disk["peak_memory_reset"] == {
        "applied": False,
        "api": None,
        "note": (
            "Peak memory is never reset after model load, so a fresh "
            "subprocess's peak reading includes load growth."
        ),
    }


def test_build_checkpoint_includes_clock_unit_and_provenance() -> None:
    checkpoint = autopsy.build_checkpoint(
        "child_start", 0, mlx_module=None, started_monotonic=time.monotonic()
    )
    assert checkpoint["wall_clock_provenance"]
    assert isinstance(checkpoint["wall_clock_provenance"], str)
    assert checkpoint["monotonic_offset_unit"] == "seconds"
    assert checkpoint["monotonic_offset_provenance"]
    assert isinstance(checkpoint["monotonic_offset_provenance"], str)
    assert checkpoint["monotonic_offset_seconds"] >= 0.0


def test_build_checkpoint_keeps_other_scopes_when_one_probe_hits_memory_error() -> None:
    def failed_rss():
        raise MemoryError("private details must not escape")

    checkpoint = autopsy.build_checkpoint(
        "child_start",
        0,
        mlx_module=None,
        started_monotonic=time.monotonic(),
        rss_probe=failed_rss,
        max_rss_probe=lambda: (2048, "getrusage"),
        swap_probe=lambda: (4096, 1024, "sysctl"),
    )
    assert checkpoint["host_process"]["rss_bytes"] == {
        "value": None,
        "provenance": "probe failed (MemoryError)",
    }
    assert checkpoint["host_process"]["max_rss_bytes"]["value"] == 2048
    assert checkpoint["host_system"]["swap_used_bytes"]["value"] == 1024
    assert "private details" not in json.dumps(checkpoint)


# ---------------------------------------------------------------------------
# Signal handling.
# ---------------------------------------------------------------------------


def test_record_signal_checkpoint_marks_incomplete_with_signal_terminal(
    tmp_path: Path,
) -> None:
    journal = _new_journal(tmp_path)
    journal.checkpoint("child_start", None)
    autopsy._record_signal_checkpoint(journal, None, autopsy.signal.SIGTERM)
    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is False
    assert on_disk["terminal"] == "signal"
    stages = [entry["stage"] for entry in on_disk["checkpoints"]]
    # signal_received is its own distinct stage, never mislabeled as cleanup.
    assert stages == ["child_start", "signal_received", "cleanup"]
    signal_entry = on_disk["checkpoints"][stages.index("signal_received")]
    assert signal_entry["signal_received"] == "SIGTERM"
    cleanup_entry = on_disk["checkpoints"][stages.index("cleanup")]
    assert cleanup_entry.get("signal_received") is None


def test_record_signal_checkpoint_is_best_effort_when_checkpoint_budget_exhausted(
    tmp_path: Path,
) -> None:
    """A failure writing one best-effort checkpoint must not block the rest."""
    journal = _new_journal(tmp_path, max_checkpoints=2)
    journal.checkpoint("child_start", None)
    journal.checkpoint("before_model_load", None)
    # The checkpoint budget is now exhausted: signal_received/cleanup writes
    # will each raise LabError internally, but finalize must still land so
    # the journal's terminal outcome is never silently dropped.
    autopsy._record_signal_checkpoint(journal, None, autopsy.signal.SIGTERM)
    on_disk = json.loads(journal.path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is False
    assert on_disk["terminal"] == "signal"
    stages = [entry["stage"] for entry in on_disk["checkpoints"]]
    assert stages == ["child_start", "before_model_load"]


# ---------------------------------------------------------------------------
# Isolated child execution.
# ---------------------------------------------------------------------------


def test_execute_autopsy_child_success_walks_full_checkpoint_sequence(
    tmp_path: Path,
) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    result = execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=lambda: _FakeRuntime(mode="completed"),
        mlx_module_factory=lambda: None,
    )
    assert result["status"] == "completed"
    assert result["reason"] is None
    assert result["journal_complete"] is True
    assert result["journal_terminal"] == "completed"
    journal = json.loads((output_dir / "journal.json").read_text(encoding="utf-8"))
    stages = [entry["stage"] for entry in journal["checkpoints"]]
    assert stages == [
        "child_start",
        "before_model_load",
        "after_model_load",
        "after_prompt_tokenization",
        "immediately_before_prefill_generation",
        "after_first_token",
        "completion",
        "cleanup",
    ]


def test_execute_autopsy_child_oom_writes_caught_oom_checkpoint(tmp_path: Path) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    result = execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=lambda: _FakeRuntime(mode="oom", tokens_before_failure=0),
        mlx_module_factory=lambda: None,
    )
    assert result["status"] == "oom"
    assert result["reason"] == "MLX/Metal reported insufficient memory"
    journal = json.loads((output_dir / "journal.json").read_text(encoding="utf-8"))
    stages = [entry["stage"] for entry in journal["checkpoints"]]
    assert "caught_oom" in stages
    assert stages[-1] == "cleanup"
    assert journal["complete"] is True
    assert journal["terminal"] == "oom"


def test_execute_autopsy_child_timeout_is_classified_and_cleaned_up(
    tmp_path: Path,
) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    result = execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=lambda: _FakeRuntime(mode="timeout", tokens_before_failure=0),
        mlx_module_factory=lambda: None,
    )
    assert result["status"] == "timeout"
    journal = json.loads((output_dir / "journal.json").read_text(encoding="utf-8"))
    assert journal["terminal"] == "timeout"
    assert journal["checkpoints"][-1]["stage"] == "cleanup"


def test_execute_autopsy_child_generic_failure_still_reaches_cleanup(
    tmp_path: Path,
) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    result = execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=lambda: _FakeRuntime(mode="failed", tokens_before_failure=0),
        mlx_module_factory=lambda: None,
    )
    assert result["status"] == "failed"
    assert result["journal_complete"] is True
    journal = json.loads((output_dir / "journal.json").read_text(encoding="utf-8"))
    assert journal["checkpoints"][-1]["stage"] == "cleanup"
    assert "caught_oom" not in [c["stage"] for c in journal["checkpoints"]]


def test_execute_autopsy_child_result_binds_journal_hash(tmp_path: Path) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    result = execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=lambda: _FakeRuntime(mode="completed"),
        mlx_module_factory=lambda: None,
    )
    assert result["journal_sha256"] == autopsy._file_sha256(output_dir / "journal.json")


def test_execute_autopsy_child_seeds_and_synchronizes_at_correct_points_never_resets_peak(
    tmp_path: Path,
) -> None:
    manifest, frontier_manifest, base = _load()
    output_dir = tmp_path / "attempt"
    runtimes: list[_FakeRuntime] = []

    def runtime_factory() -> _FakeRuntime:
        runtime = _FakeRuntime(mode="completed")
        runtimes.append(runtime)
        return runtime

    execute_autopsy_child(
        manifest,
        frontier_manifest,
        base,
        model_path=tmp_path / "model",
        output_dir=output_dir,
        run_mode="exploratory",
        clean_boot_confirmed=False,
        runtime_factory=runtime_factory,
        mlx_module_factory=lambda: None,
    )
    (runtime,) = runtimes
    # Seeded exactly once with the manifest's own generation seed, and
    # synchronized after load, after seeding/before prefill, and after
    # generation -- no periodic polling, and peak memory is never reset.
    assert runtime.seed_calls == [base.generation.seed]
    assert runtime.synchronize_calls == 3
    assert runtime.reset_peak_memory_calls == 0


# ---------------------------------------------------------------------------
# Parent orchestration: resume, staleness, timeout/signal survival.
# ---------------------------------------------------------------------------


def test_run_autopsy_resumes_a_prior_completed_attempt_without_relaunching(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    first_calls: list[str] = []
    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest, calls=first_calls),
    )
    second_calls: list[str] = []
    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest, calls=second_calls),
    )
    assert first_calls == ["exploratory"]
    assert second_calls == []
    assert state["resumed"] is True
    assert state["result"]["status"] == "completed"


@pytest.mark.parametrize("status", ["oom", "timeout", "failed"])
def test_run_autopsy_resumes_any_terminal_status_never_retries_expensive_runs(
    tmp_path, monkeypatch, status
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest, status=status),
    )
    second_calls: list[str] = []
    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(
            manifest, frontier_manifest, calls=second_calls, status=status
        ),
    )
    assert second_calls == []
    assert state["result"]["status"] == status


def test_run_autopsy_resume_rejects_stale_artifact(tmp_path, monkeypatch) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest),
    )
    result_path = workspace / "exploratory" / "result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["frontier_manifest_hash"] = "sha256:" + "0" * 64
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="stale autopsy artifact"):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=workspace,
            model_path=tmp_path / "model",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )


def test_missing_journal_is_reported_as_missing_not_invalid(tmp_path) -> None:
    manifest, frontier_manifest, base = _load()
    result = autopsy._load_journal_if_valid(
        tmp_path / "does-not-exist.json",
        manifest,
        frontier_manifest,
        run_mode="exploratory",
        clean_boot_confirmed=False,
    )
    assert result.status == "missing"
    assert result.journal is None
    assert result.digest is None
    assert result.reason is None


def test_stale_or_foreign_journal_fails_closed_and_is_ignored(tmp_path) -> None:
    manifest, frontier_manifest, base = _load()
    journal_path = tmp_path / "journal.json"
    payload = _valid_journal_payload(manifest, frontier_manifest)
    payload["autopsy_id"] = "some-other-autopsy"
    payload["envelope_sha256"] = autopsy.config_hash(
        {k: v for k, v in payload.items() if k != "envelope_sha256"}
    )
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    result = autopsy._load_journal_if_valid(
        journal_path,
        manifest,
        frontier_manifest,
        run_mode="exploratory",
        clean_boot_confirmed=False,
    )
    assert result.status == "invalid"
    assert result.journal is None
    assert result.digest is None
    assert "autopsy_id" in (result.reason or "")


def test_tampered_journal_envelope_hash_fails_closed(tmp_path) -> None:
    manifest, frontier_manifest, base = _load()
    journal_path = tmp_path / "journal.json"
    payload = _valid_journal_payload(manifest, frontier_manifest)
    payload["terminal"] = "completed-but-tampered"
    journal_path.write_text(json.dumps(payload), encoding="utf-8")
    result = autopsy._load_journal_if_valid(
        journal_path,
        manifest,
        frontier_manifest,
        run_mode="exploratory",
        clean_boot_confirmed=False,
    )
    assert result.status == "invalid"
    assert result.journal is None
    assert result.digest is None
    assert "envelope hash" in (result.reason or "")


def test_symlinked_journal_is_rejected_not_followed(tmp_path) -> None:
    manifest, frontier_manifest, base = _load()
    real_target = tmp_path / "outside.json"
    real_target.write_text(
        json.dumps(_valid_journal_payload(manifest, frontier_manifest)),
        encoding="utf-8",
    )
    journal_path = tmp_path / "journal.json"
    journal_path.symlink_to(real_target)
    result = autopsy._load_journal_if_valid(
        journal_path,
        manifest,
        frontier_manifest,
        run_mode="exploratory",
        clean_boot_confirmed=False,
    )
    assert result.status == "invalid"
    assert result.journal is None
    assert result.digest is None
    assert "symlink" in (result.reason or "")


def test_invalid_unbound_journal_blocks_verify_and_resume(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"

    def launch(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "journal.json").write_text("{broken", encoding="utf-8")
        return ChildProcessResult(
            exit_code=1, timed_out=False, descendants_cleaned=True
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["journal_sha256"] is None
    verification = verify_autopsy_evidence(
        manifest, frontier_manifest, workspace=workspace, run_mode="exploratory"
    )
    assert verification["verified"] is False
    assert any("full contract validation" in item for item in verification["failures"])
    with pytest.raises(LabError, match="checkpoint journal is invalid"):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=workspace,
            model_path=tmp_path / "model",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )


def test_finalize_rejects_candidate_result_with_a_digest_not_matching_the_real_journal(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"

    def launch(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = _valid_journal_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
        )
        (output_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        result = _valid_result_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            # A forged digest: does not match the real on-disk journal.
            journal_sha256="a" * 64,
        )
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return ChildProcessResult(
            exit_code=0, timed_out=False, descendants_cleaned=True
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["status"] == "failed"
    assert "digest does not match" in state["result"]["reason"]


def test_finalize_rejects_candidate_completion_claim_mismatching_the_real_journal(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"

    def launch(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = _valid_journal_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            terminal="oom",
        )
        (output_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        digest = autopsy._file_sha256(output_dir / "journal.json")
        result = _valid_result_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            status="oom",
            journal_sha256=digest,
            # Lies about the journal's real terminal outcome.
            journal_terminal="completed",
        )
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return ChildProcessResult(
            exit_code=1, timed_out=False, descendants_cleaned=True
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["status"] == "failed"
    assert "completion/terminal claims do not match" in state["result"]["reason"]


def test_incomplete_journal_after_signal_is_usable_failure_evidence_not_success(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"

    def launch(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = _valid_journal_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            complete=False,
            terminal="signal",
        )
        (output_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        # No result.json: the child was killed before its finally block ran.
        return ChildProcessResult(
            exit_code=None, timed_out=False, descendants_cleaned=True
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["status"] == "failed"
    assert state["result"]["journal_complete"] is False
    assert state["result"]["journal_terminal"] == "signal"
    assert state["result"]["journal_sha256"] is not None
    assert state["result"]["synthetic"] is False


def test_parent_enforced_timeout_produces_failure_shaped_result(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)

    def launch(**kwargs):
        return ChildProcessResult(
            exit_code=None, timed_out=True, descendants_cleaned=True
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["status"] == "timeout"
    assert state["result"]["reason"] == "autopsy exceeded parent-enforced timeout"


def test_cleanup_failure_produces_failure_shaped_result_never_success(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)

    def launch(**kwargs):
        return ChildProcessResult(
            exit_code=1, timed_out=False, descendants_cleaned=False
        )

    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=tmp_path / "workspace",
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    assert state["result"]["status"] == "failed"
    assert "cleanup could not be verified" in state["result"]["reason"]


# ---------------------------------------------------------------------------
# Publication / exploratory clean-boot gating.
# ---------------------------------------------------------------------------


def test_publication_requires_explicit_clean_boot_confirmation(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    with pytest.raises(LabError, match="operator assertion"):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=tmp_path,
            model_path=tmp_path / "model",
            run_mode="publication",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )


def test_exploratory_cannot_accept_clean_boot_confirmation(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    with pytest.raises(LabError, match="only valid in publication mode"):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=tmp_path,
            model_path=tmp_path / "model",
            run_mode="exploratory",
            clean_boot_confirmed=True,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )


def test_publication_with_confirmation_proceeds(tmp_path, monkeypatch) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    calls: list[str] = []
    state = run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=tmp_path,
        model_path=tmp_path / "model",
        run_mode="publication",
        clean_boot_confirmed=True,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest, calls=calls),
    )
    assert calls == ["publication"]
    assert state["clean_boot_confirmed"] is True
    assert state["result"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Reporting: deterministic, sanitized, never zero-filled for missing values.
# ---------------------------------------------------------------------------


def _run_with_checkpoints(tmp_path, monkeypatch, checkpoints):
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "private-user" / "workspace"

    def launch(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        journal = _valid_journal_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            checkpoints=checkpoints,
        )
        (output_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        digest = autopsy._file_sha256(output_dir / "journal.json")
        result = _valid_result_payload(
            manifest,
            frontier_manifest,
            run_mode=kwargs["run_mode"],
            clean_boot_confirmed=kwargs["clean_boot_confirmed"],
            journal_sha256=digest,
        )
        (output_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return ChildProcessResult(
            exit_code=0, timed_out=False, descendants_cleaned=True
        )

    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=launch,
    )
    return manifest, frontier_manifest, workspace


def test_report_is_deterministic_and_keeps_missing_values_null_not_zero(
    tmp_path, monkeypatch
) -> None:
    # A schema-valid, complete "completed" sequence, with the
    # before_model_load checkpoint swapped for a fixture whose probes
    # deliberately leave some measurements null (never zero-filled).
    fake_mlx = types.SimpleNamespace(
        get_active_memory=lambda: 100,
        get_peak_memory=lambda: 200,
        # No get_cache_memory: cache_bytes must stay null, never 0.
    )
    checkpoints = _checkpoints_for(terminal="completed")
    before_model_load_index = next(
        i for i, cp in enumerate(checkpoints) if cp["stage"] == "before_model_load"
    )
    checkpoints[before_model_load_index] = autopsy.build_checkpoint(
        "before_model_load",
        before_model_load_index,
        mlx_module=fake_mlx,
        started_monotonic=time.monotonic(),
        rss_probe=lambda: (4096, "ps"),
        max_rss_probe=lambda: (None, None),
        swap_probe=lambda: (None, 0, "sysctl"),
    )
    manifest, frontier_manifest, workspace = _run_with_checkpoints(
        tmp_path, monkeypatch, checkpoints
    )
    report = build_autopsy_report(
        manifest, frontier_manifest, workspace=workspace, run_mode="exploratory"
    )
    first = render_autopsy_report_html(report)
    second = render_autopsy_report_html(report)
    assert first == second
    row = report["checkpoints"][before_model_load_index]
    assert row["mlx_active_bytes"] == 100
    assert row["mlx_cache_bytes"] is None
    assert row["host_max_rss_bytes"] is None
    assert row["swap_total_bytes"] is None
    assert row["swap_used_bytes"] == 0
    assert str(tmp_path) not in json.dumps(report)
    assert report["synthetic"] is False
    assert report["observed_deltas"]
    delta = report["observed_deltas"][before_model_load_index - 1]
    assert delta["to_stage"] == "before_model_load"
    assert delta["mlx_cache_bytes_delta"] is None
    assert "n/a" in first
    assert (
        "0.0" not in first.split("n/a")[0][-20:]
    )  # sanity: no silent zero-fill marker
    csv_text = build_autopsy_csv(report)
    assert "n/a" in csv_text
    csv_rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert csv_rows[0]["mlx_cache_bytes"] == "n/a"
    assert csv_rows[0]["wall_clock_provenance"] == (
        "schema.utc_now_iso (ISO-8601, UTC)"
    )


def test_report_svg_chart_has_three_labeled_series() -> None:
    report = {
        "checkpoints": [
            {
                "stage": "before_model_load",
                "mlx_peak_bytes": 100,
                "host_rss_bytes": 200,
                "swap_used_bytes": 0,
            },
            {
                "stage": "after_model_load",
                "mlx_peak_bytes": 300,
                "host_rss_bytes": 400,
                "swap_used_bytes": None,
            },
        ]
    }
    svg = render_autopsy_chart_svg(report)
    assert svg.count("<polyline") == 3
    assert "MLX allocator" in svg
    assert "Host process" in svg
    assert "Host system" in svg


def test_report_refuses_when_evidence_is_invalid(tmp_path) -> None:
    manifest, frontier_manifest, base = _load()
    with pytest.raises(LabError, match="refused invalid evidence"):
        build_autopsy_report(
            manifest, frontier_manifest, workspace=tmp_path, run_mode="exploratory"
        )


def test_verify_autopsy_evidence_detects_journal_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest),
    )
    journal_path = workspace / "exploratory" / "journal.json"
    journal_path.write_text(
        journal_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    result = verify_autopsy_evidence(
        manifest, frontier_manifest, workspace=workspace, run_mode="exploratory"
    )
    assert result["verified"] is False
    assert any("journal digest" in failure for failure in result["failures"])


def test_verify_autopsy_evidence_passes_for_a_clean_completed_run(
    tmp_path, monkeypatch
) -> None:
    manifest, frontier_manifest, base = _load()
    _prepare(monkeypatch)
    workspace = tmp_path / "workspace"
    run_autopsy(
        manifest,
        frontier_manifest,
        base,
        autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
        frontier_manifest_path=None,
        workspace=workspace,
        model_path=tmp_path / "model",
        run_mode="exploratory",
        clean_boot_confirmed=False,
        resume=True,
        launcher=_launcher(manifest, frontier_manifest),
    )
    result = verify_autopsy_evidence(
        manifest, frontier_manifest, workspace=workspace, run_mode="exploratory"
    )
    assert result == {"verified": True, "failures": []}


# ---------------------------------------------------------------------------
# Run refuses without a cached model / passing safety gate.
# ---------------------------------------------------------------------------


def test_run_refuses_when_model_is_not_present(tmp_path, monkeypatch) -> None:
    manifest, frontier_manifest, base = _load()
    monkeypatch.setattr(
        autopsy, "assess_safety", lambda *args, **kwargs: _safe_decision()
    )
    with pytest.raises(LabError):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=tmp_path,
            model_path=tmp_path / "model",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )


def test_run_refuses_when_safety_gate_fails(tmp_path, monkeypatch) -> None:
    manifest, frontier_manifest, base = _load()
    monkeypatch.setattr(autopsy, "verify_model", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        autopsy,
        "assess_safety",
        lambda *args, **kwargs: SafetyDecision(
            safe=False,
            blockers=("insufficient memory",),
            snapshot=_safe_decision().snapshot,
        ),
    )
    with pytest.raises(LabError, match="safety preflight"):
        run_autopsy(
            manifest,
            frontier_manifest,
            base,
            autopsy_manifest_path=AUTOPSY_MANIFEST_PATH,
            frontier_manifest_path=None,
            workspace=tmp_path,
            model_path=tmp_path / "model",
            run_mode="exploratory",
            clean_boot_confirmed=False,
            resume=True,
            launcher=_launcher(manifest, frontier_manifest),
        )
