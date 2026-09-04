"""Load-bearing tests for the offline evidence catalog trust boundary."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtracefx.evidence import cli, core
from llmtracefx.evidence.registry import ADAPTERS, CLAIM_DIMENSIONS, SOURCES

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "examples" / "evidence-catalog" / "catalog.json"


def _catalog() -> dict:
    return json.loads(json.dumps(core.build_catalog(ROOT), allow_nan=False))


def _reseal(catalog: dict) -> dict:
    body = dict(catalog)
    body.pop("catalog_hash", None)
    catalog["catalog_hash"] = core._json_hash(body)
    return catalog


def _write_catalog(tmp_path: Path, catalog: dict) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(core.canonical_json(catalog), encoding="utf-8")
    return path


def test_committed_catalog_verifies_every_registered_adapter() -> None:
    result = core.verify_catalog(CATALOG, ROOT)
    assert result["verified"] is True
    assert result["entries"] == len(SOURCES) == 10
    assert result["edges"] == 7
    assert result["verified_evidence_ids"] == sorted(
        source["evidence_id"] for source in SOURCES
    )


@pytest.mark.parametrize("source", SOURCES, ids=lambda source: source["adapter"])
def test_every_source_adapter_verifies(source: dict) -> None:
    core.verify_source(ROOT, source)


def test_completed_crossover_adapter_is_closed_but_not_fabricated() -> None:
    assert "vllm_crossover_results_v1" in ADAPTERS
    assert all(source["adapter"] != "vllm_crossover_results_v1" for source in SOURCES)
    script, arguments = core.SCRIPT_ADAPTERS["vllm_crossover_results_v1"]
    assert script == "llmtracefx/evidence/vllm_crossover_results_verifier.py"
    assert arguments == ("verify", "--bundle-dir", "{bundle}")


def test_generation_is_deterministic_and_matches_committed_files() -> None:
    first = core.build_catalog(ROOT)
    second = core.build_catalog(ROOT)
    assert core.canonical_json(first) == core.canonical_json(second)
    assert core.render_catalog_artifacts(first) == core.render_catalog_artifacts(second)
    assert CATALOG.read_text(encoding="utf-8") == core.canonical_json(first)


def test_catalog_hash_tamper_fails() -> None:
    catalog = _catalog()
    catalog["entries"][0]["limitations"][0] = "tampered"
    with pytest.raises(core.CatalogError, match="catalog hash"):
        core.validate_catalog_document(catalog)


def test_resealed_claim_drift_still_fails_closed(tmp_path: Path) -> None:
    catalog = _catalog()
    catalog["entries"][0]["claims"]["timing"] = {
        "state": "supported",
        "provenance": "invented",
    }
    path = _write_catalog(tmp_path, _reseal(catalog))
    with pytest.raises(core.CatalogError, match="closed registry"):
        core.verify_catalog(path, ROOT)


def test_unknown_kind_and_status_are_rejected() -> None:
    for field, value, message in (
        ("kind", "unknown", "kind is unknown"),
        ("status", "stale", "status is unknown"),
        ("outcome", "maybe", "outcome is unknown"),
    ):
        catalog = _catalog()
        catalog["entries"][0][field] = value
        with pytest.raises(core.CatalogError, match=message):
            core.validate_catalog_document(_reseal(catalog))


def test_unknown_verifier_is_rejected() -> None:
    catalog = _catalog()
    catalog["entries"][0]["verifier"] = {"name": "untrusted", "version": "1"}
    with pytest.raises(core.CatalogError, match="closed registry"):
        core.validate_catalog_document(_reseal(catalog))


@pytest.mark.parametrize(
    "path",
    (
        "../escape",
        "/absolute",
        "nested\\escape",
        "a//b",
        "a/./b",
        "a/.",
        "foo/..",
        "C:/x",
    ),
)
def test_path_escape_is_rejected(path: str) -> None:
    catalog = _catalog()
    catalog["entries"][0]["public_path"] = path
    with pytest.raises(core.CatalogError, match="relative path|contained"):
        core.validate_catalog_document(_reseal(catalog))


def test_non_string_path_is_rejected() -> None:
    catalog = _catalog()
    catalog["entries"][0]["public_path"] = 42
    with pytest.raises(core.CatalogError, match="safe relative path"):
        core.validate_catalog_document(_reseal(catalog))


def test_symlinked_bundle_is_rejected(tmp_path: Path) -> None:
    target = ROOT / "examples" / "optimizer" / "qwen3-8b-m5-control"
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "bundle").symlink_to(target, target_is_directory=True)
    source = {
        "evidence_id": "symlink-test",
        "public_path": "examples/bundle",
        "adapter": "sha256_allowlist_v1",
        "artifact_files": tuple(path.name for path in target.iterdir()),
    }
    with pytest.raises(core.CatalogError, match="symlink|escapes"):
        core.verify_source(tmp_path, source)


def test_device_artifact_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "examples" / "bundle"
    bundle.mkdir(parents=True)
    os.mkfifo(bundle / "artifact.json")
    source = {
        "evidence_id": "device-test",
        "public_path": "examples/bundle",
        "artifact_files": ("artifact.json",),
    }
    with pytest.raises(core.CatalogError, match="non-regular"):
        core._artifact_set_hash(tmp_path, source)


def test_tampered_bundle_checksum_is_rejected(tmp_path: Path) -> None:
    source = next(
        source for source in SOURCES if source["adapter"] == "sha256_allowlist_v1"
    )
    destination = tmp_path / source["public_path"]
    destination.mkdir(parents=True)
    original = ROOT / source["public_path"]
    for name in source["artifact_files"]:
        (destination / name).write_bytes((original / name).read_bytes())
    with (destination / "evidence-summary.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(core.CatalogError, match="checksum mismatch"):
        core.verify_source(tmp_path, source)


def test_oversized_and_deep_catalog_json_are_rejected(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (core.MAX_CATALOG_BYTES + 1))
    with pytest.raises(core.CatalogError, match="size limit"):
        core.verify_catalog(oversized, ROOT)

    value: object = None
    for _ in range(core.MAX_JSON_DEPTH + 2):
        value = [value]
    with pytest.raises(core.CatalogError, match="maximum JSON depth"):
        core._walk_json(value)


@pytest.mark.parametrize("text", ('{"value": NaN}', '{"broken":', "[]"))
def test_nonfinite_malformed_and_wrong_type_json_are_rejected(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(core.CatalogError):
        core.verify_catalog(path, ROOT)


def test_deep_json_file_is_rejected_as_catalog_error(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text("[" * 2000 + "null" + "]" * 2000, encoding="utf-8")
    with pytest.raises(core.CatalogError, match="maximum JSON depth"):
        core._load_json(path, core.MAX_CATALOG_BYTES)


def test_catalog_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "catalog.json"
    link.symlink_to(CATALOG)
    with pytest.raises(core.CatalogError, match="must not be a symlink"):
        core.verify_catalog(link, ROOT)


def test_duplicate_dangling_and_cyclic_edges_are_rejected() -> None:
    duplicate = _catalog()
    duplicate["entries"][1]["evidence_id"] = duplicate["entries"][0]["evidence_id"]
    with pytest.raises(core.CatalogError, match="duplicate evidence IDs"):
        core.validate_catalog_document(_reseal(duplicate))

    dangling = _catalog()
    dangling["entries"][0]["dependencies"].append(
        {"evidence_id": "missing-evidence", "relation": "derived_from"}
    )
    dangling["edges"].append(
        {
            "source": dangling["entries"][0]["evidence_id"],
            "target": "missing-evidence",
            "relation": "derived_from",
        }
    )
    dangling["edges"].sort(
        key=lambda edge: (edge["source"], edge["target"], edge["relation"])
    )
    with pytest.raises(core.CatalogError, match="dangling"):
        core.validate_catalog_document(_reseal(dangling))

    cyclic = _catalog()
    first = cyclic["entries"][0]["evidence_id"]
    second = cyclic["entries"][1]["evidence_id"]
    cyclic["entries"][0]["dependencies"] = [
        {"evidence_id": second, "relation": "derived_from"}
    ]
    cyclic["entries"][1]["dependencies"] = [
        {"evidence_id": first, "relation": "derived_from"}
    ]
    cyclic["edges"] = sorted(
        (
            {
                "source": entry["evidence_id"],
                "target": dependency["evidence_id"],
                "relation": dependency["relation"],
            }
            for entry in cyclic["entries"]
            for dependency in entry["dependencies"]
        ),
        key=lambda edge: (edge["source"], edge["target"], edge["relation"]),
    )
    with pytest.raises(core.CatalogError, match="cycle"):
        core.validate_catalog_document(_reseal(cyclic))


@pytest.mark.parametrize("value", (math.inf, -math.inf, math.nan))
def test_nonfinite_budget_metrics_are_rejected(value: float) -> None:
    catalog = _catalog()
    catalog["entries"][0]["budget"]["authorized_usd"] = value
    with pytest.raises(core.CatalogError, match="finite"):
        core.validate_catalog_document(catalog)


def test_claim_matrix_is_closed_and_never_boolean() -> None:
    artifacts = core.render_catalog_artifacts(_catalog())
    matrix = json.loads(artifacts["claim-matrix.json"])
    assert matrix["dimensions"] == list(CLAIM_DIMENSIONS)
    assert len(matrix["rows"]) == 10
    for row in matrix["rows"]:
        assert set(row["claims"]) == set(CLAIM_DIMENSIONS)
        assert {claim["state"] for claim in row["claims"].values()} <= {
            "supported",
            "unsupported",
            "not_applicable",
        }
        assert all(
            not isinstance(claim["state"], bool) for claim in row["claims"].values()
        )


def test_generated_views_are_static_private_path_free_and_self_contained() -> None:
    artifacts = core.render_catalog_artifacts(_catalog())
    for name, text in artifacts.items():
        assert "/Users/" not in text
        assert "/home/" not in text
        assert "<script" not in text.lower()
        assert "@import" not in text.lower()
        if name in {"index.html", "graph.svg"}:
            assert "http://" not in text
            assert "https://" not in text
            assert "href=" not in text.lower()


def test_privacy_mutation_is_rejected_even_when_resealed() -> None:
    catalog = _catalog()
    catalog["entries"][0]["limitations"][0] = "private /Users/alice/model path"
    with pytest.raises(core.CatalogError, match="private home path"):
        core.validate_catalog_document(_reseal(catalog))


@pytest.mark.parametrize("key", ("hostname", "pid", "raw_prompt", "account_id"))
def test_private_evidence_json_fields_are_rejected(key: str) -> None:
    with pytest.raises(core.CatalogError, match="private evidence field"):
        core._scan_json_privacy({"nested": {key: "private"}})


def test_script_adapter_uses_no_shell_and_scrubs_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = next(
        source for source in SOURCES if source["adapter"] == "metal_public_v1"
    )
    observed: dict = {}

    def fake_run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-propagate")
    monkeypatch.setattr(subprocess, "run", fake_run)
    core._run_script_verifier(ROOT, source)
    assert observed["shell"] is False
    environment = observed["env"]
    assert "OPENROUTER_API_KEY" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert observed["command"][1:3] == (
        "-I",
        str(Path(core.__file__).with_name("_offline_runner.py").resolve()),
    )
    assert observed["command"][3:] == (
        str(ROOT),
        str(ROOT / "examples" / "metal_evidence" / "evidence_demo.py"),
        "verify",
        "--public-dir",
        str(ROOT / "examples" / "metal_evidence" / "public"),
    )


def test_source_revision_binding_drift_is_rejected() -> None:
    source = copy.deepcopy(
        next(
            source
            for source in SOURCES
            if source["evidence_id"] == "qwen3-8b-m5-pro-control-20260902"
        )
    )
    source["model"]["revision"] = "0" * 40
    with pytest.raises(core.CatalogError, match="model revision binding"):
        core._verify_source_bindings(ROOT, source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("captured_at", "1999-01-01", "capture timestamp binding"),
        (
            "model.revision",
            "z-ai/glm-5.3-wrong and z-ai/glm-5.3-flash-wrong",
            "model revision binding",
        ),
    ),
)
def test_hosted_source_metadata_is_bound_to_bundle(
    field: str, value: str, message: str
) -> None:
    source = copy.deepcopy(
        next(
            source
            for source in SOURCES
            if source["evidence_id"] == "openrouter-glm-2k-comparison-20260902"
        )
    )
    if field == "model.revision":
        source["model"]["revision"] = value
    else:
        source[field] = value
    with pytest.raises(core.CatalogError, match=message):
        core._verify_source_bindings(ROOT, source)


@pytest.mark.parametrize(
    "source",
    (
        "import socket\nsocket.socket()\n",
        "import subprocess\nsubprocess.run(['true'])\n",
        "import mlx\n",
        "open('side-effect', 'w').write('forbidden')\n",
        "open('/etc/hosts').read()\n",
    ),
)
def test_offline_runner_blocks_network_processes_and_model_imports(
    tmp_path: Path, source: str
) -> None:
    script = tmp_path / "attempt.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(core.__file__).with_name("_offline_runner.py").resolve()),
            str(tmp_path),
            str(script),
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    assert completed.returncode != 0
    assert (
        "disabled during verification" in completed.stderr
        or "disabled during evidence verification" in completed.stderr
        or "outside verified roots is disabled" in completed.stderr
    )


def test_isolated_runner_ignores_checkout_shadow_module(tmp_path: Path) -> None:
    shadow = tmp_path / "llmtracefx" / "evidence"
    shadow.mkdir(parents=True)
    marker = tmp_path / "shadowed"
    (shadow / "_offline_runner.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    script = tmp_path / "safe.py"
    script.write_text("pass\n", encoding="utf-8")
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(core.__file__).with_name("_offline_runner.py").resolve()),
            str(tmp_path),
            str(script),
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_published_schema_defines_entry_claim_and_edge_shapes() -> None:
    schema = core._schema_document()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["entry"]["additionalProperties"] is False
    assert schema["$defs"]["entry"]["required"]
    assert schema["$defs"]["entry"]["properties"]["claims"]["required"] == list(
        CLAIM_DIMENSIONS
    )
    assert schema["$defs"]["edge"]["properties"]["relation"]["enum"] == sorted(
        core.RELATIONS
    )
    assert schema["properties"]["entries"]["items"] == {"$ref": "#/$defs/entry"}


@pytest.mark.parametrize(
    "token",
    (
        "sk-abcdefghijklmnopqrstuvwxyz",
        "hf_abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz",
        "github_pat_abcdefghijklmnopqrstuvwxyz",
    ),
)
def test_standard_secret_token_formats_are_rejected(token: str) -> None:
    with pytest.raises(core.CatalogError, match="secret-shaped token"):
        core._scan_privacy("catalog.json", f'{{"value":"{token}"}}')


def test_missing_generated_checksums_fail_closed(tmp_path: Path) -> None:
    path = _write_catalog(tmp_path, _catalog())
    with pytest.raises(core.CatalogError, match="missing: SHA256SUMS"):
        core.verify_catalog(path, ROOT)


def test_external_cwd_verification_uses_explicit_catalog(tmp_path: Path) -> None:
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "llmtracefx.evidence.cli",
            "verify",
            "--catalog",
            str(CATALOG),
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["verified"] is True
    assert result["entries"] == 10


def test_unrelated_project_is_not_inferred_as_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='other'\n")
    (tmp_path / "examples").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "installed" / "cli.py"))
    with pytest.raises(core.CatalogError, match="pass --repo-root or --catalog"):
        cli._default_repo_root()


def test_unregistered_candidates_are_empty_on_current_main() -> None:
    assert core.build_catalog(ROOT)["unregistered_candidates"] == []


def test_unregistered_evidence_marker_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(core, "SOURCES", ())
    examples = tmp_path / "examples"
    candidate = examples / "new-public-bundle"
    candidate.mkdir(parents=True)
    (candidate / "evidence-manifest.json").write_text("{}\n", encoding="utf-8")
    assert core._discover_candidates(tmp_path) == [
        "examples/new-public-bundle",
    ]
