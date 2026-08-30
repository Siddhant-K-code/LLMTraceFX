"""Regressions for the findings raised in successive exact-head reviews.

Each test names the finding it pins. They live together rather than scattered
across the topical suites so the list can be checked against the review in one
place; the topical suites still cover the surrounding behaviour.

Everything here is synthetic. Nothing loads a model, calls an API, or runs a
benchmark.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
from _compare_fixtures import (
    COMPARE_POLICY,
    collection_dir_for,
    edit_sidecar,
    reseal_run,
    write_api_run,
    write_json,
    write_run,
)

import llmtracefx.optimizer.compare.evidence as evidence_module
from llmtracefx.optimizer.collectors._shared import sha256_bytes
from llmtracefx.optimizer.collectors.openai_api import ARTIFACT_MANIFEST_NAME
from llmtracefx.optimizer.compare.compare import compare
from llmtracefx.optimizer.compare.cost import TokenUsage, estimate_run_cost
from llmtracefx.optimizer.compare.evidence import (
    ApiEvidence,
    ApiEvidenceError,
    CompareEvidenceError,
    load_comparison_evidence,
)
from llmtracefx.optimizer.compare.explain import format_compare_report_text
from llmtracefx.optimizer.compare.identity import (
    ComparableUnitKey,
    CompareIdentityError,
)
from llmtracefx.optimizer.compare.policy import (
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
    ComparePolicyError,
)
from llmtracefx.optimizer.compare.pricing import (
    PricingEntry,
    PricingError,
    PricingManifest,
)
from llmtracefx.optimizer.compare.report import (
    CompareReport,
    CompareReportValidationError,
    StratumOutcome,
)
from llmtracefx.optimizer.compare.report_html import (
    _redact_paths_in_prose,
    render_compare_report_html,
)
from llmtracefx.optimizer.workloads.api_verify import (
    RUN_MANIFEST_NAME,
    RUN_MANIFEST_SCHEMA_VERSION,
)

_PRICING = PricingManifest.from_dict(
    {
        "schema_version": "1",
        "currency": "USD",
        "entries": [
            {
                "entry_id": "flash",
                "provider": "z-ai",
                "model_id": "glm-5.3-flash",
                "currency": "USD",
                "effective_at": "2026-01-01",
                "source": "illustrative example",
                "rates_are_illustrative": True,
                "input_per_million": 1.0,
                "output_per_million": 2.0,
            },
            {
                "entry_id": "frontier",
                "provider": "z-ai",
                "model_id": "glm-5.3",
                "currency": "USD",
                "effective_at": "2026-01-01",
                "source": "illustrative example",
                "rates_are_illustrative": True,
                "input_per_million": 3.0,
                "output_per_million": 6.0,
            },
        ],
    }
)


def _policy(
    objective: CompareObjective = CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS,
    **constraints: object,
) -> ComparePolicy:
    return ComparePolicy(
        objective=objective,
        name="hardening",
        constraints=CompareConstraints(**constraints),  # type: ignore[arg-type]
    )


# --- Finding 1: no false direct-collector ingestion claim -----------------


def test_a_raw_collector_directory_is_rejected_with_an_explanation(
    tmp_path: Path,
) -> None:
    """collect-api output has no verification.json, so nothing evaluated it."""
    raw = tmp_path / "collector-out"
    raw.mkdir()
    (raw / "record.json").write_text("{}", encoding="utf-8")
    (raw / "api_evidence.json").write_text("{}", encoding="utf-8")
    (raw / "artifacts.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CompareEvidenceError) as excinfo:
        load_comparison_evidence((raw,))
    message = str(excinfo.value)
    assert "raw collector output" in message
    assert "workloads run" in message
    assert "verification.json" in message


def test_an_ordinary_empty_directory_is_not_mistaken_for_collector_output(
    tmp_path: Path,
) -> None:
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()


# --- Finding 2: run ids collide across systems by construction ------------


def test_the_same_run_id_on_two_systems_is_the_ordinary_case(
    tmp_path: Path,
) -> None:
    """A matrix run_id names the task, not the system, so it always collides.

    This is the headline use case: one matrix executed locally and against a
    hosted API. Treating the repeated id as a conflict would reject exactly
    the comparison this command exists to make.
    """
    local = tmp_path / "local"
    hosted = tmp_path / "hosted"
    shared_id = "structured-json-profile-extraction-2k-autoregressive"
    write_run(local, shared_id, total_ms=8000.0)
    write_api_run(hosted, shared_id, total_ms=1200.0)

    loaded = load_comparison_evidence((local, hosted))
    assert len(loaded.runs) == 2
    assert len({run.system_key for run in loaded.runs}) == 2
    assert len({run.unit_key for run in loaded.runs}) == 1

    report = compare(results_dirs=(local, hosted), policy=_policy())
    assert len(report.strata) == 1
    assert len(report.strata[0].ranked) == 2


def test_the_same_run_id_twice_within_one_system_is_two_repetitions(
    tmp_path: Path,
) -> None:
    """Repetitions of one matrix row must survive, not be rejected.

    Executing a matrix twice into two results directories is how evidence
    for a repetition count above one is produced, and both executions carry
    the same ``run_id`` and the same system identity because a matrix run id
    names the task and nothing about the system or the attempt.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_run(first, "shared", total_ms=1000.0)
    write_run(second, "shared", total_ms=2000.0)
    loaded = load_comparison_evidence((first, second))
    assert len(loaded.runs) == 2
    assert {run.record.timing.total.value for run in loaded.runs} == {1000.0, 2000.0}


def test_a_directory_named_twice_is_not_double_counted(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    loaded = load_comparison_evidence((tmp_path, tmp_path))
    assert len(loaded.runs) == 1


def test_two_spellings_of_one_directory_are_not_double_counted(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, "local-1")
    alias = tmp_path / "sub" / ".."
    (tmp_path / "sub").mkdir(exist_ok=True)
    loaded = load_comparison_evidence((tmp_path, alias))
    assert len(loaded.runs) == 1


# --- Finding 3: full request and execution identity in the keys -----------


def test_a_system_prompt_splits_the_comparable_unit(tmp_path: Path) -> None:
    """The same user prompt under a system prompt is a different question."""
    write_api_run(tmp_path, "plain", model_id="glm-5.3")
    write_api_run(
        tmp_path,
        "with-system",
        model_id="glm-5.3",
        system_prompt_hash="sha256:a-system-prompt",
    )
    units = {run.unit_key for run in load_comparison_evidence((tmp_path,)).runs}
    assert len(units) == 2
    shapes = {unit.request_shape for unit in units}
    assert None in shapes
    assert any(shape is not None for shape in shapes)


def test_a_bare_prompt_api_run_still_shares_a_unit_with_a_local_run(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, "local-1", max_tokens_argv=512)
    write_api_run(tmp_path, "api-1", max_output_tokens=512)
    units = {run.unit_key for run in load_comparison_evidence((tmp_path,)).runs}
    assert len(units) == 1
    assert next(iter(units)).request_shape is None


def test_a_different_endpoint_is_a_different_system(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", endpoint_origin="https://one.invalid")
    write_api_run(tmp_path, "b", endpoint_origin="https://two.invalid")
    systems = {run.system_key for run in load_comparison_evidence((tmp_path,)).runs}
    assert len(systems) == 2
    assert {system.endpoint for system in systems} == {
        "https://one.invalid/v1/chat/completions",
        "https://two.invalid/v1/chat/completions",
    }


def test_a_different_thinking_type_is_a_different_system(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", thinking_type="enabled")
    write_api_run(tmp_path, "b", thinking_type="disabled")
    systems = {run.system_key for run in load_comparison_evidence((tmp_path,)).runs}
    assert len(systems) == 2
    assert {system.thinking_type for system in systems} == {"enabled", "disabled"}


def test_a_different_execution_config_hash_is_a_different_system(
    tmp_path: Path,
) -> None:
    """Catches a configuration difference this module does not model by name."""
    write_api_run(tmp_path, "a", config_hash="cfg-one")
    write_api_run(tmp_path, "b", config_hash="cfg-two")
    systems = {run.system_key for run in load_comparison_evidence((tmp_path,)).runs}
    assert len(systems) == 2
    assert {system.execution_config_hash for system in systems} == {
        "cfg-one",
        "cfg-two",
    }


# --- Finding 4: validate the artifact set before trusting it --------------


def test_a_sidecar_with_no_completeness_marker_excludes_the_run(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1", write_artifact_marker=False)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "completeness marker" in loaded.excluded[0].reason


def test_a_tampered_artifact_set_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1")
    response = collection_dir_for(tmp_path, "api-1") / "response.txt"
    response.write_text("swapped out from under the marker\n", encoding="utf-8")
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "does not match its own" in loaded.excluded[0].reason


def test_a_sidecar_with_an_unknown_schema_version_excludes_the_run(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1")
    edit_sidecar(tmp_path, "api-1", lambda p: p.update(schema_version="99"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "schema_version" in loaded.excluded[0].reason


def test_a_sidecar_with_no_schema_version_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1")
    edit_sidecar(tmp_path, "api-1", lambda p: p.pop("schema_version"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "no schema_version" in loaded.excluded[0].reason


def test_a_sidecar_describing_another_run_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1")
    edit_sidecar(tmp_path, "api-1", lambda p: p.update(run_id="some-other-run"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "describes a different run" in loaded.excluded[0].reason


def test_a_sidecar_describing_another_prompt_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["workload_hash"] = "sha256:a-different-prompt"

    edit_sidecar(tmp_path, "api-1", mutate)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "plan.workload_hash" in loaded.excluded[0].reason


def test_a_sidecar_describing_another_configuration_excludes_the_run(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["config_hash"] = "a-different-configuration"

    edit_sidecar(tmp_path, "api-1", mutate)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "plan.config_hash" in loaded.excluded[0].reason


# --- Finding 5: a missing cached count is not zero ------------------------


def _entry(**overrides: object) -> PricingEntry:
    payload: dict[str, object] = {
        "entry_id": "e",
        "provider": "z-ai",
        "model_id": "glm-5.3",
        "currency": "USD",
        "effective_at": "2026-01-01",
        "source": "illustrative example",
        "rates_are_illustrative": True,
        "input_per_million": 1.0,
        "output_per_million": 2.0,
    }
    payload.update(overrides)
    return PricingEntry.from_dict(payload)


def test_a_cached_rate_without_a_cached_count_makes_cost_unavailable() -> None:
    """Assuming none of the prompt was cached would overstate the cost."""
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=0),
        _entry(cached_input_per_million=0.1),
    )
    assert breakdown.amount is None
    assert any("refusing to assume" in reason for reason in breakdown.reasons)


def test_a_cached_rate_with_a_reported_zero_still_prices() -> None:
    """An explicit zero is evidence; an absent field is not."""
    breakdown = estimate_run_cost(
        TokenUsage(
            prompt_tokens=1_000_000, completion_tokens=0, cached_prompt_tokens=0
        ),
        _entry(cached_input_per_million=0.1),
    )
    assert breakdown.amount == pytest.approx(1.0)


def test_no_cached_rate_and_no_cached_count_is_unaffected() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=0), _entry()
    )
    assert breakdown.amount == pytest.approx(1.0)


# --- Finding 6: no partial usage totals ------------------------------------


def test_a_run_without_usage_nulls_the_totals_rather_than_summing_the_rest(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "a", prompt_tokens=1000, completion_tokens=400)
    write_api_run(tmp_path, "b", usage_reported=False)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    usage = report.strata[0].ranked[0].usage
    assert usage is not None
    assert usage.runs_reporting_usage == 1
    assert usage.runs_total == 2
    assert usage.complete is False
    # The one reporting run's 1000 must not be published as the total.
    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_totals_are_summed_when_every_run_reports(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", prompt_tokens=1000, completion_tokens=400)
    write_api_run(tmp_path, "b", prompt_tokens=1500, completion_tokens=600)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    usage = report.strata[0].ranked[0].usage
    assert usage is not None
    assert usage.complete is True
    assert usage.input_tokens == 2500
    assert usage.output_tokens == 1000


# --- Finding 7: cost objectives use cost dispersion ------------------------


def test_a_cost_objective_is_inconclusive_when_costs_overlap(
    tmp_path: Path,
) -> None:
    """Steady latency must not make a noisy price look decisive."""
    for index, tokens in enumerate((400_000, 1_600_000), start=1):
        write_api_run(
            tmp_path,
            f"flash-{index}",
            model_id="glm-5.3-flash",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=tokens,
        )
    for index, tokens in enumerate((450_000, 1_650_000), start=1):
        write_api_run(
            tmp_path,
            f"frontier-{index}",
            model_id="glm-5.3",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=tokens,
            config_hash="frontier-cfg",
        )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(CompareObjective.MIN_COST_PER_CORRECT_CASE),
        pricing=_PRICING,
        pricing_manifest_path="rates.json",
    )
    stratum = report.strata[0]
    # Latency is identical across every run, so a latency-derived band would
    # be zero and the tiny cost gap would read as a decisive win.
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert "measurement noise" in (stratum.inconclusive_reason or "")


def test_a_cost_objective_is_conclusive_when_costs_are_far_apart(
    tmp_path: Path,
) -> None:
    for index in (1, 2):
        write_api_run(
            tmp_path,
            f"flash-{index}",
            model_id="glm-5.3-flash",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=1000,
        )
    for index in (1, 2):
        write_api_run(
            tmp_path,
            f"frontier-{index}",
            model_id="glm-5.3",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=1_000_000,
            config_hash="frontier-cfg",
        )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(CompareObjective.MIN_COST_PER_CORRECT_CASE),
        pricing=_PRICING,
        pricing_manifest_path="rates.json",
    )
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.RECOMMENDED
    assert stratum.recommended is not None
    assert stratum.recommended.system_key.model_id == "glm-5.3-flash"


# --- Finding 9: strict numerics and an exact pricing schema version -------


def test_pricing_requires_an_explicit_schema_version() -> None:
    with pytest.raises(PricingError, match="schema_version"):
        PricingManifest.from_dict(
            {
                "currency": "USD",
                "entries": [_entry().to_dict()],
            }
        )


def test_pricing_refuses_a_non_string_schema_version() -> None:
    with pytest.raises(PricingError, match="must be a string"):
        PricingManifest.from_dict(
            {
                "schema_version": 1,
                "currency": "USD",
                "entries": [_entry().to_dict()],
            }
        )


def test_pricing_refuses_an_unknown_schema_version() -> None:
    with pytest.raises(PricingError, match="unsupported pricing manifest"):
        PricingManifest.from_dict(
            {
                "schema_version": "99",
                "currency": "USD",
                "entries": [_entry().to_dict()],
            }
        )


def test_a_float_field_too_large_to_represent_is_a_validation_error() -> None:
    """``float()`` raises OverflowError, which is not a ValueError."""
    payload = json.loads(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": "r",
                "plan": {"provider": "p", "messages": []},
                "usage": {"reported": False},
                "timeline": {},
            }
        )
    )
    payload["timeline"]["first_content_token_offset_ms"] = 10**400
    with pytest.raises(ApiEvidenceError, match="too large to represent"):
        ApiEvidence.from_dict(payload)


def test_a_pricing_rate_too_large_to_represent_is_a_validation_error() -> None:
    with pytest.raises(PricingError, match="too large to represent"):
        PricingEntry.from_dict({**_entry().to_dict(), "input_per_million": 10**400})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_pass_rate", 10**400),
        ("min_quality_score", 10**400),
        ("max_mean_total_latency_ms", 10**400),
    ],
)
def test_a_policy_threshold_too_large_to_represent_is_a_validation_error(
    field: str, value: int
) -> None:
    with pytest.raises(ComparePolicyError, match="too large to represent"):
        CompareConstraints.from_dict({field: value})


def test_a_unit_key_sampling_value_too_large_to_represent_is_refused() -> None:
    with pytest.raises(CompareIdentityError, match="too large to represent"):
        ComparableUnitKey.from_dict(
            {
                "workload_id": "w",
                "workload_version": "1",
                "workload_prompt_hash": "sha256:abc",
                "context_tier": "2k",
                "temperature": 10**400,
            }
        )


def test_a_report_measurement_too_large_to_represent_is_refused(
    tmp_path: Path,
) -> None:
    """Both the required and the optional report float readers are guarded."""
    write_run(tmp_path, "local-1")
    write_api_run(tmp_path, "api-1")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    payload = json.loads(report.to_json())

    optional = json.loads(json.dumps(payload))
    optional["strata"][0]["ranked"][0]["mean_total_latency_ms"] = 10**400
    with pytest.raises(CompareReportValidationError, match="too large to represent"):
        CompareReport.from_dict(optional)

    required = json.loads(json.dumps(payload))
    required["strata"][0]["ranked"][0]["objective_value"] = 10**400
    with pytest.raises(CompareReportValidationError, match="too large to represent"):
        CompareReport.from_dict(required)


def test_the_cli_refuses_an_unrepresentable_pricing_rate_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This used to escape the CLI as an unhandled OverflowError traceback."""
    from llmtracefx.optimizer.cli import main

    results = tmp_path / "results"
    write_run(results, "local-1")
    write_api_run(results, "api-1")
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    manifest = write_json(
        tmp_path / "rates.json",
        {
            "schema_version": "1",
            "currency": "USD",
            "entries": [{**_entry().to_dict(), "input_per_million": 10**400}],
        },
    )
    try:
        main(
            [
                "compare",
                "--results",
                str(results),
                "--policy",
                str(policy),
                "--pricing",
                str(manifest),
            ]
        )
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 1
    assert "Invalid pricing manifest" in capsys.readouterr().err


def test_the_cli_refuses_an_unrepresentable_report_value_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llmtracefx.optimizer.cli import main

    results = tmp_path / "results"
    write_run(results, "local-1")
    write_api_run(results, "api-1")
    report = compare(results_dirs=(results,), policy=_policy())
    payload = json.loads(report.to_json())
    payload["strata"][0]["ranked"][0]["mean_total_latency_ms"] = 10**400
    corrupted = write_json(tmp_path / "compare.json", payload)

    try:
        main(
            [
                "compare-report",
                "--input",
                str(corrupted),
                "--output",
                str(tmp_path / "out.html"),
            ]
        )
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 1
    assert "Invalid compare report" in capsys.readouterr().err


#: Float coercions in ``compare/`` that provably cannot raise ``OverflowError``
#: and so need no guard. Each is listed with the reason, because an unexplained
#: allowlist is how a ratchet quietly stops ratcheting.
#:
#: * ``decode_settings_from_argv`` coerces an argv **string**. ``float(str)``
#:   returns ``inf`` on overflow rather than raising, and the surrounding
#:   ``math.isfinite`` check drops it; only ``ValueError`` is reachable, and
#:   that is already caught.
#: * The ``temperature``/``top_p`` coercions in the same function receive
#:   values already produced by that guarded string coercion, so they are
#:   floats already and ``float(float)`` cannot overflow.
#: * ``_percentiles`` coerces the median of an already-validated finite list.
_UNGUARDED_FLOAT_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("evidence.py", "decode_settings_from_argv"),
        ("compare.py", "_percentiles"),
    }
)


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str:
    best = "<module>"
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = candidate.end_lineno or candidate.lineno
        if candidate.lineno <= node.lineno <= end:  # type: ignore[attr-defined]
            best = candidate.name
    return best


def _catches_overflow(handler: ast.ExceptHandler) -> bool:
    names: list[str] = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
    return bool({"OverflowError", "ArithmeticError"} & set(names))


def test_every_float_coercion_in_compare_is_guarded_or_allowlisted() -> None:
    """A ratchet, so a new parser cannot reintroduce the CLI crash.

    ``float()`` on a large Python int raises ``OverflowError``, which is an
    ``ArithmeticError`` and therefore caught by none of the CLI's typed
    handlers. Every input this package parses is a user-supplied file it
    treats as untrusted, so a bare coercion is a traceback waiting for a
    malformed manifest, policy or report.

    The pattern matches ``float(<anything>)``, not only ``float(<name>)``, so
    a coercion wrapped around a call cannot slip past.
    """
    package = Path(evidence_module.__file__).parent
    unguarded: list[str] = []

    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "float"
                and node.args
            ):
                continue
            if isinstance(node.args[0], ast.Constant):
                continue
            guarded = any(
                block.body
                and block.body[0].lineno
                <= node.lineno
                <= (block.body[-1].end_lineno or node.lineno)
                and any(_catches_overflow(handler) for handler in block.handlers)
                for block in tries
            )
            if guarded:
                continue
            where = (path.name, _enclosing_function(tree, node))
            if where in _UNGUARDED_FLOAT_ALLOWLIST:
                continue
            unguarded.append(f"{path.name}:{node.lineno} in {where[1]}")

    assert not unguarded, (
        "unguarded float() coercion of untrusted input; wrap it in "
        "try/except OverflowError raising this module's typed error, or add "
        "it to _UNGUARDED_FLOAT_ALLOWLIST with a reason: " + ", ".join(unguarded)
    )


def test_the_float_guard_ratchet_would_catch_a_new_bare_coercion() -> None:
    """The ratchet must actually bite, not just pass vacuously."""
    source = "def parse(value):\n    numeric = float(value)\n    return numeric\n"
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "float"
    ]
    assert len(calls) == 1
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert _enclosing_function(tree, calls[0]) == "parse"


def test_the_cli_reports_a_raw_collector_directory_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llmtracefx.optimizer.cli import main

    raw = tmp_path / "collector-out"
    raw.mkdir()
    (raw / "api_evidence.json").write_text("{}", encoding="utf-8")
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    try:
        main(["compare", "--results", str(raw), "--policy", str(policy)])
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 1
    assert "raw collector output" in capsys.readouterr().err


# --- Second review round -------------------------------------------------


def test_a_relative_results_path_resolves_without_double_prefixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`verify` records collection_dir relative to the working directory.

    Re-joining it under the results directory doubled the prefix, so every
    hosted run silently lost its API evidence and was compared as though the
    provider had reported nothing.
    """
    monkeypatch.chdir(tmp_path)
    results = Path("artifacts/run")
    write_api_run(results, "api-1")

    verification_path = results / "runs" / "api-1" / "verification.json"
    payload = json.loads(verification_path.read_text(encoding="utf-8"))
    # Exactly what verify writes for a relative --output-dir.
    payload["collection_dir"] = "artifacts/run/runs/api-1/collection"
    payload["final_record_path"] = "artifacts/run/runs/api-1/final_record.json"
    verification_path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_run(results, "api-1")

    loaded = load_comparison_evidence((results,))
    assert loaded.excluded == ()
    assert len(loaded.runs) == 1
    run = loaded.runs[0]
    assert run.api_evidence is not None
    assert run.api_evidence.usage.prompt_tokens == 1000


def test_the_cli_compares_relative_results_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from llmtracefx.optimizer.cli import main

    monkeypatch.chdir(tmp_path)
    results = Path("artifacts/run")
    write_run(results, "local-1", total_ms=8000.0)
    write_api_run(results, "api-1", total_ms=1200.0)
    for run_id in ("local-1", "api-1"):
        path = results / "runs" / run_id / "verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["collection_dir"] = f"artifacts/run/runs/{run_id}/collection"
        payload["final_record_path"] = f"artifacts/run/runs/{run_id}/final_record.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reseal_run(results, run_id)

    policy = write_json(Path("policy.json"), COMPARE_POLICY)
    try:
        main(["compare", "--results", str(results), "--policy", str(policy)])
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    stdout = capsys.readouterr().out
    assert code == 0, stdout
    # Provider-reported usage proves the sidecar was actually found.
    assert "provider-reported usage" in stdout


def test_a_sidecar_missing_a_required_identity_field_excludes_the_run(
    tmp_path: Path,
) -> None:
    """A v1 sidecar always carries these, so an absent one is not trustworthy."""
    for field in ("run_id",):
        write_api_run(tmp_path, "api-1")
        edit_sidecar(tmp_path, "api-1", lambda p, f=field: p.pop(f))
        loaded = load_comparison_evidence((tmp_path,))
        assert loaded.runs == ()
        assert "records no run_id" in loaded.excluded[0].reason


@pytest.mark.parametrize("field", ["model_id", "workload_hash", "config_hash"])
def test_a_sidecar_missing_a_required_plan_field_excludes_the_run(
    tmp_path: Path, field: str
) -> None:
    write_api_run(tmp_path, "api-1")
    edit_sidecar(tmp_path, "api-1", lambda p, f=field: p["plan"].pop(f))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert f"records no plan.{field}" in loaded.excluded[0].reason


def test_a_sidecar_revision_the_record_lacks_excludes_the_run(
    tmp_path: Path,
) -> None:
    """Nullability must agree, not just values that both happen to exist."""
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["model_revision"] = "2026-06"

    edit_sidecar(tmp_path, "api-1", mutate)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "disagree about whether a model revision" in loaded.excluded[0].reason


def test_an_artifact_marker_from_another_run_excludes_the_run(
    tmp_path: Path,
) -> None:
    """Hash completeness proves the files match, not that they are this run's."""
    write_api_run(tmp_path, "api-1")
    marker = collection_dir_for(tmp_path, "api-1") / "artifacts.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["run_id"] = "a-different-run"
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "belongs to a different run" in loaded.excluded[0].reason


def test_runs_differing_only_in_sidecar_evidence_are_not_deduplicated(
    tmp_path: Path,
) -> None:
    """TTFT, cache and reasoning live only in the sidecar.

    These two runs are distinct artifacts, so both are kept as repetitions.
    Fingerprinting just the verification and the record would call them
    identical, and the byte-identical rule would then drop one of them and
    take its sidecar usage and timing with it.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_api_run(first, "api-1", first_content_token_offset_ms=200.0)
    write_api_run(second, "api-1", first_content_token_offset_ms=900.0)
    loaded = load_comparison_evidence((first, second))
    assert len(loaded.runs) == 2
    ttfts = {
        run.api_evidence.client_ttft_ms
        for run in loaded.runs
        if run.api_evidence is not None
    }
    assert ttfts == {200.0, 900.0}


def test_byte_identical_runs_still_deduplicate(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for target in (first, second):
        write_api_run(target, "api-1", first_content_token_offset_ms=200.0)
    # The two trees differ only in the absolute paths recorded inside them,
    # so this is the copied-results case rather than a genuine conflict.
    loaded = load_comparison_evidence((first,))
    assert len(loaded.runs) == 1


def test_an_exclusion_reason_never_publishes_an_absolute_path(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, "good")
    write_api_run(tmp_path, "other")
    write_api_run(tmp_path, "broken", write_artifact_marker=False)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    document = render_compare_report_html(report)
    assert str(tmp_path) not in document
    assert "Excluded runs" in document


def test_prose_path_redaction_keeps_a_filename_and_hides_a_directory() -> None:
    """A filename is a useful handle; an anchored directory is a leak."""
    text = "could not read /home/alice/SECRET/runs/x/record.json here"
    redacted = _redact_paths_in_prose(text, redact_paths=True)
    assert "SECRET" not in redacted
    assert "record.json here" in redacted
    assert _redact_paths_in_prose(text, redact_paths=False) == text

    directory = "the artifact set in /home/alice/SECRET/runs/x/collection is broken"
    assert (
        _redact_paths_in_prose(directory, redact_paths=True)
        == "the artifact set in <path> is broken"
    )


def test_cost_noise_scales_per_attempt_dispersion_by_the_pass_rate(
    tmp_path: Path,
) -> None:
    """cost_per_correct_case divides by correct cases, not by attempts.

    With half the attempts failing, a swing of x per attempt moves the
    objective by 2x, so the raw per-attempt dispersion understates the band.
    """
    for index, tokens in enumerate((400_000, 1_600_000), start=1):
        write_api_run(
            tmp_path,
            f"flash-{index}",
            model_id="glm-5.3-flash",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=tokens,
            success=index == 1,
            quality_score=1.0 if index == 1 else 0.0,
        )
    for index, tokens in enumerate((500_000, 1_700_000), start=1):
        write_api_run(
            tmp_path,
            f"frontier-{index}",
            model_id="glm-5.3",
            total_ms=1000.0,
            prompt_tokens=0,
            completion_tokens=tokens,
            config_hash="frontier-cfg",
            success=index == 1,
            quality_score=1.0 if index == 1 else 0.0,
        )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(CompareObjective.MIN_COST_PER_CORRECT_CASE, min_pass_rate=0.5),
        pricing=_PRICING,
        pricing_manifest_path="rates.json",
    )
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert "measurement noise" in (stratum.inconclusive_reason or "")


@pytest.mark.parametrize(
    ("loader", "error"),
    [
        ("policy", ComparePolicyError),
        ("pricing", PricingError),
        ("report", CompareReportValidationError),
    ],
)
def test_json_limits_are_typed_errors_not_tracebacks(
    loader: str, error: type[Exception]
) -> None:
    """A huge int literal raises ValueError, deep nesting RecursionError."""
    from llmtracefx.optimizer.compare.policy import ComparePolicy

    payloads = {
        "policy": ComparePolicy.from_json,
        "pricing": PricingManifest.from_json,
        "report": CompareReport.from_json,
    }
    huge = '{"schema_version": ' + "1" * 5000 + "}"
    deep = "[" * 200_000 + "]" * 200_000
    for text in (huge, deep):
        with pytest.raises(error):
            payloads[loader](text)


def test_a_broken_run_level_seal_excludes_the_run(tmp_path: Path) -> None:
    """`run-api` seals verification.json and final_record.json too.

    The collector's own marker covers only the four files in the collection
    directory, leaving the graded outcome and the summary this loader reads
    outside every integrity check. When the run-level seal is present it is
    enforced, so editing either of those is caught.
    """
    write_api_run(tmp_path, "api-1")
    run_dir = tmp_path / "runs" / "api-1"
    collection = collection_dir_for(tmp_path, "api-1")
    marker = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "api-1",
        "artifacts": [
            {
                "name": f"collection/{ARTIFACT_MANIFEST_NAME}",
                "sha256": sha256_bytes(
                    (collection / ARTIFACT_MANIFEST_NAME).read_bytes()
                ),
            },
            {
                "name": "final_record.json",
                "sha256": sha256_bytes((run_dir / "final_record.json").read_bytes()),
            },
            {
                "name": "verification.json",
                "sha256": sha256_bytes((run_dir / "verification.json").read_bytes()),
            },
        ],
    }
    (run_dir / RUN_MANIFEST_NAME).write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )

    # Sealed and untouched: the run loads.
    assert len(load_comparison_evidence((tmp_path,)).runs) == 1

    # Regrade the outcome without regenerating the seal.
    record_path = run_dir / "final_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["outcome"]["quality_score"] = 0.0
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "seal" in loaded.excluded[0].reason
    assert RUN_MANIFEST_NAME in loaded.excluded[0].reason


def test_a_local_run_without_a_run_level_seal_is_still_accepted(
    tmp_path: Path,
) -> None:
    """`workloads run` writes no such marker, so it must not be required."""
    write_run(tmp_path, "local-1")
    assert not (tmp_path / "runs" / "local-1" / RUN_MANIFEST_NAME).exists()
    assert len(load_comparison_evidence((tmp_path,)).runs) == 1


# --- Third review round --------------------------------------------------


def test_a_relative_recording_is_never_read_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standing in one results tree must not poison a comparison of another.

    A relative `--output-dir` records paths relative to the working directory
    of *that* run. Reading them literally resolves them against whichever
    tree this process happens to be in, and every identity check passes
    because a matrix run_id names the task, not the system. The result is a
    confidently labelled comparison of the wrong system.
    """
    here = tmp_path / "tree-a"
    other = tmp_path / "tree-b"
    for target, total_ms, model in (
        (here, 1000.0, "model-a"),
        (other, 9999.0, "model-b"),
    ):
        write_run(target, "shared-row", total_ms=total_ms, model_id=model)
        path = target / "runs" / "shared-row" / "verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Exactly what a relative --output-dir records.
        payload["final_record_path"] = "runs/shared-row/final_record.json"
        payload["collection_dir"] = "runs/shared-row/collection"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reseal_run(target, "shared-row")

    monkeypatch.chdir(here)
    loaded = load_comparison_evidence((other,))
    assert len(loaded.runs) == 1
    run = loaded.runs[0]
    assert run.system_key.model_id == "model-b"
    assert run.total_ms == pytest.approx(9999.0)


def test_two_copies_of_one_matrix_tree_do_not_collapse_into_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copied-tree case must resolve each tree against itself."""
    monkeypatch.chdir(tmp_path)
    mine = Path("artifacts/mine")
    theirs = Path("incoming/theirs")
    write_api_run(mine, "shared-row", model_id="glm-5.3", total_ms=1000.0)
    write_api_run(
        theirs,
        "shared-row",
        model_id="glm-5.3-flash",
        total_ms=4000.0,
        config_hash="other-cfg",
    )
    for target in (mine, theirs):
        path = target / "runs" / "shared-row" / "verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["final_record_path"] = "runs/shared-row/final_record.json"
        payload["collection_dir"] = "runs/shared-row/collection"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reseal_run(target, "shared-row")

    loaded = load_comparison_evidence((mine, theirs))
    assert len(loaded.runs) == 2
    assert {run.system_key.model_id for run in loaded.runs} == {
        "glm-5.3",
        "glm-5.3-flash",
    }


def test_a_relative_ancestor_never_survives_into_the_shared_report(
    tmp_path: Path,
) -> None:
    """The documented workflow uses a relative --output-dir, so this is the
    common leak shape, not an exotic one."""
    results = tmp_path / "acme-secret-client" / "eval-2026"
    write_run(results, "good")
    write_api_run(results, "other")
    write_run(results, "bad-1", corrupt_final_record=True)
    report = compare(results_dirs=(results,), policy=_policy())
    document = render_compare_report_html(report)
    # The ancestor is the private part and must not survive anywhere.
    assert "acme-secret-client" not in document
    # The final component is deliberately kept: it is what tells two inputs
    # apart in the report, and it names no ancestor.
    assert "eval-2026" in document


def test_the_prose_scrubber_leaves_ordinary_words_alone() -> None:
    """A naive contains-a-slash rule mangles 'read/parse' into 'readparse'."""
    text = "could not read/parse verification.json: boom"
    assert _redact_paths_in_prose(text, redact_paths=True) == text


@pytest.mark.parametrize(
    ("text", "must_not_contain"),
    [
        ("failed at acme-client/eval/runs/x/record.json now", "acme-client"),
        ("failed at /home/alice/SECRET/runs/x/collection now", "SECRET"),
        ("failed at run/record.json now", "run/record.json"),
    ],
)
def test_the_prose_scrubber_reduces_paths_to_their_final_component(
    text: str, must_not_contain: str
) -> None:
    redacted = _redact_paths_in_prose(text, redact_paths=True)
    assert must_not_contain not in redacted
    assert redacted.startswith("failed at ")
    assert redacted.endswith(" now")


@pytest.mark.parametrize("suffix", ["yaml", "yml"])
def test_a_malformed_yaml_policy_is_a_typed_error(tmp_path: Path, suffix: str) -> None:
    """`yaml.YAMLError` is not a ValueError, so it escaped every handler."""
    from llmtracefx.optimizer.compare.policy import ComparePolicy

    path = tmp_path / f"policy.{suffix}"
    path.write_text("objective: [unclosed", encoding="utf-8")
    with pytest.raises(ComparePolicyError, match="invalid YAML"):
        ComparePolicy.from_file(path)


def test_a_malformed_yaml_pricing_manifest_is_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "rates.yaml"
    path.write_text("entries: [unclosed", encoding="utf-8")
    with pytest.raises(PricingError, match="invalid YAML"):
        PricingManifest.from_file(path)


def test_the_cli_refuses_a_malformed_yaml_policy_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from llmtracefx.optimizer.cli import main

    results = tmp_path / "results"
    write_run(results, "local-1")
    write_api_run(results, "api-1")
    policy = tmp_path / "policy.yaml"
    policy.write_text("objective: [unclosed", encoding="utf-8")
    try:
        main(["compare", "--results", str(results), "--policy", str(policy)])
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 1
    assert "Invalid compare policy" in capsys.readouterr().err


def test_a_deeply_nested_verification_is_a_typed_error(tmp_path: Path) -> None:
    """RecursionError is not a ValueError and escaped the loader."""
    write_run(tmp_path, "local-1")
    path = tmp_path / "runs" / "local-1" / "verification.json"
    depth = 200_000
    path.write_text("[" * depth + "]" * depth, encoding="utf-8")
    # Either excluded with a reason or refused as a typed input error, but
    # never a traceback.
    try:
        loaded = load_comparison_evidence((tmp_path,))
    except CompareEvidenceError:
        return
    assert loaded.runs == ()


# ---------------------------------------------------------------------------
# Exact-head review, eighth round.
#
# The finding that drove most of this round: identity is not what
# de-duplicates evidence, the artifact is. A matrix ``run_id`` names the task,
# so every repetition of a matrix shares it, and keying de-duplication on
# identity therefore rejected the ordinary way repetitions are collected.
# ---------------------------------------------------------------------------


def test_repetitions_from_distinct_trees_satisfy_a_repetition_requirement(
    tmp_path: Path,
) -> None:
    """Finding 1. ``min_measured_repetitions`` above one must be reachable.

    Three executions of one matrix row into three results directories are
    three repetitions. Previously the second and third were either rejected
    as conflicts (when the timings differed, which they always do) or
    deduplicated away (when they did not), so no policy asking for more than
    one measurement could ever be satisfied.
    """
    trees = []
    for index, total in enumerate((8000.0, 8250.0, 8100.0)):
        tree = tmp_path / f"rep-{index}"
        write_run(tree, "shared", total_ms=total)
        trees.append(tree)
    hosted = []
    for index, total in enumerate((1200.0, 1260.0, 1230.0)):
        tree = tmp_path / f"hosted-{index}"
        write_api_run(tree, "shared", total_ms=total)
        hosted.append(tree)

    policy = ComparePolicy.from_dict(
        {**COMPARE_POLICY, "constraints": {"min_measured_repetitions": 3}}
    )
    report = compare(results_dirs=tuple(trees + hosted), policy=policy)
    ranked = report.strata[0].ranked
    assert len(ranked) == 2
    assert {entry.evidence_count for entry in ranked} == {3}
    for entry in ranked:
        assert not any(
            "requires at least" in reason for reason in entry.missing_evidence
        )


def test_one_directory_reached_through_two_paths_is_counted_once(
    tmp_path: Path,
) -> None:
    """Finding 1, the other side. An alias is not a repetition."""
    tree = tmp_path / "results"
    write_run(tree, "local-1")
    alias = tmp_path / "alias"
    alias.symlink_to(tree, target_is_directory=True)
    loaded = load_comparison_evidence((tree, alias, tree))
    assert len(loaded.runs) == 1


def test_a_copied_results_tree_is_not_counted_twice(tmp_path: Path) -> None:
    """Finding 1, the third case. A copy is not a second measurement.

    Two genuinely separate executions cannot share ``started_at`` and
    ``ended_at`` down to the byte, so a byte-identical record is a copied
    tree. Counting it as a repetition would inflate the evidence count and
    understate every dispersion derived from it.
    """
    import shutil

    first = tmp_path / "a"
    write_run(first, "local-1", total_ms=1000.0)
    second = tmp_path / "b"
    shutil.copytree(first, second)
    loaded = load_comparison_evidence((first, second))
    assert len(loaded.runs) == 1


def test_a_hosted_run_without_a_sidecar_is_excluded_loudly(tmp_path: Path) -> None:
    """Finding 3. A hosted run with no API evidence is incomplete, not local.

    The record names a provider, so the sidecar exists by construction. Its
    absence used to be read as "this run simply has no API evidence", which
    skipped artifact-set validation entirely and published the run with no
    endpoint, no request identity, no provider usage and no TTFT.
    """
    import shutil

    write_api_run(tmp_path, "api-1")
    shutil.rmtree(collection_dir_for(tmp_path, "api-1"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    reason = loaded.excluded[0].reason
    assert "hosted API" in reason
    assert "missing" in reason


def test_a_hosted_run_with_the_sidecar_file_removed_is_excluded(
    tmp_path: Path,
) -> None:
    """Finding 3. Same rule when only the sidecar file is gone."""
    write_api_run(tmp_path, "api-1")
    (collection_dir_for(tmp_path, "api-1") / "api_evidence.json").unlink()
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "hosted API" in loaded.excluded[0].reason


def test_throughput_noise_ignores_the_timings_of_failed_runs(
    tmp_path: Path,
) -> None:
    """Finding 4. Correct cases per minute counts only passing timed runs.

    Its noise band has to come from those same samples. This system passes
    twice, steadily, and fails once after a minute. The failure contributes
    nothing to the throughput figure, but it dominated the all-runs
    coefficient of variation, and borrowing that figure inflated the noise
    band far past the real gap between the two systems: a decisive result
    was reported as a tie because of evidence the objective never used.
    """
    fast = tmp_path / "fast"
    for index, (total, ok) in enumerate(
        ((1000.0, True), (1005.0, True), (60000.0, False))
    ):
        write_run(fast / f"r{index}", "shared", total_ms=total, success=ok)
    slow = tmp_path / "slow"
    for index, total in enumerate((3000.0, 3010.0)):
        write_api_run(slow / f"r{index}", "shared", total_ms=total)

    dirs = tuple(sorted(fast.glob("r*"))) + tuple(sorted(slow.glob("r*")))
    policy = ComparePolicy.from_dict(
        {
            **COMPARE_POLICY,
            "objective": "max_correct_cases_per_minute",
            "constraints": {},
        }
    )
    report = compare(results_dirs=dirs, policy=policy)
    stratum = report.strata[0]

    # The dispersion of the two passing runs is three orders of magnitude
    # smaller than the dispersion of all three, which is the whole point.
    from statistics import mean, pstdev

    passing = [1000.0, 1005.0]
    every = [1000.0, 1005.0, 60000.0]
    assert pstdev(passing) / mean(passing) < (pstdev(every) / mean(every)) / 100

    assert stratum.outcome is StratumOutcome.RECOMMENDED
    assert stratum.ranked[0].correct_cases_per_minute is not None
    assert stratum.ranked[0].correct_cases_per_minute > (
        stratum.ranked[1].correct_cases_per_minute or 0.0
    )


def test_prose_redaction_hides_ancestors_of_a_path_containing_whitespace() -> None:
    """Finding 5. A space split the path and published the private half."""
    text = (
        "the artifact set in /Users/secret client/results/runs/x/collection "
        "is broken"
    )
    redacted = _redact_paths_in_prose(text, redact_paths=True)
    assert "secret" not in redacted
    assert "/Users" not in redacted


def test_prose_redaction_hides_a_unc_and_a_quoted_path() -> None:
    """Finding 5. Windows, UNC and quoted forms redact too."""
    unc = _redact_paths_in_prose(
        r"could not read \\server\share\SECRET-CLIENT here", redact_paths=True
    )
    assert "SECRET-CLIENT" not in unc

    windows = _redact_paths_in_prose(
        r'could not read "C:\Users\SECRET\runs\x\record.json" here',
        redact_paths=True,
    )
    assert "SECRET" not in windows
    assert "record.json" in windows


def test_a_list_valued_identity_field_is_excluded_not_a_typeerror(
    tmp_path: Path,
) -> None:
    """Finding 6. Identity fields reach dict keys, so they must be strings.

    Nothing upstream promises that: ``RowVerification.from_dict`` copies the
    value through, so a list here used to surface far away as ``TypeError:
    unhashable type: 'list'`` with no run named in the message.
    """
    for field in ("workload_id", "context_tier", "decode_mode"):  # noqa: B007
        tree = tmp_path / field
        write_run(tree, "row-1")
        path = tree / "runs" / "row-1" / "verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = ["a", "b"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_comparison_evidence((tree,))
        assert loaded.runs == ()
        assert field in loaded.excluded[0].reason


def test_the_artifact_marker_is_read_through_the_bounded_reader(
    tmp_path: Path,
) -> None:
    """Finding 7. The marker must not be read around the collector's guard.

    The collector reads this file through a bounded regular-file reader that
    refuses a symlink, a device node, or a file large enough to be a denial
    of service. Parsing it here with a bare ``read_text`` first stepped
    around exactly that guard.
    """
    source = ast.parse(
        Path(evidence_module.__file__).read_text(encoding="utf-8"),
        filename=evidence_module.__file__,
    )
    marker_fn = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_artifact_marker_identity_error"
    )
    calls = {
        node.func.attr
        for node in ast.walk(marker_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_text" not in calls, "the marker must go through the bounded reader"
    names = {
        node.func.id
        for node in ast.walk(marker_fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_read_bounded_regular_file" in names


def test_an_oversized_artifact_marker_is_refused(tmp_path: Path) -> None:
    """Finding 7, behaviourally. Past the limit the marker is not parsed."""
    from llmtracefx.optimizer.collectors.openai_api import (
        _MAX_ARTIFACT_MANIFEST_BYTES,
    )

    write_api_run(tmp_path, "api-1")
    marker = collection_dir_for(tmp_path, "api-1") / ARTIFACT_MANIFEST_NAME
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["padding"] = "x" * (_MAX_ARTIFACT_MANIFEST_BYTES + 1024)
    marker.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    # An oversized marker also fails the completeness check, so asserting
    # only that the run was excluded would hold with the bound removed.
    assert "byte limit" in loaded.excluded[0].reason


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf"), -0.5, -1])
def test_a_pricing_entry_refuses_a_bad_rate_on_direct_construction(
    rate: float,
) -> None:
    """Finding 8. ``from_dict`` is not the only way in.

    ``PricingEntry`` is public, so a caller can construct one directly and
    bypass the loader's validation entirely. A negative or non-finite rate
    accepted here poisons every cost derived from it.
    """
    with pytest.raises(PricingError):
        PricingEntry(
            entry_id="illustrative",
            provider="example-provider",
            model_id="example-model",
            currency="USD",
            effective_at="2026-01-01",
            source="https://example.invalid/pricing",
            rates_are_illustrative=True,
            input_per_million=rate,
        )


def test_a_pricing_entry_refuses_a_non_numeric_rate_on_direct_construction() -> None:
    """Finding 8. A string or a bool is not a rate either."""
    for bad in ("1.0", True):
        with pytest.raises(PricingError):
            PricingEntry(
                entry_id="illustrative",
                provider="example-provider",
                model_id="example-model",
                currency="USD",
                effective_at="2026-01-01",
                source="https://example.invalid/pricing",
                rates_are_illustrative=True,
                output_per_million=bad,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Exact-head review, ninth round. Containment, constructor invariants, and
# the two ways a report can be internally false.
# ---------------------------------------------------------------------------


def test_an_absolute_record_path_cannot_leave_the_selected_results_tree(
    tmp_path: Path,
) -> None:
    """Finding 2. The recorded path is data from the artifact being checked.

    Honouring an absolute one let a results tree serve another tree's
    measurements while every identity check passed, because a matrix
    ``run_id`` names the task and both trees agree on it.
    """
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    write_run(inside, "row-1", total_ms=1000.0)
    write_run(outside, "row-1", total_ms=9999.0)
    (inside / "runs" / "row-1" / "final_record.json").unlink()
    path = inside / "runs" / "row-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_record_path"] = str(outside / "runs" / "row-1" / "final_record.json")
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((inside,))
    assert loaded.runs == ()
    assert "could not read final_record.json" in loaded.excluded[0].reason


def test_a_symlinked_record_cannot_leave_the_selected_results_tree(
    tmp_path: Path,
) -> None:
    """Finding 2. Resolution happens before containment, not after."""
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    write_run(inside, "row-1", total_ms=1000.0)
    write_run(outside, "row-1", total_ms=7777.0)
    canonical = inside / "runs" / "row-1" / "final_record.json"
    canonical.unlink()
    canonical.symlink_to(outside / "runs" / "row-1" / "final_record.json")

    loaded = load_comparison_evidence((inside,))
    assert loaded.runs == ()


def test_a_symlinked_collection_cannot_leave_the_selected_results_tree(
    tmp_path: Path,
) -> None:
    """Finding 2. The same rule for the API collection directory."""
    import shutil

    inside, outside = tmp_path / "inside", tmp_path / "outside"
    write_api_run(inside, "api-1")
    write_api_run(outside, "api-1")
    canonical = collection_dir_for(inside, "api-1")
    shutil.rmtree(canonical)
    canonical.symlink_to(collection_dir_for(outside, "api-1"), target_is_directory=True)

    loaded = load_comparison_evidence((inside,))
    assert loaded.runs == ()


def test_an_ordinary_results_tree_still_loads(tmp_path: Path) -> None:
    """Finding 2, the other direction: containment must not break the norm."""
    write_run(tmp_path, "row-1", total_ms=1234.0)
    loaded = load_comparison_evidence((tmp_path,))
    assert len(loaded.runs) == 1
    assert loaded.runs[0].record.timing.total.value == 1234.0


def test_an_unrepresentable_measurement_is_excluded_not_raised(
    tmp_path: Path,
) -> None:
    """Finding 6. ``OverflowError`` is an ``ArithmeticError``.

    Nothing above catches it, so a numeric literal too large for a float
    escaped the loader as a stack trace instead of excluding one run.
    """
    write_run(tmp_path, "row-1")
    path = tmp_path / "runs" / "row-1" / "final_record.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["timing"]["total"]["value"] = int("9" * 400)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()


def test_a_deeply_nested_run_seal_is_excluded_not_raised(tmp_path: Path) -> None:
    """Finding 6 and 7. Deep nesting raises ``RecursionError`` while parsing.

    The shared verifier reads this marker with an unbounded ``read_text``, so
    it is bounded and parsed defensively here before being handed over.
    """
    from llmtracefx.optimizer.workloads.api_verify import RUN_MANIFEST_NAME

    write_api_run(tmp_path, "api-1")
    seal = tmp_path / "runs" / "api-1" / RUN_MANIFEST_NAME
    seal.write_text("[" * 60_000 + "]" * 60_000, encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert RUN_MANIFEST_NAME in loaded.excluded[0].reason


def test_an_oversized_run_seal_is_refused(tmp_path: Path) -> None:
    """Finding 7. Bounded before parse, for every marker not just one."""
    from llmtracefx.optimizer.collectors.openai_api import (
        _MAX_ARTIFACT_MANIFEST_BYTES,
    )
    from llmtracefx.optimizer.workloads.api_verify import RUN_MANIFEST_NAME

    write_api_run(tmp_path, "api-1")
    seal = tmp_path / "runs" / "api-1" / RUN_MANIFEST_NAME
    seal.write_text("x" * (_MAX_ARTIFACT_MANIFEST_BYTES + 1024), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "byte limit" in loaded.excluded[0].reason


def test_a_negative_time_to_first_token_is_refused(tmp_path: Path) -> None:
    """Finding 9. An offset from the start of a request cannot precede it.

    Unchecked it would flatter its own system on every TTFT comparison.
    """
    write_api_run(tmp_path, "api-1")
    edit_sidecar(
        tmp_path,
        "api-1",
        lambda payload: payload["timeline"].__setitem__(
            "first_content_token_offset_ms", -5.0
        ),
    )
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "must be >= 0" in loaded.excluded[0].reason


@pytest.mark.parametrize(
    "field_name",
    ["prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens"],
)
def test_token_usage_refuses_a_negative_count_on_direct_construction(
    field_name: str,
) -> None:
    """Finding 8. A negative count subtracts from a bill."""
    with pytest.raises(PricingError):
        TokenUsage(**{field_name: -1})


def test_token_usage_refuses_a_non_integer_count() -> None:
    """Finding 8. A bool is an int in Python, and is not a token count."""
    with pytest.raises(PricingError):
        TokenUsage(prompt_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(PricingError):
        TokenUsage(prompt_tokens=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_pass_rate": 5.0},
        {"min_pass_rate": -1.0},
        {"max_mean_total_latency_ms": -3.0},
        {"max_cost_per_correct_case": -0.01},
        {"max_coefficient_of_variation": -0.5},
        # A ceiling of exactly zero is unsatisfiable rather than strict: it
        # rejects every system while reading as a deliberate threshold.
        {"max_mean_total_latency_ms": 0.0},
        {"max_cost_per_correct_case": 0.0},
        # ``bool`` is an ``int``, so this passed the ``< 1`` check and was
        # rendered as "at least True measured run(s)".
        {"min_measured_repetitions": True},
    ],
)
def test_constraints_refuse_an_out_of_range_bound_on_direct_construction(
    kwargs: dict[str, float],
) -> None:
    """Finding 8. An unsatisfiable bound is reported as a real threshold.

    Constructed directly it bypassed the parser, so it would exclude every
    system, or admit every system, while reading as a deliberate choice.
    """
    with pytest.raises(ComparePolicyError):
        CompareConstraints(**kwargs)  # type: ignore[arg-type]


def test_a_valid_constraint_set_still_constructs() -> None:
    """Finding 8, the other direction."""
    constraints = CompareConstraints(min_pass_rate=0.9)
    assert constraints.min_pass_rate == 0.9


def _report_payload(tmp_path: Path) -> dict[str, object]:
    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    report = compare(
        results_dirs=(tmp_path,),
        policy=ComparePolicy.from_dict(
            {**COMPARE_POLICY, "objective": "min_mean_total_latency_ms"}
        ),
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert isinstance(payload, dict)
    return payload


def test_a_report_claiming_an_order_it_does_not_have_is_refused(
    tmp_path: Path,
) -> None:
    """Finding 11. Rank order and objective value are two claims, not one.

    A file can assert both and have them disagree. That is not a schema
    error, which is why it needs checking: such a report renders cleanly and
    reads as authoritative while recommending the system that lost.
    """
    payload = _report_payload(tmp_path)
    ranked = payload["strata"][0]["ranked"]  # type: ignore[index]
    # Swap only the rank labels, so each entry stays self-consistent with
    # its own backing metric and the *ordering* is the single false claim.
    ranked[0]["rank"], ranked[1]["rank"] = ranked[1]["rank"], ranked[0]["rank"]
    with pytest.raises(CompareReportValidationError, match="not ordered by"):
        CompareReport.from_dict(payload)


def test_a_report_whose_objective_contradicts_its_metric_is_refused(
    tmp_path: Path,
) -> None:
    """Finding 11. The objective is defined as a metric printed beside it."""
    payload = _report_payload(tmp_path)
    payload["strata"][0]["ranked"][0]["objective_value"] = 0.001  # type: ignore[index]
    with pytest.raises(CompareReportValidationError, match="contradicts the evidence"):
        CompareReport.from_dict(payload)


def test_a_monetary_objective_is_checked_against_its_cost_column(
    tmp_path: Path,
) -> None:
    """Finding 11, the monetary half.

    ``cost_per_correct_case`` is an ordinary reported column, so a report
    can name an objective value three orders of magnitude away from the cost
    printed next to it and still be internally checkable. Exempting the
    money objectives left exactly the hole this validation exists to close.
    """
    write_api_run(tmp_path / "cheap", "shared", model_id="glm-5.3-flash")
    write_api_run(tmp_path / "dear", "shared", model_id="glm-5.3")
    manifest = tmp_path / "pricing.json"
    report = compare(
        results_dirs=(tmp_path / "cheap", tmp_path / "dear"),
        policy=ComparePolicy.from_dict(
            {**COMPARE_POLICY, "objective": "min_cost_per_correct_case"}
        ),
        pricing=_PRICING,
        pricing_manifest_path=str(manifest),
    )
    payload = json.loads(json.dumps(report.to_dict()))
    ranked = payload["strata"][0]["ranked"]
    if ranked[0].get("cost", {}).get("cost_per_correct_case") is None:
        pytest.skip("no priced evidence in this fixture")

    assert CompareReport.from_dict(json.loads(json.dumps(payload))).strata
    ranked[0]["objective_value"] = 3e-07
    with pytest.raises(CompareReportValidationError, match="contradicts the evidence"):
        CompareReport.from_dict(payload)


def test_an_honest_report_round_trips(tmp_path: Path) -> None:
    """Finding 11, the other direction: real reports must still load."""
    payload = _report_payload(tmp_path)
    assert CompareReport.from_dict(payload).strata


def test_every_verdict_states_the_constraints_it_cleared(tmp_path: Path) -> None:
    """Finding 10. The README promises this and neither renderer did it.

    "Cleared every constraint" is unreadable on its own: there is no way to
    tell a demanding comparison from one that constrained nothing.
    """
    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    policy = ComparePolicy.from_dict(
        {
            **COMPARE_POLICY,
            "constraints": {
                "min_pass_rate": 0.5,
                "max_mean_total_latency_ms": 30000.0,
            },
        }
    )
    report = compare(results_dirs=(tmp_path,), policy=policy)

    text = format_compare_report_text(report, verbose=False)
    assert "Constraints in force:" in text
    assert "pass rate >= 0.5" in text
    assert "mean total latency <= 30000.0 ms" in text

    document = render_compare_report_html(report)
    assert "Constraints in force" in document
    assert "pass rate &gt;= 0.5" in document
    assert "mean total latency &lt;= 30000.0 ms" in document


def test_an_unset_constraint_is_not_claimed_as_a_bar(tmp_path: Path) -> None:
    """Finding 10. Naming an unset constraint would overstate the bar."""
    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    report = compare(
        results_dirs=(tmp_path,),
        policy=ComparePolicy.from_dict({**COMPARE_POLICY, "constraints": {}}),
    )
    text = format_compare_report_text(report, verbose=False)
    assert "Constraints in force:" in text
    assert "mean total latency" not in text
    assert "cost per correct case" not in text


@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        ("model", "model_id"),
        ("model", "model_revision"),
        ("model", "quantization"),
        ("runtime", "name"),
        ("runtime", "backend"),
        ("runtime", "provider"),
        ("platform", "accelerator"),
        ("command", "config_hash"),
        ("outcome", "quality_metric"),
    ],
)
def test_a_list_valued_record_identity_field_is_excluded_not_a_typeerror(
    tmp_path: Path, section: str, field_name: str
) -> None:
    """The record is an identity source too, not just the verification.

    Nine values read off ``final_record.json`` end up inside ``SystemKey``
    or ``ComparableUnitKey``, and so inside a dict key. The record schema
    validates ranges and consistency but never these types, so a list in any
    of them surfaced as ``TypeError: unhashable type`` from inside
    de-duplication, or -- for ``quality_metric``, which ``sort_key`` folds
    away with ``or ""`` -- survived the loader and aborted ``compare()``
    instead. Either way the whole comparison died rather than one run being
    excluded with a reason.
    """
    write_run(tmp_path, "row-1")
    path = tmp_path / "runs" / "row-1" / "final_record.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[section][field_name] = ["a", "b"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert field_name in loaded.excluded[0].reason


def test_the_identity_probe_covers_what_compare_actually_hashes() -> None:
    """The net must probe the dataclasses, not their sort keys.

    ``compare()`` uses ``ComparableUnitKey`` and ``SystemKey`` themselves as
    dict keys. ``sort_key()`` folds every optional through ``x or default``,
    so a falsy unhashable value hashes fine as a sort key and still raises on
    the dict insertion. A probe over sort keys would pass and the comparison
    would die anyway.
    """
    unit = ComparableUnitKey(
        workload_id="w",
        workload_version="1",
        workload_prompt_hash="sha256:x",
        context_tier="2k",
        quality_metric=[],  # type: ignore[arg-type]
        max_output_tokens=512,
        temperature=0.0,
        top_p=1.0,
        request_shape=None,
    )
    # The folded sort key hides it ...
    hash(unit.sort_key())
    # ... but the object itself, which is what becomes a dict key, does not.
    with pytest.raises(TypeError):
        hash(unit)


def test_an_unhashable_identity_excludes_one_run_not_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the loader's probe, not just the dataclass property.

    Every field feeding these keys is validated upstream since the record
    and verification schemas were hardened, so the probe is unreachable
    through ordinary input; that is exactly what it is for -- the field that
    is not covered yet. ``request_shape`` is used here because ``sort_key``
    folds it, so a falsy unhashable value hashes as a sort key and only the
    dataclass probe catches it.
    """
    monkeypatch.setattr(
        evidence_module,
        "request_shape_for",
        lambda messages, *, workload_prompt_hash: [],
    )
    write_api_run(tmp_path, "api-1")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "cannot be used as a comparison key" in loaded.excluded[0].reason


def test_a_system_graded_by_another_evaluator_is_rejected(tmp_path: Path) -> None:
    """A quality score only means something with the metric it was earned on.

    The policy refuses ``min_quality_score`` without
    ``required_quality_metric`` for exactly this reason, so the engine must
    actually enforce the pairing rather than compare a score against a bar
    set for a different evaluator.
    """
    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    report = compare(
        results_dirs=(tmp_path,),
        policy=ComparePolicy.from_dict(
            {
                **COMPARE_POLICY,
                "constraints": {
                    "min_quality_score": 0.5,
                    "required_quality_metric": "human_expert_blind_review",
                },
            }
        ),
    )
    stratum = report.strata[0]
    assert stratum.ranked == ()
    assert stratum.rejected
    assert any(
        "human_expert_blind_review" in reason
        for entry in stratum.rejected
        for reason in entry.reasons
    )


def test_a_report_claiming_another_evaluators_bar_is_refused(
    tmp_path: Path,
) -> None:
    """Same pairing, re-applied when the report is loaded back.

    Editing only the policy block let a document assert a human-expert bar
    while its ranked system carried a machine-checked metric, and both the
    JSON and the rendered HTML presented that bar as having been cleared.
    """
    payload = _constrained_report(tmp_path)
    constraints = payload["policy"]["constraints"]  # type: ignore[index]
    constraints["required_quality_metric"] = "human_expert_blind_review"
    constraints["min_quality_score"] = 0.95
    with pytest.raises(CompareReportValidationError, match="graded by"):
        CompareReport.from_dict(payload)


@pytest.mark.parametrize("bad", [["x"], {"a": 1}, 7])
def test_a_non_string_collection_dir_is_excluded_not_raised(
    tmp_path: Path, bad: object
) -> None:
    """A recorded path reaches ``Path()``, not a hash.

    That is why the identity checks missed it. It stays invisible while the
    canonical ``collection/`` exists, so it only surfaced on a tree whose
    collection had been pruned -- and then as a ``TypeError`` from the middle
    of the loader that killed the whole comparison rather than one run.
    """
    import shutil

    write_run(tmp_path, "row-1")
    shutil.rmtree(collection_dir_for(tmp_path, "row-1"), ignore_errors=True)
    path = tmp_path / "runs" / "row-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["collection_dir"] = bad
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "collection_dir" in loaded.excluded[0].reason


@pytest.mark.parametrize("bad", [["y"], {"b": 2}, 9])
def test_a_non_string_final_record_path_is_excluded_not_raised(
    tmp_path: Path, bad: object
) -> None:
    """The sibling field, in the shared loader this change already touches."""
    write_run(tmp_path, "row-1")
    path = tmp_path / "runs" / "row-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_record_path"] = bad
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "final_record_path" in loaded.excluded[0].reason


def test_a_malformed_path_field_does_not_abort_the_whole_comparison(
    tmp_path: Path,
) -> None:
    """One bad run is excluded; the healthy runs beside it still compare."""
    import shutil

    good = tmp_path / "good"
    write_run(good, "row-1", total_ms=8000.0)
    write_api_run(good, "row-2", total_ms=1200.0)

    bad = tmp_path / "bad"
    write_run(bad, "row-1")
    shutil.rmtree(collection_dir_for(bad, "row-1"), ignore_errors=True)
    path = bad / "runs" / "row-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["collection_dir"] = ["not", "a", "path"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = compare(
        results_dirs=(good, bad), policy=ComparePolicy.from_dict(COMPARE_POLICY)
    )
    assert report.excluded_runs
    assert report.strata


# ---------------------------------------------------------------------------
# Exact-head review, tenth round.
# ---------------------------------------------------------------------------


def test_a_hosted_run_without_a_run_seal_is_excluded(tmp_path: Path) -> None:
    """``workloads run-api`` seals every run that produced a full set.

    So a hosted run with no ``run.json`` is not something the pipeline
    produces. Accepting it left the record and the verification -- the
    graded outcome and the summary this loader actually reads -- covered by
    no integrity marker at all.
    """
    from llmtracefx.optimizer.workloads.api_verify import RUN_MANIFEST_NAME

    write_api_run(tmp_path, "api-1")
    (tmp_path / "runs" / "api-1" / RUN_MANIFEST_NAME).unlink()

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    reason = loaded.excluded[0].reason
    assert "hosted API" in reason
    assert RUN_MANIFEST_NAME in reason


def test_a_local_run_without_a_run_seal_is_still_accepted(tmp_path: Path) -> None:
    """The local exception is exactly that and nothing wider.

    ``workloads run`` writes no seal, so requiring one would exclude every
    local run and make local-vs-hosted comparison impossible.
    """
    from llmtracefx.optimizer.workloads.api_verify import RUN_MANIFEST_NAME

    write_run(tmp_path, "local-1")
    assert not (tmp_path / "runs" / "local-1" / RUN_MANIFEST_NAME).exists()
    loaded = load_comparison_evidence((tmp_path,))
    assert len(loaded.runs) == 1


def test_a_hosted_run_with_a_broken_run_seal_is_excluded(tmp_path: Path) -> None:
    """Editing a sealed artifact without resealing must be caught."""
    write_api_run(tmp_path, "api-1")
    record = tmp_path / "runs" / "api-1" / "final_record.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["outcome"]["quality_score"] = 0.99
    record.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "seal" in loaded.excluded[0].reason


@pytest.mark.parametrize(
    "payload",
    [
        {"constraints": {"min_pass_rate": 0.5, "max_mean_latency_ms": 100}},
        {"constraints": {"min_pass_rate": 0.5, "minimum_pass_rate": 0.9}},
        {"objectives": "min_mean_total_latency_ms"},
        {"contraints": {"min_pass_rate": 0.5}},
    ],
)
def test_an_unknown_policy_key_is_refused(payload: dict[str, object]) -> None:
    """A misspelled constraint silently did nothing.

    The comparison then reported clearing a bar that was never applied,
    which reads exactly like the bar having been enforced. Every one of
    these is a plausible typo for a real field.
    """
    with pytest.raises(ComparePolicyError, match="unknown field"):
        ComparePolicy.from_dict({**COMPARE_POLICY, **payload})


def test_a_valid_policy_is_still_accepted() -> None:
    """The other direction: every documented field must still load."""
    policy = ComparePolicy.from_dict(
        {
            "schema_version": "1",
            "name": "n",
            "description": "d",
            "objective": "min_mean_total_latency_ms",
            "constraints": {
                "required_statuses": ["completed"],
                "allowed_provenances": ["measured_wall_clock"],
                "min_pass_rate": 0.5,
                "min_quality_score": 0.5,
                "required_quality_metric": "m",
                "max_mean_total_latency_ms": 1000.0,
                "max_cost_per_correct_case": 1.0,
                "min_measured_repetitions": 1,
                "max_coefficient_of_variation": 0.5,
            },
        }
    )
    assert policy.constraints.min_pass_rate == 0.5


def _constrained_report(tmp_path: Path) -> dict[str, object]:
    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    report = compare(
        results_dirs=(tmp_path,),
        policy=ComparePolicy.from_dict(
            {
                **COMPARE_POLICY,
                "objective": "min_mean_total_latency_ms",
                "constraints": {"min_pass_rate": 0.9},
            }
        ),
    )
    payload = json.loads(json.dumps(report.to_dict()))
    assert isinstance(payload, dict)
    return payload  # type: ignore[no-any-return]


def test_a_ranked_system_below_its_own_policy_bar_is_refused(
    tmp_path: Path,
) -> None:
    """A report carries the bar and the systems said to have cleared it.

    Those are two claims about one fact, and a file can assert both and have
    them disagree. ``compare()`` rejects a system that misses the bar, so a
    loaded report ranking one did not come from an honest run.
    """
    payload = _constrained_report(tmp_path)
    stratum = payload["strata"][0]
    stratum["ranked"][0]["pass_rate"] = 0.1
    if stratum.get("recommended"):
        stratum["recommended"]["pass_rate"] = 0.1
    with pytest.raises(CompareReportValidationError, match="does not satisfy"):
        CompareReport.from_dict(payload)


def test_a_ranked_system_with_too_few_repetitions_is_refused(
    tmp_path: Path,
) -> None:
    """The repetition floor is a constraint like any other."""
    payload = _constrained_report(tmp_path)
    payload["policy"]["constraints"]["min_measured_repetitions"] = 5
    with pytest.raises(CompareReportValidationError, match="requires at least"):
        CompareReport.from_dict(payload)


def test_an_honest_constrained_report_still_round_trips(tmp_path: Path) -> None:
    """The other direction: real reports must keep loading."""
    payload = _constrained_report(tmp_path)
    assert CompareReport.from_dict(payload).strata


def test_a_symlinked_verification_is_not_followed(tmp_path: Path) -> None:
    """The loader reads its two artifacts bounded and without following.

    ``RowVerification.read_json`` and ``ExperimentRecord.read_json`` both use
    an unbounded ``read_text`` that follows symlinks, so a link at either
    path was read from wherever it pointed -- the same escape the
    containment rules exist to close.
    """
    inside, outside = tmp_path / "inside", tmp_path / "outside"
    write_run(inside, "row-1", total_ms=1000.0)
    write_run(outside, "row-1", total_ms=9999.0)
    target = inside / "runs" / "row-1" / "verification.json"
    target.unlink()
    target.symlink_to(outside / "runs" / "row-1" / "verification.json")

    loaded = load_comparison_evidence((inside,))
    assert loaded.runs == ()


def test_an_oversized_verification_is_refused(tmp_path: Path) -> None:
    """Bounded before parse, like every other artifact this layer reads."""
    from llmtracefx.optimizer._artifact_io import MAX_METADATA_ARTIFACT_BYTES

    write_run(tmp_path, "row-1")
    path = tmp_path / "runs" / "row-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["padding"] = "x" * (MAX_METADATA_ARTIFACT_BYTES + 1024)
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()


@pytest.mark.parametrize(
    ("section", "field_name"),
    [
        (None, "workload_id"),
        (None, "workload_version"),
        (None, "context_tier"),
        (None, "decode_mode"),
        ("model", "model_id"),
        ("runtime", "name"),
    ],
)
def test_a_null_required_identity_field_is_excluded_not_a_sort_crash(
    tmp_path: Path, section: str | None, field_name: str
) -> None:
    """``None`` is hashable, so the identity probe cannot catch it.

    These six are emitted raw by ``sort_key()`` rather than folded through
    ``or ""``, because the schema requires them. A ``None`` therefore passed
    every check and then raised ``TypeError: '<' not supported`` from inside
    ``sorted()``, aborting the whole comparison -- or, with a single system,
    produced a report the tool's own loader rejects.
    """
    write_run(tmp_path, "row-1")
    name = "verification.json" if section is None else "final_record.json"
    path = tmp_path / "runs" / "row-1" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload if section is None else payload[section]
    target[field_name] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert field_name in loaded.excluded[0].reason


def test_an_unwritable_compare_output_is_a_stated_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The comparison already succeeded; losing it to a traceback is worst."""
    from llmtracefx.optimizer.cli import main

    write_run(tmp_path, "a", total_ms=8000.0)
    write_api_run(tmp_path, "b", total_ms=1200.0)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    directory = tmp_path / "not-a-file"
    directory.mkdir()

    try:
        code = main(
            [
                "compare",
                "--results",
                str(tmp_path),
                "--policy",
                str(policy),
                "--output",
                str(directory),
            ]
        )
    except SystemExit as exc:
        code = int(exc.code or 0)
    assert code == 1
    assert "Could not write compare report JSON" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Exact-head review, eleventh round: guarantees that were live but unpinned,
# so a mutation disabling them left the whole suite green.
# ---------------------------------------------------------------------------


def test_a_huge_but_finite_timing_never_produces_an_infinite_p50(
    tmp_path: Path,
) -> None:
    """The median of two finite floats can still overflow.

    ``(a + b) / 2`` is computed in float, so filtering the inputs for
    finiteness does not make the median finite. An infinite p50 printed as
    "p50 inf ms" and made ``to_json`` raise, because the report is written
    with ``allow_nan=False`` -- an unhandled traceback out of the CLI on
    ordinary artifact input.
    """
    for name in ("a", "b"):
        write_run(tmp_path / name, "row-1", total_ms=1.7e308)
    report = compare(
        results_dirs=(tmp_path / "a", tmp_path / "b"),
        policy=ComparePolicy.from_dict(COMPARE_POLICY),
    )
    # Serialising must not raise: this is the failure the CLI surfaced.
    report.to_json()
    for stratum in report.strata:
        for system in stratum.ranked:
            assert system.p50_total_latency_ms is None or math.isfinite(
                system.p50_total_latency_ms
            )


def test_a_coefficient_of_variation_ceiling_actually_rejects(
    tmp_path: Path,
) -> None:
    """A ceiling nobody enforces still renders as being in force."""
    write_run(tmp_path / "a", "row-1", total_ms=1000.0)
    write_run(tmp_path / "b", "row-1", total_ms=9000.0)
    write_api_run(tmp_path / "c", "row-1", total_ms=1200.0)
    report = compare(
        results_dirs=(tmp_path / "a", tmp_path / "b", tmp_path / "c"),
        policy=ComparePolicy.from_dict(
            {**COMPARE_POLICY, "constraints": {"max_coefficient_of_variation": 0.05}}
        ),
    )
    stratum = report.strata[0]
    assert any(
        "coefficient of variation" in reason
        for entry in stratum.rejected
        for reason in entry.reasons
    )


def test_a_mean_quality_score_below_the_bar_is_rejected(tmp_path: Path) -> None:
    """The mean is what is compared, so the mean is what must be checked."""
    write_run(tmp_path / "a", "row-1", quality_score=0.2)
    write_run(tmp_path / "b", "row-1", quality_score=0.9)
    write_api_run(tmp_path / "c", "row-1")
    report = compare(
        results_dirs=(tmp_path / "a", tmp_path / "b", tmp_path / "c"),
        policy=ComparePolicy.from_dict(
            {
                **COMPARE_POLICY,
                "constraints": {
                    "min_quality_score": 0.9,
                    "required_quality_metric": "structured_json_exact_field_match",
                },
            }
        ),
    )
    stratum = report.strata[0]
    assert any(
        "mean quality_score" in reason
        for entry in stratum.rejected
        for reason in entry.reasons
    )


def test_a_non_finite_quality_score_is_rejected(tmp_path: Path) -> None:
    """NaN is not a grade. Unchecked it entered the ranking."""
    write_run(tmp_path, "row-1")
    write_api_run(tmp_path, "api-1")
    path = tmp_path / "runs" / "row-1" / "final_record.json"
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    payload["outcome"]["quality_score"] = float("nan")
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = compare(
        results_dirs=(tmp_path,), policy=ComparePolicy.from_dict(COMPARE_POLICY)
    )
    stratum = report.strata[0]
    assert (
        any(
            "non-finite" in reason
            for entry in stratum.rejected
            for reason in entry.reasons
        )
        or not stratum.ranked
    )


def test_time_to_first_token_never_mixes_two_bases(tmp_path: Path) -> None:
    """A local prefill and a client-observed stream are different quantities.

    Averaging them would publish a number that is a measurement of nothing.
    Provider identity keeps the two apart in ordinary input, so the guard is
    exercised directly on the collector rather than through the loader.
    """
    from llmtracefx.optimizer.compare.compare import _collect_ttft

    write_run(tmp_path / "local", "row-1")
    write_api_run(tmp_path / "hosted", "row-1")
    local = load_comparison_evidence((tmp_path / "local",)).runs
    hosted = load_comparison_evidence((tmp_path / "hosted",)).runs
    assert local and hosted

    # Each basis alone reports a value.
    local_mean, local_basis, _ = _collect_ttft(local)
    hosted_mean, hosted_basis, _ = _collect_ttft(hosted)
    assert local_mean is not None and local_basis is not None
    assert hosted_mean is not None and hosted_basis is not None
    assert local_basis != hosted_basis

    # Together, neither is reported and the reason says why.
    mixed_mean, mixed_basis, mixed_missing = _collect_ttft([*local, *hosted])
    assert mixed_mean is None
    assert mixed_basis is None
    assert mixed_missing


def test_a_hosted_run_with_no_recorded_collection_dir_is_excluded(
    tmp_path: Path,
) -> None:
    """Otherwise it is published as hosted with no endpoint, usage or TTFT.

    It also loses its decode settings, which silently moves it into a
    different comparable unit.
    """
    write_api_run(tmp_path, "api-1")
    path = tmp_path / "runs" / "api-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["collection_dir"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_run(tmp_path, "api-1")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "hosted API" in loaded.excluded[0].reason


def test_an_absolute_collection_dir_cannot_leave_the_results_tree(
    tmp_path: Path,
) -> None:
    """The compare-side twin of the record-path containment test.

    Two trees from one matrix share every id, so the identity checks cannot
    catch this: without containment, tree A's run is attributed tree B's
    ``api_evidence.json``.
    """
    import shutil

    inside, outside = tmp_path / "inside", tmp_path / "outside"
    write_api_run(inside, "api-1", model_id="glm-5.3")
    write_api_run(outside, "api-1", model_id="glm-5.3-flash")
    shutil.rmtree(collection_dir_for(inside, "api-1"))

    path = inside / "runs" / "api-1" / "verification.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["collection_dir"] = str(collection_dir_for(outside, "api-1"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_run(inside, "api-1")

    loaded = load_comparison_evidence((inside,))
    assert loaded.runs == ()
    assert "does not resolve to a directory inside" in loaded.excluded[0].reason


@pytest.mark.parametrize(
    "field_name", ["pass_rate", "mean_quality_score", "quality_metric"]
)
def test_nulling_a_constrained_metric_does_not_evade_the_recheck(
    tmp_path: Path, field_name: str
) -> None:
    """``null`` on a ranked system is not absent evidence.

    ``compare()`` rejects a system whose value is missing for any of these
    constraints, so a loaded report carrying ``null`` did not come from an
    honest run. Skipping it made nulling the field the simplest way to evade
    the bar entirely.
    """
    payload = _constrained_report(tmp_path)
    constraints = payload["policy"]["constraints"]  # type: ignore[index]
    constraints["required_quality_metric"] = "structured_json_exact_field_match"
    constraints["min_quality_score"] = 0.5
    stratum = payload["strata"][0]  # type: ignore[index]
    stratum["ranked"][0][field_name] = None
    if stratum.get("recommended"):
        stratum["recommended"][field_name] = None
    with pytest.raises(CompareReportValidationError):
        CompareReport.from_dict(payload)
