"""Offline, scientifically honest comparison of systems that ran the same task.

The tuner answers "which configuration of this model on this machine is
fastest". This package answers the different question "how do a local model,
a frontier API and a cheap fast API actually compare on work they have both
already done", using only evidence that was already collected.

Nothing in here loads a model, calls an API, deploys anything, or executes a
benchmark. It reads ``workloads run`` results directories and produces a
versioned JSON report plus a self-contained branded HTML rendering of it.

The invariants are the point:

* Two runs are only ever compared when they share a ``ComparableUnitKey``:
  identical workload, version, prompt hash, context tier, evaluator, output
  cap and sampling. Different identities become different strata.
* Systems keep their labels (model/revision, provider, runtime/backend,
  accelerator, quantization, reasoning effort, decode mode) and unlike
  systems are never averaged together.
* A measurement that was not recorded stays unavailable. It never becomes
  zero, and it never quietly disqualifies the axis it belongs to.
* Money is derived from provider-reported usage and an explicit versioned
  pricing manifest, is labeled estimated everywhere, and is refused outright
  when the manifest is ambiguous or the usage needed for it is missing.
* Ranking is constraints-first and single-objective. Ties and noise produce
  an explicit inconclusive verdict rather than a coin flip, and the frontier
  is published so no single universal winner is ever implied.
"""

from __future__ import annotations

from .compare import FRONTIER_AXES, compare
from .cost import (
    MONETARY_BASIS,
    CostBreakdown,
    TokenUsage,
    correct_cases_per_currency_unit,
    cost_per_case,
    estimate_run_cost,
)
from .evidence import (
    ApiEvidence,
    ApiEvidenceError,
    CompareEvidenceError,
    DecodeSettings,
    SystemRun,
    load_comparison_evidence,
)
from .identity import ComparableUnitKey, CompareIdentityError, SystemKey
from .policy import (
    COMPARE_POLICY_SCHEMA_VERSION,
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
    ComparePolicyError,
)
from .pricing import (
    PRICING_MANIFEST_SCHEMA_VERSION,
    PricingEntry,
    PricingError,
    PricingManifest,
)
from .report import (
    COMPARE_REPORT_SCHEMA_VERSION,
    CompareReport,
    CompareReportValidationError,
    CostSummary,
    FrontierEntry,
    ParetoAxis,
    PricingProvenance,
    RejectedSystemReport,
    StratumOutcome,
    StratumReport,
    SystemReport,
    TtftBasis,
    UsageTotals,
)
from .report_html import render_compare_report_html

__all__ = [
    "COMPARE_POLICY_SCHEMA_VERSION",
    "COMPARE_REPORT_SCHEMA_VERSION",
    "FRONTIER_AXES",
    "MONETARY_BASIS",
    "PRICING_MANIFEST_SCHEMA_VERSION",
    "ApiEvidence",
    "ApiEvidenceError",
    "ComparableUnitKey",
    "CompareConstraints",
    "CompareEvidenceError",
    "CompareIdentityError",
    "CompareObjective",
    "ComparePolicy",
    "ComparePolicyError",
    "CompareReport",
    "CompareReportValidationError",
    "CostBreakdown",
    "CostSummary",
    "DecodeSettings",
    "FrontierEntry",
    "ParetoAxis",
    "PricingEntry",
    "PricingError",
    "PricingManifest",
    "PricingProvenance",
    "RejectedSystemReport",
    "StratumOutcome",
    "StratumReport",
    "SystemKey",
    "SystemReport",
    "SystemRun",
    "TokenUsage",
    "TtftBasis",
    "UsageTotals",
    "compare",
    "correct_cases_per_currency_unit",
    "cost_per_case",
    "estimate_run_cost",
    "load_comparison_evidence",
    "render_compare_report_html",
]
