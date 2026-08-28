"""Deterministic, versioned workload catalog and matrix generation.

This package defines a small, redistributable set of code-completion,
constrained-structured-JSON, and prose/reasoning prompts (see
``catalog.py``), how they are materialized to target context sizes
without silently fabricating padding (see ``materialize.py``), how the
matrix of (workload, context tier, decode mode) combinations is
generated deterministically without executing or downloading anything
(see ``matrix.py``), and deterministic evaluators suitable for these
workloads (see ``evaluators.py``).
"""

from __future__ import annotations

WORKLOAD_SCHEMA_VERSION = "1"
