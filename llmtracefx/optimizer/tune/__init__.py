"""Offline, evidence-constrained inference-configuration tuning.

This package turns the verified evidence produced by
``llmtracefx.optimizer.workloads.verify`` (``verification.json`` +
``final_record.json`` under a results directory) into constraint-filtered,
ranked configuration recommendations.

It never loads a model, requires a GPU, executes a benchmark, or downloads
anything. It only reads already-collected, already-verified artifacts from
disk, re-validates them, and reports:

* which comparable group each candidate configuration belongs to,
* which candidates satisfy every configured constraint and which do not
  (with the precise reason for every rejection),
* the single recommended candidate per comparable group under exactly one
  ranking objective, and
* an optional autoregressive-baseline comparison using the existing
  speculative-decoding doctor rule.

See ``policy.py`` for the tuning policy schema, ``tuner.py`` for the
grouping/constraint/ranking engine, ``report.py`` for the output schema,
and ``explain.py`` for the human-readable terminal summary.
"""

from __future__ import annotations

TUNE_SCHEMA_VERSION = "1"
