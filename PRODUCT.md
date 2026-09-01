# Product

<!-- impeccable:product-schema 1 -->

> Provenance note. This record was written from the repository (README, `pyproject.toml`,
> `llmtracefx/optimizer/**`, tests) plus an explicit written brief, in a session where the
> interview round was answered in advance by that brief. Facts below are drawn from code and
> documentation. Items marked **(inferred)** were not stated outright and should be corrected
> by a maintainer if wrong.

## Platform

web

## Users

Performance engineers and ML practitioners who need to know whether an inference configuration
change actually helped without separating performance from correctness. The concrete situations
the repository supports today:

- Apple silicon through MLX and MLX-LM, plus evidence-honest Metal interval collection through
  Apple Instruments.
- OpenAI-compatible streaming providers, measured from the client boundary.
- NVIDIA and other llama.cpp environments through captured stdout and stderr imported into the
  canonical schema.

The job is not "watch a dashboard". It is: run a workload matrix, decide which configuration to
ship under a stated budget, and be able to defend that decision later with the artifacts that
produced it. **(inferred)** The reader of a tune report is often not the person who ran it, which
is why the report must travel as one file and stay legible offline.

## Product Purpose

LLMTraceFX measures inference, verifies that a configuration still produces correct output,
compares local and hosted systems on identical work, tunes configuration choices against explicit
constraints, and emits reproducible offline evidence of the result. Success is a recommendation a
reader can audit: which candidates were accepted, which were rejected and for exactly which
violated constraint, which runs were excluded outright, and which source artifacts back every
number.

## Positioning

Provenance is a first-class field, not a footnote. Every measurement carries a `MetricProvenance`
(`measured_native`, `measured_wall_clock`, `provider_reported`, `derived`, `estimated`), so a
reader can tell a runtime counter, a client timer, a provider report, and an estimate apart. The
system refuses rather than guesses: the speculative-decoding doctor returns `inconclusive` when
runs are not comparable or the delta is inside run-to-run noise; `native-mtp collect` writes an
explicit unsupported record instead of relabeling generic draft-model speculation as native MTP;
`tune` and `compare` optimize exactly one declared objective and never blend several into a
composite score.

## Operating Context

- The optimizer is driven from a terminal. Local collectors require model directories that
  already exist on disk, `workloads generate-matrix` plans without loading a model, and hosted API
  collection is an explicit optional path.
- The local pipeline is `manifest` -> `workloads generate-matrix` -> `workloads run` -> `tune` ->
  `tune-report`, with `optimize` composing the executing phases and writing `optimize_summary.json`.
  `compare` consumes already-verified local and API result directories and `compare-report` renders
  that separate cross-system decision.
- Artifacts are files on disk: `record.json`, `final_record.json`, `verification.json`,
  `capability_report.json`, materialized prompts, and the tune and comparison report JSON and HTML.
- Report HTML is shared by attaching the file itself to an issue, a chat message, or a shared
  drive. It is opened from disk, frequently without a network connection.
- Long-running collection can happen on a local machine or metered infrastructure, so re-running
  may be expensive and resumability matters.

## Capabilities and Constraints

- **Offline and self-contained.** Generated tune and comparison reports are single HTML files with
  inline CSS, no JavaScript, no CDN, and no external runtime asset of any kind.
- **Deterministic.** Rendering the same `TuneReport` twice is byte-identical. The only clock in the
  document is the report's own `generated_at`, never a timestamp taken at render time.
- **Escaped.** Every string that originates in report JSON passes through `html.escape` before it
  reaches the document.
- **Privacy-safe by default.** Local artifact paths are redacted to stable
  `runs/<run_id>/<file>` labels; `--include-paths` opts into full paths. The environment manifest
  never collects secrets, usernames, hostnames, or full environment dumps.
- **Report sections that must never be dropped:** policy and constraints, the run summary, group
  identity, the recommendation and its rationale, the ranked accepted candidates, rejected
  candidates with every violated constraint, inconclusive states with their recorded reason, the
  speculative-versus-baseline comparison, excluded runs with reasons, and the source run IDs and
  artifact paths behind each candidate.
- **Vocabulary that carries meaning and must be preserved:** group, candidate, objective,
  constraint, violation, accepted, rejected, excluded, recommended, inconclusive, evidence count,
  provenance, coefficient of variation, pass rate, context tier, decode mode, speculative.
- **Runtime dependency floor.** The optimizer package is standard library only. Report rendering
  must not add a runtime dependency.
- Python 3.10+, Apache-2.0, formatted with Black and isort, linted with Ruff, typed with mypy.

## Brand Commitments

- Name is `LLMTraceFX`; the CLI entry points are `llmtracefx` and `llmtracefx-optimizer`.
- Precise, calm, technical, premium, and original. Explicitly ruled out by the brief: generic
  purple AI gradients, heavy glassmorphism, neon cyberpunk, pill overload, oversized empty hero
  sections, and decoration that competes with evidence.
- The identity mark must express trace, evidence, and optimization. A brain, sparkles, a robot, or
  a generic audio waveform are all rejected.
- Prose in this project's own surfaces avoids em dashes.

## Evidence on Hand

- `examples/optimizer/tune-report-example.json`: a complete synthetic report that exercises every
  section of the viewer. It is labeled SYNTHETIC in its own content and is not benchmark data.
- `examples/optimizer/compare-report-example.json`: a synthetic cross-system report with explicit
  non-benchmark and illustrative-pricing labels.
- `tests/optimizer/fixtures/llama_cpp/*.log`: synthetic hand-written logs with a `PROVENANCE.md`
  that says so.
- `tests/optimizer/_tune_fixtures.py`: builds `workloads run`-shaped artifact trees for tests.
- **No real Qwen3.8-27B benchmark results exist.** No performance number may be presented as
  measured. There are no customers, no pricing, and no third-party endorsements to cite.
- **No real Qwen3-8B M5 Pro self-conversion or benchmark results exist.** The planned, preparatory
  self-conversion control (`llmtracefx-m5-control`, `llmtracefx/optimizer/lab/qwen3_8b/`) ships its
  conversion/benchmark manifests, subprocess-isolated runner, and offline tests; the packaged
  benchmark manifest is a template only (deliberately missing the output file hashes) until a real
  conversion produces a receipt to bind it from -- no fabricated hash may ever be committed to fill
  that gap. The one real attempt so far was refused by the pre-conversion safety gate before any
  download (see the committed refusal artifact in `examples/optimizer/qwen3-8b-m5-control/`);
  execution remains blocked pending a clean reboot and a passing safe preflight.
- The older public Modal analyzer endpoint and video-led synthetic dashboard walkthrough have been
  removed from the current README. Legacy analyzer code remains for compatibility but is not the
  primary product path.

## Product Principles

1. Evidence outranks conclusions. Every claim in a report must be traceable to an artifact the
   reader can open.
2. Refuse before you guess. An explicit `inconclusive` or `unsupported` result is a successful
   outcome, not a failure to be smoothed over.
3. One objective at a time, under stated constraints. No composite scores, no hidden weighting.
4. The artifact must outlive the machine that made it: offline, deterministic, and safe to share.
5. Say what was excluded. Rejected and excluded material is part of the finding, never noise to
   be hidden.

## Accessibility & Inclusion

The report is read in varied conditions: a laptop screen, a phone in a lab, and printed or saved
to PDF for a review. Status must never be encoded by color alone; every state also carries a word
or a glyph. The report is a light-mode document and must hold contrast when printed in grayscale.
