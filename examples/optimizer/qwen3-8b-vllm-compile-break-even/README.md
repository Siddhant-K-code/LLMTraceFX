# Qwen3 8B vLLM compilation break-even on CloudRift

This bundle compares eager execution with vLLM compilation and CUDA graphs on
one fixed CloudRift RTX 4090. Both cells used the same immutable runtime, exact
model revision, token arrays, request order, and bounded generation settings.

Compilation did not repay its initialization cost within the 12 observed
requests. Repeating the exact request sequence without any other change yields
a modeled crossing at request 113. That crossing is an extrapolation, not an
observed request.

All 24 responses passed their deterministic workload evaluators. The measured
VM accounting window through scheduled OS shutdown is a $0.393033 list-rate
lower bound. Provider-reported spend and final spend through console termination
are unavailable. The experiment containers, GPU processes, model data, runtime
images, result caches, and temporary public key were removed before shutdown.
CloudRift console termination remains unconfirmed and billing may continue.

Run `python evidence_bundle.py verify` from this directory to verify the closed
file set, checksums, privacy rules, model and runtime pins, request contract,
correctness, break-even arithmetic, cost scope, and teardown status.
