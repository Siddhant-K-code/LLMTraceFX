# Qwen3 8B vLLM compilation break-even on CloudRift

This bundle compares eager execution with vLLM compilation and CUDA graphs on
one fixed CloudRift RTX 4090. Both cells used the same immutable runtime, exact
model revision, token arrays, request order, and bounded generation settings.

Compilation did not repay its initialization cost within the 12 observed
requests. Repeating the exact request sequence without any other change yields
a modeled crossing at request 113. That crossing is an extrapolation, not an
observed request.

Twenty-two of 24 responses passed their deterministic workload evaluators.
Eager execution returned an incorrect `3.5` answer for both 16K prose requests;
compiled execution returned correct answers for all 12 requests. Eight of 12
paired outputs had identical token IDs. The boot-to-console-termination window
implies $0.484358 at the observed list rate. This remains a lower bound because
the provisioning-to-boot interval is unavailable. Provider-reported spend is
unavailable. The experiment containers, GPU processes, model data, runtime
images, result caches, and temporary public key were removed before shutdown.
The user confirmed CloudRift console termination separately.

Run `python evidence_bundle.py verify` from this directory to verify the closed
file set, checksums, privacy rules, model and runtime pins, request contract,
correctness, break-even arithmetic, cost scope, and teardown status.
