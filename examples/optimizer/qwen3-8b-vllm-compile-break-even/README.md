# Qwen3 8B vLLM compilation break-even on CloudRift

This bundle compares eager execution with vLLM compilation and CUDA graphs on
one fixed CloudRift RTX 4090. Staging hash-verified the exact model revision.
Both cells reported the same runtime package versions and matching input-token
arrays, request order, and bounded generation settings. Per-cell model and image
binding limitations are stated below. The runtime was vLLM 0.28.0, Python 3.12,
PyTorch 2.13.0+cu130, CUDA 13.0, and Transformers 5.15.1.

The compiled lifecycle did not cross the eager lifecycle time within the 12
observed requests. Frozen exact-observed-outcome repeated-sequence time arithmetic
crosses at request 113. It repeats the observed eager and compiled outcomes unchanged,
including different output token arrays, lengths, and correctness at ordinals 7,
8, 11, and 12. Eager produced 426 output tokens and compiled produced 444. The
crossing is not observed, output-controlled, or a causal compilation speedup.
There is no replicated or counterbalanced lifecycle and no fixed-output-token run.
It does not establish general break-even.

Twenty-two of 24 responses passed their deterministic workload evaluators.
Eager execution returned an incorrect `3.5` answer for both 16K prose requests;
compiled execution returned correct answers for all 12 requests. Eight of 12
paired outputs had identical token IDs. The boot-to-console-termination window
implies a $0.484358 list-rate lower bound because the provisioning-to-boot
interval is unavailable. Provider-reported spend is unavailable, so the
$4.515642 remainder under the $5 cap is an upper bound, not an exact balance.
The experiment containers, GPU processes, model data, runtime images, result
caches, and temporary public key were removed before the scheduled OS shutdown.
OS shutdown itself was not observed. The user externally confirmed CloudRift
console termination. Prior Modal attempts produced no benchmark and are excluded.

The collection runner verified the staging and prompt receipts independently.
It did not rehash the live 16 GB model or cross-check the two receipt hashes
before each measured cell.
The public verifier binds the committed inventory, prompt arrays, collection
source, metadata, and hashes. It does not independently verify the underlying
private GPU identity or provider-console event, and these collection limitations
cannot be retroactively removed.
No independent host orchestration receipt was retained for the fresh-container,
cache-drop, timeout-wrapper, bind-mount, network, or Docker image-inspection
controls. The records prove ordered non-overlapping processes and no warmup
requests, but those additional host controls remain unverified.

From a clean checkout, run `uv run python -I evidence_bundle.py verify` in this
directory. The wrapper resolves the repository root before importing the verifier.
