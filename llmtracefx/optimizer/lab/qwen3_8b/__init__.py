"""Qwen3-8B M5 Pro self-conversion control -- planned, preparatory.

**No conversion or benchmark has run yet.** This subpackage ships the
offline framework (manifests, self-conversion runner, bound-manifest
binder, subprocess-isolated benchmark runner) for a future self-converted
Qwen3-8B positive control; it is intended to eventually serve that role,
but nothing here should be read as a claim that a conversion, benchmark,
or comparison has actually completed.

This subpackage is deliberately isolated from the packaged Qwen3.8-27B
lab (``llmtracefx.optimizer.lab.manifest``/``core``/``frontier``/
``autopsy``): every packaged manifest, cache path, workspace path, and
CLI entrypoint here lives in its own ``qwen3-8b`` namespace so nothing in
this subpackage can read, write, or overwrite the 27B artifacts.

Unlike the 27B lab, which pins an already-quantized ``mlx-community``
checkpoint byte-for-byte, this control is designed to self-convert the
official, public, ungated ``Qwen/Qwen3-8B`` checkpoint with the
repository's own pinned ``mlx-lm`` using explicit, recorded quantization
parameters. It is designed to never claim byte-equivalence with any
third-party conversion; see ``conversion_manifest.py`` for the exact
provenance recorded once a real conversion exists.
"""

from __future__ import annotations
