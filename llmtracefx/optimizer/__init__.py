"""Foundation primitives for the LLMTraceFX inference optimizer.

This package is the first step in evolving LLMTraceFX from a trace
*analyzer* into a workload-aware inference *optimizer*. It establishes
reliable, tested building blocks — a canonical evidence schema, an
environment manifest collector, a safe experiment runner, a llama.cpp
output parser, and a first "doctor" diagnostic rule — that later
collectors (native Metal/CUDA counters) and tuning logic can build on.

Nothing in this package downloads models, requires a GPU, or performs
brute-force tuning.
"""
