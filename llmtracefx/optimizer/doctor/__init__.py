"""Evidence-based analysis rules ("doctor") for the inference optimizer.

Each rule reads canonical ``ExperimentRecord`` evidence and returns a
verdict; rules never guess when the evidence is insufficient, mismatched,
or noisy — they report "inconclusive" instead.
"""
