"""Deriving monetary cost from provider-reported usage and a pricing manifest.

Two rules shape everything in this module.

**Cost is derived, never measured.** Token counts come from the provider's
own accounting (``MetricProvenance.PROVIDER_REPORTED``); rates come from a
user-supplied manifest. Neither is something this client observed, so every
monetary value produced here is labeled ``estimated`` and is kept strictly
apart from the wall-clock timings the client did measure.

**A missing input makes the answer unavailable, never zero.** If usage was
not reported, or the manifest has no rate for the tokens that were used, the
cost is ``None`` with a recorded reason. A free-looking number is worse than
no number, because it silently wins every cost ranking.

The two composition subtleties an OpenAI-compatible ``usage`` block hides are
both handled explicitly rather than assumed away:

* ``prompt_tokens`` already *includes* ``prompt_tokens_details.cached_tokens``.
  Billing both at full rate would double-count the cache.
* ``completion_tokens`` already *includes*
  ``completion_tokens_details.reasoning_tokens``. Reasoning is normally billed
  at the output rate, which is what happens when the manifest declares no
  separate reasoning rate. Crucially, when the provider does not report
  reasoning tokens at all this module does **not** invent them: it bills the
  completion total at the output rate and records that reasoning usage was
  unavailable, so nobody reads the result as "reasoning was free".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .pricing import PricingEntry, PricingError

#: Provider usage is billed per million tokens.
TOKENS_PER_RATE_UNIT = 1_000_000

#: The largest token count this module will multiply by a rate. Above it a
#: Python int either raises ``OverflowError`` on conversion to a float or
#: converts with silent precision loss, so the product is not arithmetic.
#: ``compare.evidence`` already refuses a count this large when it loads a
#: sidecar; the bound is repeated here because this function is public and a
#: caller can hand it any ``TokenUsage`` at all.
MAX_BILLABLE_TOKEN_COUNT = 2**53

#: Recorded on every monetary value this module produces, so a reader never
#: has to guess whether a currency figure was observed or computed.
MONETARY_BASIS = "estimated_from_provider_reported_usage_and_manifest_rates"


@dataclass(frozen=True)
class TokenUsage:
    """Provider-reported token accounting for one run, as reported.

    Every field is exactly what the provider said, with no derivation. A
    field the provider omitted stays ``None``.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        """Mirror the loader's rule, because this constructor is public.

        The sidecar parser already refuses a negative count, but a caller
        constructing this directly bypasses that parser entirely, and a
        negative count subtracts from a bill.

        Representability is deliberately *not* enforced here. A count past
        ``MAX_BILLABLE_TOKEN_COUNT`` is refused where it is parsed, and
        ``estimate_run_cost`` treats one that reached it by another route as
        an unpriceable run rather than raising. Refusing it in the
        constructor as well would remove the only way to exercise that
        defence, and the guarantee worth keeping is that the cost helper
        never raises, not that the value cannot be built.
        """
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise PricingError(
                    f"usage.{name} must be an integer or null, got {value!r}"
                )
            if value < 0:
                raise PricingError(f"usage.{name} must be >= 0, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class CostBreakdown:
    """One run's estimated cost, or the reason it could not be estimated."""

    amount: float | None
    currency: str
    entry_id: str
    entry_sha256: str
    rates_are_illustrative: bool
    billed_input_tokens: int | None = None
    billed_cached_tokens: int | None = None
    billed_output_tokens: int | None = None
    billed_reasoning_tokens: int | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    """Why ``amount`` is ``None``, or caveats that apply to a computed one."""

    @property
    def available(self) -> bool:
        return self.amount is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "currency": self.currency,
            "monetary_basis": MONETARY_BASIS,
            "estimated": True,
            "pricing_entry_id": self.entry_id,
            "pricing_entry_sha256": self.entry_sha256,
            "rates_are_illustrative": self.rates_are_illustrative,
            "billed_input_tokens": self.billed_input_tokens,
            "billed_cached_tokens": self.billed_cached_tokens,
            "billed_output_tokens": self.billed_output_tokens,
            "billed_reasoning_tokens": self.billed_reasoning_tokens,
            "reasons": list(self.reasons),
        }


def _unavailable(
    entry: PricingEntry, reasons: list[str], **billed: int | None
) -> CostBreakdown:
    return CostBreakdown(
        amount=None,
        currency=entry.currency,
        entry_id=entry.entry_id,
        entry_sha256=entry.content_sha256,
        rates_are_illustrative=entry.rates_are_illustrative,
        reasons=tuple(reasons),
        **billed,
    )


def estimate_run_cost(usage: TokenUsage, entry: PricingEntry) -> CostBreakdown:
    """Estimate one run's cost, refusing to guess at anything missing.

    Returns a breakdown whose ``amount`` is ``None`` (with reasons) whenever
    the usage or the manifest cannot support an honest total.
    """
    reasons: list[str] = []

    oversized = [
        name
        for name, count in (
            ("prompt_tokens", usage.prompt_tokens),
            ("completion_tokens", usage.completion_tokens),
            ("cached_prompt_tokens", usage.cached_prompt_tokens),
            ("reasoning_tokens", usage.reasoning_tokens),
        )
        if count is not None and count > MAX_BILLABLE_TOKEN_COUNT
    ]
    if oversized:
        # Multiplying one of these by a rate raises ``OverflowError``, which
        # is an ``ArithmeticError`` and so is caught by nothing downstream.
        # Refusing here keeps the failure inside this function's own
        # unavailable-with-a-reason contract instead of crashing the run.
        reasons.append(
            f"provider-reported {', '.join(oversized)} exceeds "
            f"{MAX_BILLABLE_TOKEN_COUNT}, above the largest count that can be "
            "billed exactly; refusing to derive a cost from it"
        )
        return _unavailable(entry, reasons)

    if usage.prompt_tokens is None:
        reasons.append(
            "provider did not report prompt_tokens, so input cost cannot be "
            "estimated"
        )
    if usage.completion_tokens is None:
        reasons.append(
            "provider did not report completion_tokens, so output cost cannot "
            "be estimated"
        )
    if reasons:
        return _unavailable(entry, reasons)

    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    assert prompt_tokens is not None and completion_tokens is not None

    cached = usage.cached_prompt_tokens
    if cached is None and entry.cached_input_per_million is not None:
        # The manifest prices a cache tier, so this provider has one. With no
        # reported cached count there is no way to know how much of the
        # prompt was served from it. Treating the absence as "nothing was
        # cached" bills every token at the full input rate, which overstates
        # the cost; treating it as fully cached understates it. Neither is a
        # measurement, so the total is unavailable.
        reasons.append(
            f"pricing entry {entry.entry_id!r} declares a "
            "cached_input_per_million rate but the provider reported no "
            "cached prompt token count; refusing to assume none of the "
            "prompt was cached"
        )
        return _unavailable(entry, reasons)
    if cached is not None and cached > prompt_tokens:
        reasons.append(
            f"provider reported {cached} cached prompt tokens but only "
            f"{prompt_tokens} prompt tokens in total; the usage block is "
            "internally inconsistent and cannot be billed"
        )
        return _unavailable(entry, reasons)

    if cached and entry.cached_input_per_million is None:
        # The provider says part of the prompt was served from cache, which
        # means a cache price exists; billing it at the full input rate would
        # overstate the cost, and billing it at zero would understate it.
        reasons.append(
            f"provider reported {cached} cached prompt token(s) but pricing "
            f"entry {entry.entry_id!r} declares no cached_input_per_million "
            "rate; refusing to bill cached tokens at the full input rate or "
            "at zero"
        )
        return _unavailable(entry, reasons)

    billed_cached = cached if entry.cached_input_per_million is not None else None
    billed_input = prompt_tokens - (billed_cached or 0)

    reasoning = usage.reasoning_tokens
    if entry.reasoning_per_million is not None and reasoning is None:
        reasons.append(
            f"pricing entry {entry.entry_id!r} declares a separate "
            "reasoning_per_million rate but the provider did not report "
            "reasoning tokens; refusing to infer hidden reasoning usage"
        )
        return _unavailable(entry, reasons)

    if reasoning is not None and reasoning > completion_tokens:
        reasons.append(
            f"provider reported {reasoning} reasoning tokens but only "
            f"{completion_tokens} completion tokens in total; the usage block "
            "is internally inconsistent and cannot be billed"
        )
        return _unavailable(entry, reasons)

    if entry.reasoning_per_million is not None:
        billed_reasoning = reasoning
        billed_output = completion_tokens - (reasoning or 0)
    else:
        # Standard OpenAI-compatible behaviour: reasoning tokens are part of
        # completion_tokens and are billed at the output rate.
        billed_reasoning = None
        billed_output = completion_tokens
        if reasoning is None:
            reasons.append(
                "provider did not report reasoning token usage; completion "
                "tokens are billed at the output rate and no hidden reasoning "
                "cost is inferred"
            )

    components: list[tuple[str, int, float | None]] = [
        ("input_per_million", billed_input, entry.input_per_million),
        ("output_per_million", billed_output, entry.output_per_million),
    ]
    if billed_cached is not None:
        components.append(
            ("cached_input_per_million", billed_cached, entry.cached_input_per_million)
        )
    if billed_reasoning is not None:
        components.append(
            ("reasoning_per_million", billed_reasoning, entry.reasoning_per_million)
        )

    missing = [name for name, tokens, rate in components if tokens > 0 and rate is None]
    if missing:
        reasons.append(
            f"pricing entry {entry.entry_id!r} is missing rate(s) "
            f"{', '.join(sorted(missing))} needed for the reported usage"
        )
        return _unavailable(
            entry,
            reasons,
            billed_input_tokens=billed_input,
            billed_cached_tokens=billed_cached,
            billed_output_tokens=billed_output,
            billed_reasoning_tokens=billed_reasoning,
        )

    amount = sum(
        tokens * (rate or 0.0) / TOKENS_PER_RATE_UNIT
        for _name, tokens, rate in components
    )
    if not math.isfinite(amount):  # pragma: no cover - defensive
        reasons.append("computed cost is not a finite number")
        return _unavailable(entry, reasons)

    return CostBreakdown(
        amount=amount,
        currency=entry.currency,
        entry_id=entry.entry_id,
        entry_sha256=entry.content_sha256,
        rates_are_illustrative=entry.rates_are_illustrative,
        billed_input_tokens=billed_input,
        billed_cached_tokens=billed_cached,
        billed_output_tokens=billed_output,
        billed_reasoning_tokens=billed_reasoning,
        reasons=tuple(reasons),
    )


def cost_per_case(total_cost: float | None, case_count: int) -> float | None:
    """Mean estimated cost of one evaluated case, or ``None`` if undefined."""
    if total_cost is None or case_count <= 0:
        return None
    value = total_cost / case_count
    return value if math.isfinite(value) else None


def correct_cases_per_currency_unit(
    correct_cases: int, total_cost: float | None
) -> float | None:
    """Correct cases bought per unit of currency, or ``None`` if undefined.

    A zero total cost is *not* treated as infinite throughput-per-money: it
    makes the ratio undefined, and undefined is reported as unavailable.
    """
    if total_cost is None or total_cost <= 0 or correct_cases <= 0:
        return None
    value = correct_cases / total_cost
    return value if math.isfinite(value) else None
