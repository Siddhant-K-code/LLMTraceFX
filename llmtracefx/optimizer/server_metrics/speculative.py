"""Speculative decoding acceptance from server counters.

A runtime profile names the counters its server exports for speculative
decoding through :class:`SpeculativeCounters`. The evidence block built
here has the same shape for every runtime, so a new profile only has to
name its families. Every value is a difference of monotonic counters over
the scrape window. A gauge that holds the most recent value, such as
SGLang's ``sglang:spec_accept_length``, says nothing about the window and
is never differenced; it is listed under ``ignored_gauges`` instead.

When speculation is off or the server does not export the counters, the
block says so in ``status`` and carries no acceptance values at all, so a
missing measurement is never read as an acceptance of zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..schema import MetricProvenance
from .prometheus import MetricFamily

MEASURED = "measured"
NO_DRAFTS = "no_drafts"
NOT_EXPORTED = "not_exported"
UNSUPPORTED = "unsupported"
WITHHELD = "withheld"

POSITION_LABEL = "position"


@dataclass(frozen=True)
class SpeculativeCounters:
    """Counter families a runtime exports for speculative decoding.

    Names are canonical family names (counters without ``_total``). Every
    family must be a counter. Leave a field ``None`` when the runtime has no
    such counter; the value that needs it is then reported as not derivable
    rather than guessed.
    """

    verify_steps: str | None = None
    """Verification steps. One step emits the accepted drafts plus one
    token from the target model."""
    draft_tokens: str | None = None
    """Draft tokens proposed to the target model."""
    accepted_tokens: str | None = None
    """Draft tokens the target model accepted."""
    emitted_tokens: str | None = None
    """Tokens emitted. Used for the acceptance length only when
    ``accepted_tokens`` is ``None``."""
    accepted_per_position: str | None = None
    """Accepted drafts counted per draft position, labelled ``position``."""
    ignored_gauges: tuple[str, ...] = ()
    """Last-value gauges the server also exports. Named so the evidence can
    say they were not used."""

    def families(self) -> tuple[str, ...]:
        """The families whose presence means the server exports speculation."""
        return tuple(
            name
            for name in (
                self.verify_steps,
                self.draft_tokens,
                self.accepted_tokens,
                self.accepted_per_position,
            )
            if name is not None
        )


def _value_samples(family: MetricFamily) -> list[tuple[dict[str, str], float]]:
    return [
        (dict(sample.labels), sample.value)
        for sample in family.samples
        if not sample.name.endswith("_created")
    ]


def _total_delta(
    name: str,
    before: Mapping[str, MetricFamily],
    after: Mapping[str, MetricFamily],
) -> float | None:
    family = after.get(name)
    if family is None:
        return None
    end = math.fsum(value for _, value in _value_samples(family))
    previous = before.get(name)
    start = (
        0.0
        if previous is None
        else math.fsum(value for _, value in _value_samples(previous))
    )
    return end - start


def _position_deltas(
    name: str,
    before: Mapping[str, MetricFamily],
    after: Mapping[str, MetricFamily],
) -> dict[int, float]:
    def by_position(family: MetricFamily | None) -> dict[int, float]:
        totals: dict[int, float] = {}
        if family is None:
            return totals
        for labels, value in _value_samples(family):
            raw = labels.get(POSITION_LABEL)
            if raw is None or not raw.isdigit():
                continue
            totals[int(raw)] = totals.get(int(raw), 0.0) + value
        return totals

    start = by_position(before.get(name))
    end = by_position(after.get(name))
    return {pos: end[pos] - start.get(pos, 0.0) for pos in sorted(end)}


def _count(value: float, unit: str, source: str) -> dict[str, Any]:
    return {
        "value": int(value) if value.is_integer() else value,
        "provenance": MetricProvenance.MEASURED_NATIVE.value,
        "unit": unit,
        "statistic": "delta",
        "source": source,
    }


def _derived(value: float, unit: str, source: str, basis: str) -> dict[str, Any]:
    return {
        "value": value,
        "provenance": MetricProvenance.DERIVED.value,
        "unit": unit,
        "statistic": "ratio_of_deltas",
        "source": source,
        "basis": basis,
    }


def speculative_evidence(
    spec: SpeculativeCounters | None,
    runtime: str,
    before: Mapping[str, MetricFamily],
    after: Mapping[str, MetricFamily],
    *,
    attributable: bool,
    withheld_reason: str,
) -> dict[str, Any]:
    """Build the ``speculative`` block of server metrics evidence."""
    if spec is None:
        return {
            "status": UNSUPPORTED,
            "reason": f"the {runtime} profile names no speculative decoding counters",
        }
    block: dict[str, Any] = {
        "ignored_gauges": sorted(g for g in spec.ignored_gauges if g in after),
    }
    present = [name for name in spec.families() if name in after]
    not_counters = sorted(
        name
        for name in (*spec.families(), spec.emitted_tokens)
        if name is not None and name in after and after[name].type != "counter"
    )
    if not_counters:
        return {
            "status": WITHHELD,
            "reason": (
                f"{', '.join(not_counters)} is not exported as a counter, so "
                "its window delta means nothing"
            ),
            **block,
        }
    if not present:
        return {
            "status": NOT_EXPORTED,
            "reason": (
                f"the server exports none of {', '.join(spec.families())}; "
                "speculative decoding is off or this server version does not "
                "export it"
            ),
            **block,
        }
    if not attributable:
        return {
            "status": WITHHELD,
            "reason": withheld_reason,
            **block,
        }

    deltas: dict[str, tuple[float, str]] = {}
    for role in ("verify_steps", "draft_tokens", "accepted_tokens", "emitted_tokens"):
        name = getattr(spec, role)
        if name is None:
            continue
        delta = _total_delta(name, before, after)
        if delta is None:
            continue
        if not math.isfinite(delta) or delta < 0:
            return {
                "status": WITHHELD,
                "reason": f"{name} did not increase monotonically in the window",
                **block,
            }
        deltas[role] = (delta, name)

    steps = deltas.get("verify_steps", (0.0, ""))[0]
    drafted = deltas.get("draft_tokens", (0.0, ""))[0]
    if steps <= 0 and drafted <= 0:
        counted = ", ".join(
            deltas[role][1]
            for role in ("verify_steps", "draft_tokens")
            if role in deltas
        )
        return {
            "status": NO_DRAFTS,
            "reason": (
                f"{counted or ', '.join(present)} did not increase in the "
                "window; no speculative step ran, so there is no acceptance "
                "to report"
            ),
            **block,
        }

    result: dict[str, Any] = {
        "status": MEASURED,
        "reason": "speculative decoding counters increased in the window",
        "counters": {
            role: _count(value, "steps" if role == "verify_steps" else "tokens", name)
            for role, (value, name) in deltas.items()
        },
    }
    not_derived: dict[str, str] = {}

    if "accepted_tokens" in deltas and drafted > 0:
        accepted, accepted_name = deltas["accepted_tokens"]
        result["acceptance_rate"] = _derived(
            accepted / drafted,
            "ratio",
            f"{accepted_name} / {deltas['draft_tokens'][1]}",
            "accepted draft tokens / proposed draft tokens",
        )
    else:
        not_derived["acceptance_rate"] = (
            "needs counters of both proposed and accepted draft tokens"
        )

    if steps > 0 and "accepted_tokens" in deltas:
        accepted, accepted_name = deltas["accepted_tokens"]
        result["acceptance_length"] = _derived(
            1.0 + accepted / steps,
            "tokens/step",
            f"1 + {accepted_name} / {deltas['verify_steps'][1]}",
            "1 + accepted draft tokens / verification steps; the token the "
            "target model adds at each step is counted",
        )
    elif steps > 0 and "emitted_tokens" in deltas:
        emitted, emitted_name = deltas["emitted_tokens"]
        result["acceptance_length"] = _derived(
            emitted / steps,
            "tokens/step",
            f"{emitted_name} / {deltas['verify_steps'][1]}",
            "tokens emitted / verification steps; counts every request that "
            "finished in the window, so requests that did not speculate "
            "raise it",
        )
    else:
        not_derived["acceptance_length"] = (
            "needs a counter of verification steps with accepted or emitted tokens"
        )

    if spec.accepted_per_position is not None and steps > 0:
        positions = _position_deltas(spec.accepted_per_position, before, after)
        rates = [
            {
                "position": pos,
                **_derived(
                    max(delta, 0.0) / steps,
                    "ratio",
                    f'{spec.accepted_per_position}{{position="{pos}"}} / '
                    f"{deltas['verify_steps'][1]}",
                    "drafts accepted at this position / verification steps",
                ),
            }
            for pos, delta in positions.items()
        ]
        if rates:
            result["per_position_acceptance_rate"] = rates
        else:
            not_derived["per_position_acceptance_rate"] = (
                f"{spec.accepted_per_position} has no position series"
            )
    elif spec.accepted_per_position is None:
        not_derived["per_position_acceptance_rate"] = (
            "the server exports no per-position counter"
        )

    if not_derived:
        result["not_derived"] = not_derived
    result.update(block)
    return result
