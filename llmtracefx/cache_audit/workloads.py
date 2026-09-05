"""Deterministic adversarial token workloads for cache auditing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from .schema import PairRole, RequestSpec, ScenarioKind


def adversarial_requests(*, block_size: int = 4) -> tuple[RequestSpec, ...]:
    """Return a compact deterministic workload covering identity edge cases."""

    if block_size < 2:
        raise ValueError("block_size must be at least two")
    base = tuple(range(10, 10 + block_size * 2 + 1))

    def request(
        request_id: str,
        scenario: ScenarioKind,
        tokens: tuple[int, ...],
        *,
        mutation_position: int | None = None,
        namespace_id: str = "tenant-a",
        predecessors: tuple[str, ...] = (),
        pair_role: PairRole = PairRole.SINGLE,
    ) -> RequestSpec:
        return RequestSpec(
            request_id=request_id,
            scenario=scenario,
            order=len(items),
            input_token_ids=tokens,
            input_token_count=len(tokens),
            output_tokens=2,
            pair_id="base-pair" if request_id in {"cold", "identical"} else None,
            pair_role=pair_role,
            mutation_position=mutation_position,
            expected_predecessors=predecessors,
            namespace_id=namespace_id,
        )

    items: list[RequestSpec] = []
    items.append(request("cold", ScenarioKind.COLD, base, pair_role=PairRole.CONTROL))
    items.append(
        request(
            "identical",
            ScenarioKind.IDENTICAL_PREFIX,
            base,
            predecessors=("cold",),
            pair_role=PairRole.TREATMENT,
        )
    )
    items.append(
        request(
            "first-token-mutation",
            ScenarioKind.FIRST_TOKEN_MUTATION,
            (999,) + base[1:],
            mutation_position=0,
        )
    )
    inside = max(1, block_size - 1)
    items.append(
        request(
            "within-block-mutation",
            ScenarioKind.WITHIN_BLOCK_MUTATION,
            base[:inside] + (998,) + base[inside + 1 :],
            mutation_position=inside,
        )
    )
    items.append(
        request(
            "boundary-mutation",
            ScenarioKind.BLOCK_BOUNDARY_MUTATION,
            base[:block_size] + (997,) + base[block_size + 1 :],
            mutation_position=block_size,
        )
    )
    items.append(
        request(
            "suffix-change",
            ScenarioKind.SUFFIX_CHANGE,
            base[:-1] + (996,),
            mutation_position=len(base) - 1,
        )
    )
    items.append(
        request(
            "same-length-different-ids",
            ScenarioKind.SAME_LENGTH_DIFFERENT_IDS,
            tuple(token + 1000 for token in base),
        )
    )
    items.append(
        request(
            "duplicate",
            ScenarioKind.DUPLICATE,
            base,
            predecessors=("identical",),
        )
    )
    items.append(
        request(
            "namespace-isolation",
            ScenarioKind.NAMESPACE_ISOLATION,
            base,
            namespace_id="tenant-b",
            predecessors=("cold",),
        )
    )
    return tuple(items)


def workload_digest(requests: Sequence[RequestSpec]) -> str:
    """Hash an executable workload without relying on object identity."""

    payload = [request.to_dict(include_tokens=True) for request in requests]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def eviction_requests() -> tuple[RequestSpec, ...]:
    """Return a controlled sequence whose first entry is evicted at capacity one."""

    return (
        RequestSpec(
            request_id="eviction-seed-a",
            scenario=ScenarioKind.COLD,
            order=0,
            input_token_ids=(1, 2, 3, 4),
            input_token_count=4,
        ),
        RequestSpec(
            request_id="eviction-seed-b",
            scenario=ScenarioKind.COLD,
            order=1,
            input_token_ids=(5, 6, 7, 8),
            input_token_count=4,
        ),
        RequestSpec(
            request_id="eviction-revisit-a",
            scenario=ScenarioKind.EVICTION_COUNT,
            order=2,
            input_token_ids=(1, 2, 3, 4),
            input_token_count=4,
            expected_predecessors=("eviction-seed-a",),
        ),
    )


def gated_extension_requests() -> tuple[RequestSpec, ...]:
    """Return scenarios that require backend-specific instrumentation gates."""

    scenarios = (
        ScenarioKind.MIXED_LENGTH_CONCURRENT,
        ScenarioKind.SAVED_CACHE_MISMATCH,
        ScenarioKind.QUANTIZED_CACHE,
        ScenarioKind.ROTATING_CACHE,
        ScenarioKind.MULTIMODAL_IDENTITY,
        ScenarioKind.HASH_COLLISION,
        ScenarioKind.PREEMPTION,
        ScenarioKind.SPECULATIVE,
    )
    return tuple(
        RequestSpec(
            request_id=f"gated-{scenario.value}",
            scenario=scenario,
            order=index,
            input_token_ids=(10, 11, 12, 13 + index),
            input_token_count=4,
        )
        for index, scenario in enumerate(scenarios)
    )
