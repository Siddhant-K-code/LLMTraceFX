from __future__ import annotations

import pytest

from vllm_kv_truth import workload

#: The plan's exact preregistered table:
#: | Probe | Expected reusable tokens / blocks |
_EXPECTED_TABLE = {
    "cold_seed": (0, 0),
    "exact_full_hit": (256, 16),
    "shorter_prefix": (128, 8),
    "longer_prefix": (256, 16),
    "within_block_mutation": (16, 1),
    "block_boundary_mutation": (32, 2),
    "same_length_different_ids": (0, 0),
    "suffix_only_change": (256, 16),
    "duplicate_request": (320, 20),
    "salt_isolation": (0, 0),
}


def test_nested_probe_names_match_plan_order() -> None:
    assert tuple(probe.name for probe in workload.NESTED_PROBES) == tuple(
        _EXPECTED_TABLE
    )


@pytest.mark.parametrize("name", list(_EXPECTED_TABLE))
def test_nested_probe_matches_plan_table(name: str) -> None:
    probe = workload.probe_by_name(name)
    expected_tokens, expected_blocks = _EXPECTED_TABLE[name]
    assert probe.expected_reusable_tokens == expected_tokens
    assert probe.expected_reusable_blocks == expected_blocks


def test_probe_by_name_raises_on_unknown_probe() -> None:
    with pytest.raises(KeyError):
        workload.probe_by_name("does-not-exist")


def test_seed_and_longer_array_shapes() -> None:
    assert len(workload.BASE_ARRAY) == 256
    assert len(workload.SEED_ARRAY) == 257
    assert len(workload.LONGER_ARRAY) == 321
    assert workload.SEED_ARRAY[:256] == workload.BASE_ARRAY
    assert workload.LONGER_ARRAY[:257] == workload.SEED_ARRAY


def test_all_named_arrays_are_mutually_disjoint() -> None:
    seen: set[int] = set()
    groups = [
        workload.BASE_ARRAY,
        (workload.SEED_ARRAY[256],),  # the extra "token_a"
        workload.LONGER_ARRAY[257:],  # the extension
        workload._DISJOINT_ARRAY,
        *workload._EVICTION_FILLERS,
    ]
    for group in groups:
        for token in group:
            assert token not in seen, "workload token pool is not disjoint"
            seen.add(token)


def test_mutation_probes_differ_from_seed_at_the_expected_index() -> None:
    within = workload.probe_by_name("within_block_mutation").request_tokens
    boundary = workload.probe_by_name("block_boundary_mutation").request_tokens
    suffix = workload.probe_by_name("suffix_only_change").request_tokens
    seed = workload.SEED_ARRAY

    assert within[:31] == seed[:31]
    assert within[31] != seed[31]
    assert within[32:] == seed[32:]

    assert boundary[:32] == seed[:32]
    assert boundary[32] != seed[32]
    assert boundary[33:] == seed[33:]

    assert suffix[:256] == seed[:256]
    assert suffix[256] != seed[256]


def test_same_length_different_ids_probe_is_fully_disjoint_from_seed() -> None:
    probe = workload.probe_by_name("same_length_different_ids")
    assert len(probe.request_tokens) == len(workload.SEED_ARRAY)
    assert set(probe.request_tokens).isdisjoint(workload.SEED_ARRAY)


def test_duplicate_request_probe_repeats_longer_array_exactly() -> None:
    probe = workload.probe_by_name("duplicate_request")
    assert probe.request_tokens == workload.LONGER_ARRAY
    assert probe.cached_reference == workload.LONGER_ARRAY


def test_salt_isolation_probe_disables_identity_match() -> None:
    probe = workload.probe_by_name("salt_isolation")
    assert probe.request_tokens == workload.SEED_ARRAY
    assert probe.identity_matches is False


def test_cold_seed_probe_disables_identity_match_and_is_a_pure_miss() -> None:
    probe = workload.probe_by_name("cold_seed")
    assert probe.identity_matches is False
    assert probe.expected_reusable_tokens == 0


def test_eviction_lane_has_seed_five_fillers_then_a_final_seed_repeat() -> None:
    requests = workload.EVICTION_LANE_REQUESTS
    assert len(requests) == 7
    assert requests[0] == workload.SEED_ARRAY
    assert requests[-1] == workload.SEED_ARRAY
    for filler in requests[1:6]:
        assert len(filler) == 257
        assert set(filler).isdisjoint(workload.SEED_ARRAY)
    # the five fillers are themselves mutually disjoint
    seen: set[int] = set()
    for filler in requests[1:6]:
        assert seen.isdisjoint(filler)
        seen.update(filler)


def test_eviction_lane_cache_pressure_exceeds_usable_capacity() -> None:
    assert workload.CACHEABLE_BLOCKS_PER_EVICTION_LANE_REQUEST == 16
    assert workload.TOTAL_CACHEABLE_BLOCKS_BEFORE_FINAL_PROBE == 96
    assert workload.USABLE_BLOCK_CAPACITY == 95
    assert (
        workload.TOTAL_CACHEABLE_BLOCKS_BEFORE_FINAL_PROBE
        > workload.USABLE_BLOCK_CAPACITY
    )


def test_ordinary_token_range_excludes_low_special_ids() -> None:
    all_tokens = {
        *workload.BASE_ARRAY,
        *workload.SEED_ARRAY,
        *workload.LONGER_ARRAY,
        *workload._DISJOINT_ARRAY,
        *(token for filler in workload._EVICTION_FILLERS for token in filler),
    }
    for token in all_tokens:
        assert (
            workload.ORDINARY_TOKEN_ID_LOW
            <= token
            < workload.ORDINARY_TOKEN_ID_HIGH_EXCLUSIVE
        )
