import pytest

from llmtracefx.hardware import (
    get_hardware_profile,
    hardware_profiles,
    normalize_hardware_name,
    supported_hardware,
)


def test_gb10_profile_uses_dgx_spark_specs():
    profile = get_hardware_profile("GB10")

    assert profile.backend == "CUDA"
    assert profile.memory_size_gb == 128
    assert profile.memory_bandwidth_gb_s == 273
    assert profile.unified_memory is True


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("dgx-spark", "GB10"),
        ("apple silicon", "MLX"),
        ("metal", "MLX"),
        ("h100", "H100"),
    ],
)
def test_hardware_aliases(alias, expected):
    assert normalize_hardware_name(alias) == expected


def test_unknown_hardware_has_actionable_error():
    with pytest.raises(ValueError, match="Choose one of: A10G, A100, H100, GB10, MLX"):
        get_hardware_profile("unknown")


def test_supported_hardware_includes_new_profiles():
    assert supported_hardware() == ["A10G", "A100", "H100", "GB10", "MLX"]


def test_hardware_profiles_are_json_ready_and_complete():
    profiles = hardware_profiles()

    assert [profile["name"] for profile in profiles] == supported_hardware()
    assert all(profile["backend"] in {"CUDA", "Metal"} for profile in profiles)
    assert next(profile for profile in profiles if profile["name"] == "MLX") == {
        "name": "MLX",
        "display_name": "Apple Silicon (MLX / Metal)",
        "vendor": "Apple",
        "backend": "Metal",
        "memory_bandwidth_gb_s": None,
        "memory_size_gb": None,
        "compute_units": None,
        "unified_memory": True,
        "occupancy_label": "GPU occupancy",
    }
