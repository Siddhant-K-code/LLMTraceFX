"""Hardware profiles supported by LLMTraceFX."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    """Static details used to interpret a trace for a target accelerator."""

    name: str
    display_name: str
    vendor: str
    backend: str
    memory_bandwidth_gb_s: float | None
    memory_size_gb: float | None
    compute_units: int | None
    unified_memory: bool = False
    occupancy_label: str = "SM occupancy"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


_PROFILES = {
    "A10G": HardwareProfile(
        name="A10G",
        display_name="NVIDIA A10G",
        vendor="NVIDIA",
        backend="CUDA",
        memory_bandwidth_gb_s=600,
        memory_size_gb=24,
        compute_units=80,
    ),
    "A100": HardwareProfile(
        name="A100",
        display_name="NVIDIA A100 80GB",
        vendor="NVIDIA",
        backend="CUDA",
        memory_bandwidth_gb_s=1935,
        memory_size_gb=80,
        compute_units=108,
    ),
    "H100": HardwareProfile(
        name="H100",
        display_name="NVIDIA H100 80GB",
        vendor="NVIDIA",
        backend="CUDA",
        memory_bandwidth_gb_s=3350,
        memory_size_gb=80,
        compute_units=132,
    ),
    "GB10": HardwareProfile(
        # https://docs.nvidia.com/dgx/dgx-spark/hardware.html
        name="GB10",
        display_name="NVIDIA GB10 / DGX Spark",
        vendor="NVIDIA",
        backend="CUDA",
        memory_bandwidth_gb_s=273,
        memory_size_gb=128,
        compute_units=None,
        unified_memory=True,
    ),
    "MLX": HardwareProfile(
        name="MLX",
        display_name="Apple Silicon (MLX / Metal)",
        vendor="Apple",
        backend="Metal",
        memory_bandwidth_gb_s=None,
        memory_size_gb=None,
        compute_units=None,
        unified_memory=True,
        occupancy_label="GPU occupancy",
    ),
}

_ALIASES = {
    "APPLE_SILICON": "MLX",
    "DGX_SPARK": "GB10",
    "METAL": "MLX",
}


def _clean_name(name: str) -> str:
    return name.strip().upper().replace("-", "_").replace(" ", "_")


def normalize_hardware_name(name: str) -> str:
    """Normalize a hardware name or raise a useful error."""
    cleaned = _clean_name(name)
    canonical = _ALIASES.get(cleaned, cleaned)
    if canonical not in _PROFILES:
        choices = ", ".join(supported_hardware())
        raise ValueError(f"Unsupported hardware '{name}'. Choose one of: {choices}")
    return canonical


def get_hardware_profile(name: str) -> HardwareProfile:
    """Return a hardware profile, accepting common aliases."""
    return _PROFILES[normalize_hardware_name(name)]


def supported_hardware() -> list[str]:
    """Return canonical hardware names in CLI display order."""
    return list(_PROFILES)


def hardware_profiles() -> list[dict[str, Any]]:
    """Return all profiles for API clients and user interfaces."""
    return [profile.to_dict() for profile in _PROFILES.values()]
