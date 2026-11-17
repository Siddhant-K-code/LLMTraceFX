"""NVIDIA GPU energy and power evidence through NVML.

``pynvml`` (the optional ``nvidia-ml-py`` extra) is only imported by
:func:`.nvml.load_backend`, so importing this package never needs it.
"""

from .capability import NvmlCapability, check_capability, probe_capability
from .evidence import EVIDENCE_KIND, build_evidence, write_evidence
from .nvml import NvmlBackend, NvmlCallError, NvmlUnavailable, load_backend
from .window import WindowRecord, record_window

__all__ = [
    "EVIDENCE_KIND",
    "NvmlBackend",
    "NvmlCallError",
    "NvmlCapability",
    "NvmlUnavailable",
    "WindowRecord",
    "build_evidence",
    "check_capability",
    "load_backend",
    "probe_capability",
    "record_window",
    "write_evidence",
]
