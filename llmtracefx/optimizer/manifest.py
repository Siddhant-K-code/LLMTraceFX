"""CPU-only environment/manifest collector.

Records deterministic, non-sensitive environment metadata that later
collectors and the doctor can use to check "comparability" between runs
(e.g. did two runs execute on the same OS/architecture with the same
package versions?).

This module intentionally does **not** collect: secrets, usernames,
hostnames, serial numbers, full environment variable dumps, or absolute
paths that could leak a user's home directory or username. Only package
names come from this project's own dependency list (``pyproject.toml``),
so there is nothing dynamic or user-supplied to leak.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, utc_now_iso

# Fixed, deterministic list of first/third-party packages this project
# depends on (mirrors pyproject.toml). Kept as a tuple literal (rather than
# parsed from pyproject.toml at runtime) so manifest collection has no
# dependency on the source layout being present, e.g. when installed as a
# wheel.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "llmtracefx",
    "fastapi",
    "uvicorn",
    "aiohttp",
    "plotly",
    "pandas",
    "numpy",
    "modal",
    "streamlit",
)


@dataclass(frozen=True)
class EnvironmentManifest:
    """Deterministic, non-sensitive environment metadata for a run."""

    schema_version: str
    collected_at: str
    os_name: str
    os_release: str
    architecture: str
    python_implementation: str
    python_version: str
    cpu_count: int | None
    total_memory_gb: float | None
    package_versions: dict[str, str] = field(default_factory=dict)
    generator: str = "llmtracefx.optimizer.manifest"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentManifest:
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            collected_at=data["collected_at"],
            os_name=data["os_name"],
            os_release=data["os_release"],
            architecture=data["architecture"],
            python_implementation=data["python_implementation"],
            python_version=data["python_version"],
            cpu_count=data.get("cpu_count"),
            total_memory_gb=data.get("total_memory_gb"),
            package_versions=dict(data.get("package_versions", {})),
            generator=data.get("generator", "llmtracefx.optimizer.manifest"),
        )

    def comparability_key(self) -> tuple[str, str, str]:
        """A coarse key for deciding whether two manifests are comparable.

        Two runs on different OS families or CPU architectures are not
        directly comparable for latency/throughput purposes.
        """
        return (self.os_name, self.architecture, self.python_implementation)


def _total_memory_gb() -> float | None:
    """Best-effort total physical memory in GiB, or ``None`` if unknown.

    Uses only OS-level facts (no user/process info). Returns ``None``
    rather than guessing when the platform is unsupported or the lookup
    fails, so a missing value is never silently mistaken for zero.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            return int(output) / (1024**3)
        if system == "Linux":
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            match = re.search(r"^MemTotal:\s+(\d+)\s*kB", meminfo, re.MULTILINE)
            if match is None:
                return None
            return int(match.group(1)) * 1024 / (1024**3)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    """Resolve installed versions for a fixed, deterministic package list."""
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return dict(sorted(versions.items()))


def collect_environment_manifest(
    *, extra_packages: tuple[str, ...] = ()
) -> EnvironmentManifest:
    """Collect a deterministic, non-sensitive snapshot of the environment.

    ``extra_packages`` lets callers (e.g. the llama.cpp collector) request
    versions of additional installed packages beyond the tracked defaults,
    without introducing arbitrary/user-controlled lookups by default.
    """
    tracked = tuple(dict.fromkeys(_TRACKED_PACKAGES + extra_packages))
    return EnvironmentManifest(
        schema_version=SCHEMA_VERSION,
        collected_at=utc_now_iso(),
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count(),
        total_memory_gb=_total_memory_gb(),
        package_versions=_package_versions(tracked),
    )
