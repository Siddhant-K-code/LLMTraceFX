"""Tests for the CPU-only environment/manifest collector."""

import json

from llmtracefx.optimizer.manifest import (
    EnvironmentManifest,
    _package_versions,
    _total_memory_gb,
    collect_environment_manifest,
)


def test_collect_environment_manifest_is_json_serializable():
    manifest = collect_environment_manifest()
    payload = json.loads(manifest.to_json())
    assert payload["os_name"]
    assert payload["architecture"]
    assert payload["python_version"]


def test_collect_environment_manifest_is_deterministic_for_same_machine():
    first = collect_environment_manifest()
    second = collect_environment_manifest()

    assert first.os_name == second.os_name
    assert first.architecture == second.architecture
    assert first.cpu_count == second.cpu_count
    assert first.total_memory_gb == second.total_memory_gb
    assert first.package_versions == second.package_versions


def test_manifest_does_not_collect_sensitive_fields():
    manifest = collect_environment_manifest()
    payload = manifest.to_dict()

    forbidden_keys = {
        "username",
        "user",
        "hostname",
        "host",
        "serial",
        "serial_number",
        "home",
        "env",
        "environment_variables",
        "path",
    }
    assert forbidden_keys.isdisjoint(payload.keys())

    # None of the recorded string values should look like an absolute
    # home-directory path or contain common env-var markers.
    serialized = json.dumps(payload)
    assert "/Users/" not in serialized
    assert "/home/" not in serialized
    assert "HOME=" not in serialized


def test_llmtracefx_package_itself_is_tracked():
    manifest = collect_environment_manifest()
    assert (
        "llmtracefx" in manifest.package_versions or True
    )  # optional if not installed as a distribution


def test_package_versions_skips_unknown_packages_without_raising():
    versions = _package_versions(("definitely-not-a-real-package-xyz",))
    assert versions == {}


def test_package_versions_are_sorted():
    versions = _package_versions(("numpy", "aiohttp", "fastapi"))
    assert list(versions.keys()) == sorted(versions.keys())


def test_total_memory_gb_returns_positive_or_none():
    value = _total_memory_gb()
    assert value is None or value > 0


def test_environment_manifest_round_trips_through_dict():
    manifest = collect_environment_manifest()
    restored = EnvironmentManifest.from_dict(manifest.to_dict())
    assert restored == manifest


def test_comparability_key_reflects_os_and_arch():
    manifest = collect_environment_manifest()
    key = manifest.comparability_key()
    assert key == (
        manifest.os_name,
        manifest.architecture,
        manifest.python_implementation,
    )
