"""The Qwen3-8B benchmark/control manifest: a template bound from a receipt.

The packaged control-manifest template intentionally omits
``model.files``, ``model.expected_download_bytes``, and
``model.revision``: those three fields can only be known once a real
self-conversion has actually produced and hashed output files, and this
project never commits fabricated hashes to make a manifest "look"
complete. ``bind_control_manifest`` fills them in from a verified
``ConversionReceipt`` and returns a fully valid
``llmtracefx.optimizer.lab.manifest.LabManifest`` -- the exact same
class the packaged 27B lab manifest parses as, so every existing
``lab.core`` verification/report/evidence function keeps working
unmodified against it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..manifest import LabManifest, LabManifestError
from .conversion import ConversionReceipt, conversion_manifest_hash
from .conversion_manifest import ConversionManifest

CONTROL_MANIFEST_TEMPLATE_SCHEMA_VERSION = "1"

#: ``model`` fields a template must NOT carry: they can only be known
#: from a verified, completed conversion receipt (see module docstring).
_PENDING_MODEL_FIELDS = ("files", "expected_download_bytes", "revision")

#: Placeholder used only to validate every *other* template section
#: through the real ``LabManifest`` parser without ever writing a fake
#: hash anywhere persistent. Never returned to a caller.
_PROBE_FILE_SHA256 = "0" * 64
_PROBE_REVISION = "0" * 40


class ControlManifestError(ValueError):
    """Raised when the control manifest template or a bind attempt is invalid."""


def _probe_model(model_template: dict[str, Any]) -> dict[str, Any]:
    probe = copy.deepcopy(model_template)
    probe["files"] = [
        {"path": "PENDING-CONVERSION", "size_bytes": 1, "sha256": _PROBE_FILE_SHA256}
    ]
    probe["expected_download_bytes"] = 1
    probe["revision"] = _PROBE_REVISION
    return probe


class ControlManifestTemplate:
    """A control-manifest template, valid in every section but ``model``'s
    conversion-derived identity, which is bound in later from a receipt."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControlManifestTemplate:
        if not isinstance(data, dict):
            raise ControlManifestError("control manifest template must be an object")
        schema_version = data.get("schema_version")
        if schema_version != CONTROL_MANIFEST_TEMPLATE_SCHEMA_VERSION:
            raise ControlManifestError(
                "unsupported control manifest template schema_version "
                f"{schema_version!r}"
            )
        model_template = data.get("model")
        if not isinstance(model_template, dict):
            raise ControlManifestError(
                "control manifest template.model must be an object"
            )
        present_pending_fields = [
            field for field in _PENDING_MODEL_FIELDS if field in model_template
        ]
        if present_pending_fields:
            raise ControlManifestError(
                "control manifest template.model must not pre-commit "
                f"{present_pending_fields}; those are only known after a "
                "verified conversion receipt is bound"
            )
        # Validate every other section (runtime/generation/repetitions/
        # safety/workloads/context_tiers/artifacts/environment_capture)
        # through the real, load-bearing LabManifest parser by injecting
        # a clearly-fake, never-persisted probe identity. This reuses the
        # existing manifest validators completely instead of duplicating
        # them, and it never risks a real hash being read from this path.
        probe = dict(data)
        probe["model"] = _probe_model(model_template)
        try:
            LabManifest.from_dict(probe)
        except LabManifestError as exc:
            raise ControlManifestError(
                f"control manifest template is invalid: {exc}"
            ) from exc
        return cls(data)

    @classmethod
    def from_json(cls, payload: str) -> ControlManifestTemplate:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise ControlManifestError(
                f"invalid control manifest template JSON: {exc}"
            ) from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> ControlManifestTemplate:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise ControlManifestError(
                f"invalid control manifest template file: {exc}"
            ) from exc
        return cls.from_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._raw)


def _synthetic_revision(receipt: ConversionReceipt) -> str:
    """A deterministic 40-hex identity fingerprint for one self-converted
    output, used exactly like ``ModelPin.revision`` is used elsewhere: to
    detect a stale or drifted binding, never as a claim of an upstream
    git commit. Documented in the bound manifest's own sources list.

    Callers must never present this value as an upstream git commit
    hash; it is a locally derived fingerprint of the conversion
    identity (see ``binding_revision_provenance`` in ``report.py``),
    coincidentally sharing a git-revision-shaped 40-hex-character
    encoding only because ``ModelPin.revision`` requires that shape.
    """
    payload = {
        "conversion_id": receipt.conversion_id,
        "source_official_revision": receipt.source["official_revision"],
        "converter_git_revision": receipt.converter["git_revision"],
        "converter_version": receipt.converter["version"],
        "parameters": receipt.parameters,
        "output_files": [dict(item) for item in receipt.output_files],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:40]


def _provenance_mismatches(
    receipt: ConversionReceipt, conversion_manifest: ConversionManifest
) -> tuple[str, ...]:
    """Every field where the receipt disagrees with the packaged spec.

    A receipt binds only if it describes *this exact* pinned official
    source identity, converter identity, conversion ID, and full
    quantization parameter set -- never a receipt that merely has the
    right shape.
    """
    mismatches: list[str] = []

    def check(label: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            mismatches.append(f"{label}: receipt has {actual!r}, expected {expected!r}")

    check(
        "source.official_id",
        receipt.source.get("official_id"),
        conversion_manifest.source.official_id,
    )
    check(
        "source.official_revision",
        receipt.source.get("official_revision"),
        conversion_manifest.source.official_revision,
    )
    check(
        "source.license",
        receipt.source.get("license"),
        conversion_manifest.source.license,
    )
    check(
        "converter.package",
        receipt.converter.get("package"),
        conversion_manifest.converter.package,
    )
    check(
        "converter.version",
        receipt.converter.get("version"),
        conversion_manifest.converter.version,
    )
    check(
        "converter.git_revision",
        receipt.converter.get("git_revision"),
        conversion_manifest.converter.git_revision,
    )
    check("conversion_id", receipt.conversion_id, conversion_manifest.conversion_id)
    check(
        "parameters",
        receipt.parameters,
        dataclasses.asdict(conversion_manifest.parameters),
    )
    check(
        "conversion_manifest_hash",
        receipt.conversion_manifest_hash,
        conversion_manifest_hash(conversion_manifest),
    )
    return tuple(mismatches)


def build_bound_manifest_payload(
    template: ControlManifestTemplate,
    receipt: ConversionReceipt,
    *,
    conversion_manifest: ConversionManifest,
) -> dict[str, Any]:
    """The plain JSON-shaped dict a bound control manifest would contain.

    Exposed separately from ``bind_control_manifest`` so callers that
    need to persist the exact bound manifest to disk (the ``bind`` CLI
    action) never have to reconstruct ``model.files``/``revision`` by
    hand from the returned ``LabManifest`` -- there is exactly one place
    that assembles this shape.
    """
    if receipt.status != "completed":
        raise ControlManifestError(
            "cannot bind a control manifest from a non-completed conversion receipt"
        )
    mismatches = _provenance_mismatches(receipt, conversion_manifest)
    if mismatches:
        raise ControlManifestError(
            "conversion receipt provenance does not match the packaged "
            "conversion spec, refusing to bind: " + "; ".join(mismatches)
        )
    data = template.to_dict()
    model = dict(data["model"])
    model["files"] = [
        {
            "path": item["path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for item in receipt.output_files
    ]
    model["expected_download_bytes"] = receipt.output_total_bytes
    model["revision"] = _synthetic_revision(receipt)
    data["model"] = model
    return data


def bind_control_manifest(
    template: ControlManifestTemplate,
    receipt: ConversionReceipt,
    *,
    conversion_manifest: ConversionManifest,
) -> LabManifest:
    """Fill in ``model.files``/``expected_download_bytes``/``revision``
    from a verified, completed conversion receipt and return a fully
    valid ``LabManifest``.

    ``conversion_manifest`` is the packaged (or explicitly pinned)
    conversion spec this receipt claims to be the output of. Binding is
    refused if the receipt's official source ID/revision/license,
    converter package/version/git revision, conversion ID, or full
    quantization parameters differ from it in any way -- a receipt with
    the right *shape* but the wrong *identity* must never silently bind.

    Raises ``ControlManifestError`` if the receipt is not a completed
    conversion, if its provenance disagrees with ``conversion_manifest``,
    or if binding still produces an internally inconsistent manifest
    (``LabManifest.from_dict`` re-validates everything).
    """
    data = build_bound_manifest_payload(
        template, receipt, conversion_manifest=conversion_manifest
    )
    try:
        return LabManifest.from_dict(data)
    except LabManifestError as exc:
        raise ControlManifestError(f"bound control manifest is invalid: {exc}") from exc


__all__ = [
    "CONTROL_MANIFEST_TEMPLATE_SCHEMA_VERSION",
    "ControlManifestError",
    "ControlManifestTemplate",
    "bind_control_manifest",
    "build_bound_manifest_payload",
]
