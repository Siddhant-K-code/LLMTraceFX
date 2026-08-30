"""Strict, versioned pricing input for cross-system cost comparison.

Money is the one number in this project that cannot be measured locally, so
it is treated as an *input* with the same suspicion as any other untrusted
artifact rather than as a constant baked into the source:

* Rates arrive only through a user-supplied manifest file. Nothing here ships
  a live price, and nothing here fetches one.
* Every entry carries its own ``currency``, ``effective_at`` date and a
  ``source`` reference, so a number in a report can always be traced back to
  what it was read from and when.
* Matching an entry to a system is exact and unambiguous. Two entries that
  both match one system is a refusal, never a silent "first wins".
* A manifest whose entries are not explicitly sourced must declare
  ``rates_are_illustrative``. Illustrative rates still produce numbers, but
  the report labels every one of them as an example rather than a price.

The exact entry used for a cost figure is recorded by id *and* by content
hash (``PricingEntry.content_sha256``), so editing a manifest after the fact
cannot quietly change what a published report claimed to have used.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRICING_MANIFEST_SCHEMA_VERSION = "1"

#: ISO 4217 alphabetic codes are exactly three uppercase letters. Anything
#: else (a symbol, a lowercase code, a free-text name) is rejected so two
#: manifests can never be compared on a currency that was never pinned
#: down. Public so the report loader can hold a persisted currency to the
#: identical standard rather than restating the rule.
CURRENCY_PATTERN = re.compile(r"^[A-Z]{3}$")

#: Minimal ISO-8601 date or UTC timestamp. Deliberately narrow: a price is
#: only meaningful with a date attached, and a date this module cannot parse
#: is not a date it should accept.
_EFFECTIVE_AT_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)?$"
)

#: The rate fields an entry may declare, and the usage each one prices.
RATE_FIELDS: tuple[str, ...] = (
    "input_per_million",
    "output_per_million",
    "cached_input_per_million",
    "reasoning_per_million",
)


class PricingError(ValueError):
    """Raised when a pricing manifest is invalid, ambiguous, or unusable."""


def _require(data: Any, key: str, *, context: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise PricingError(f"{context} is missing required field: {key!r}")
    return data[key]


def _require_str(data: Any, key: str, *, context: str) -> str:
    value = _require(data, key, context=context)
    if not isinstance(value, str) or not value:
        raise PricingError(f"{context}.{key} must be a non-empty string, got {value!r}")
    return value


def _optional_str(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise PricingError(
            f"{context}.{key} must be a non-empty string or null, got {value!r}"
        )
    return value


def _require_bool(data: dict[str, Any], key: str, *, context: str) -> bool:
    value = _require(data, key, context=context)
    if not isinstance(value, bool):
        raise PricingError(f"{context}.{key} must be a boolean, got {value!r}")
    return value


def _optional_rate(data: dict[str, Any], key: str, *, context: str) -> float | None:
    """Read one per-million rate, refusing anything that is not a real price.

    Booleans, strings, NaN, infinities and negatives are all rejected rather
    than coerced. Zero is allowed and meaningful (a genuinely free tier), but
    it must be written as ``0``; a missing rate stays ``None`` and makes the
    metric that needs it unavailable instead of free.
    """
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PricingError(f"{context}.{key} must be a number or null, got {value!r}")
    try:
        numeric = float(value)
    except OverflowError as exc:
        # A JSON integer literal too large for a float arrives here as a
        # Python int. ``float()`` raises ``OverflowError``, an
        # ``ArithmeticError`` rather than a ``ValueError``, so no caller
        # catches it and it escapes as a traceback. Every one of these
        # inputs is a user-supplied file this module treats as untrusted,
        # so it becomes the same typed validation failure as any other
        # malformed value.
        raise PricingError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric):
        raise PricingError(
            f"{context}.{key} must be a finite number, got {numeric!r}; a "
            "non-finite rate would silently poison every cost derived from it"
        )
    if numeric < 0:
        raise PricingError(f"{context}.{key} must be >= 0, got {numeric!r}")
    return numeric


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing: sorted keys, no insignificant whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PricingEntry:
    """One provider/model price line, with its currency, date and source."""

    entry_id: str
    provider: str
    model_id: str
    currency: str
    effective_at: str
    source: str
    rates_are_illustrative: bool
    model_revision: str | None = None
    """When set, this entry only matches runs of that exact model revision."""
    input_per_million: float | None = None
    output_per_million: float | None = None
    cached_input_per_million: float | None = None
    reasoning_per_million: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not CURRENCY_PATTERN.match(self.currency):
            raise PricingError(
                f"pricing entry {self.entry_id!r} currency must be a three-letter "
                f"ISO 4217 code (e.g. 'USD'), got {self.currency!r}"
            )
        if not _EFFECTIVE_AT_PATTERN.match(self.effective_at):
            raise PricingError(
                f"pricing entry {self.entry_id!r} effective_at must be an ISO-8601 "
                f"date (YYYY-MM-DD) or UTC timestamp, got {self.effective_at!r}"
            )
        # Enforced here as well as in ``from_dict`` because this dataclass is
        # public: a caller constructing one directly bypasses the loader, and
        # a negative or non-finite rate would then silently poison every cost
        # derived from it instead of being refused at the boundary.
        for name in RATE_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PricingError(
                    f"pricing entry {self.entry_id!r} {name} must be a number "
                    f"or null, got {value!r}"
                )
            if not math.isfinite(value):
                raise PricingError(
                    f"pricing entry {self.entry_id!r} {name} must be a finite "
                    f"number, got {value!r}"
                )
            if value < 0:
                raise PricingError(
                    f"pricing entry {self.entry_id!r} {name} must be >= 0, "
                    f"got {value!r}"
                )
        if not any(getattr(self, name) is not None for name in RATE_FIELDS):
            raise PricingError(
                f"pricing entry {self.entry_id!r} declares no rate at all; an "
                "entry that prices nothing cannot be matched to anything"
            )

    def matches(
        self, *, provider: str | None, model_id: str, model_revision: str | None
    ) -> bool:
        """Exact match only. A revision-less entry matches any revision."""
        if provider is None or provider != self.provider:
            return False
        if model_id != self.model_id:
            return False
        if self.model_revision is None:
            return True
        return self.model_revision == model_revision

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "currency": self.currency,
            "effective_at": self.effective_at,
            "source": self.source,
            "rates_are_illustrative": self.rates_are_illustrative,
            "input_per_million": self.input_per_million,
            "output_per_million": self.output_per_million,
            "cached_input_per_million": self.cached_input_per_million,
            "reasoning_per_million": self.reasoning_per_million,
            "notes": self.notes,
        }

    @property
    def content_sha256(self) -> str:
        """Hash of this entry's exact content, for provenance in a report."""
        return hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> PricingEntry:
        if not isinstance(data, dict):
            raise PricingError("pricing entry must be a JSON object")
        entry_id = _require_str(data, "entry_id", context="pricing entry")
        context = f"pricing entry {entry_id!r}"
        return cls(
            entry_id=entry_id,
            provider=_require_str(data, "provider", context=context),
            model_id=_require_str(data, "model_id", context=context),
            model_revision=_optional_str(data, "model_revision", context=context),
            currency=_require_str(data, "currency", context=context),
            effective_at=_require_str(data, "effective_at", context=context),
            source=_require_str(data, "source", context=context),
            rates_are_illustrative=_require_bool(
                data, "rates_are_illustrative", context=context
            ),
            input_per_million=_optional_rate(
                data, "input_per_million", context=context
            ),
            output_per_million=_optional_rate(
                data, "output_per_million", context=context
            ),
            cached_input_per_million=_optional_rate(
                data, "cached_input_per_million", context=context
            ),
            reasoning_per_million=_optional_rate(
                data, "reasoning_per_million", context=context
            ),
            notes=_optional_str(data, "notes", context=context),
        )


@dataclass(frozen=True)
class PricingManifest:
    """A complete, validated set of price lines in one single currency."""

    schema_version: str
    currency: str
    entries: tuple[PricingEntry, ...]
    name: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PRICING_MANIFEST_SCHEMA_VERSION:
            raise PricingError(
                f"unsupported pricing manifest schema_version "
                f"{self.schema_version!r}, expected "
                f"{PRICING_MANIFEST_SCHEMA_VERSION!r}"
            )
        if not CURRENCY_PATTERN.match(self.currency):
            raise PricingError(
                "pricing manifest currency must be a three-letter ISO 4217 code "
                f"(e.g. 'USD'), got {self.currency!r}"
            )
        if not self.entries:
            raise PricingError("pricing manifest must declare at least one entry")

        seen_ids: set[str] = set()
        for entry in self.entries:
            if entry.entry_id in seen_ids:
                raise PricingError(
                    f"pricing manifest contains duplicate entry_id {entry.entry_id!r}"
                )
            seen_ids.add(entry.entry_id)
            if entry.currency != self.currency:
                raise PricingError(
                    f"pricing entry {entry.entry_id!r} is priced in "
                    f"{entry.currency!r} but the manifest declares "
                    f"{self.currency!r}; mixing currencies in one comparison "
                    "would produce a meaningless total"
                )

        # Two entries that can both match the same run make every cost drawn
        # from this manifest arbitrary, so the ambiguity is rejected at load
        # time rather than at the first lookup that happens to hit it.
        for index, entry in enumerate(self.entries):
            for other in self.entries[index + 1 :]:
                if entry.provider != other.provider or entry.model_id != other.model_id:
                    continue
                if (
                    entry.model_revision is None
                    or other.model_revision is None
                    or entry.model_revision == other.model_revision
                ):
                    raise PricingError(
                        f"pricing entries {entry.entry_id!r} and "
                        f"{other.entry_id!r} can both match provider "
                        f"{entry.provider!r} model {entry.model_id!r}; refusing "
                        "an ambiguous manifest. Give each entry a distinct "
                        "model_revision, or remove the duplicate."
                    )

    @property
    def rates_are_illustrative(self) -> bool:
        """True when any entry declares itself an example rather than a price."""
        return any(entry.rates_are_illustrative for entry in self.entries)

    def resolve(
        self, *, provider: str | None, model_id: str, model_revision: str | None
    ) -> PricingEntry | None:
        """Find the single entry for one system, or ``None`` if unpriced.

        Raises ``PricingError`` when more than one entry matches. Load-time
        validation already rejects manifests that *can* be ambiguous, so this
        is defense in depth for manifests built in code rather than loaded.
        """
        matches = [
            entry
            for entry in self.entries
            if entry.matches(
                provider=provider, model_id=model_id, model_revision=model_revision
            )
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise PricingError(
                f"provider {provider!r} model {model_id!r} revision "
                f"{model_revision!r} matches {len(matches)} pricing entries "
                f"({', '.join(sorted(entry.entry_id for entry in matches))}); "
                "refusing to pick one"
            )
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "currency": self.currency,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> PricingManifest:
        if not isinstance(data, dict):
            raise PricingError("pricing manifest must be a JSON/YAML object")
        context = "pricing manifest"
        entries_raw = _require(data, "entries", context=context)
        if not isinstance(entries_raw, list):
            raise PricingError(f"{context}.entries must be a list")
        # ``schema_version`` is required and matched exactly rather than
        # defaulted. Defaulting it would let a manifest written against a
        # future or past contract load silently under this build's reading of
        # the field names, which is precisely the mistake versioning exists
        # to prevent. Every rate in the file feeds a published monetary
        # figure, so an unversioned file is not a file to guess about.
        declared = _require(data, "schema_version", context=context)
        if not isinstance(declared, str):
            raise PricingError(
                f"{context}.schema_version must be a string, got {declared!r}"
            )
        return cls(
            schema_version=declared,
            name=_optional_str(data, "name", context=context),
            description=_optional_str(data, "description", context=context),
            currency=_require_str(data, "currency", context=context),
            entries=tuple(PricingEntry.from_dict(item) for item in entries_raw),
        )

    @classmethod
    def from_json(cls, payload: str) -> PricingManifest:
        try:
            data = json.loads(payload)
        # ``json`` raises past its own limits with exceptions that are
        # not ``JSONDecodeError``: an integer literal over the
        # interpreter's digit cap raises a plain ``ValueError``, and deep
        # nesting raises ``RecursionError``. Neither is caught by any
        # caller, so both used to escape as a traceback from a merely
        # malformed file.
        except (ValueError, RecursionError) as exc:
            raise PricingError(f"invalid JSON for pricing manifest: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> PricingManifest:
        """Load a manifest from ``.json`` or ``.yaml``/``.yml``.

        Mirrors ``TunePolicy.from_file``'s extension dispatch and its explicit
        (never silent) failure on an unsupported extension or a missing
        PyYAML dependency.
        """
        manifest_path = Path(path)
        text = manifest_path.read_text(encoding="utf-8")
        suffix = manifest_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise PricingError(
                    "YAML pricing manifest requires PyYAML to be installed "
                    "(`uv add pyyaml`); use a .json manifest instead if it is "
                    "unavailable"
                ) from exc
            try:
                data = yaml.safe_load(text)
            # ``yaml.YAMLError`` subclasses ``Exception``, not
            # ``ValueError``, so neither this loader nor the CLI
            # caught it and a merely malformed YAML file escaped as a
            # traceback. YAML is an advertised input format for this
            # flag, and it is far easier to malform than JSON.
            except (yaml.YAMLError, ValueError, RecursionError) as exc:
                raise PricingError(f"invalid YAML in {manifest_path}: {exc}") from exc
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except (ValueError, RecursionError) as exc:
                raise PricingError(f"invalid JSON in {manifest_path}: {exc}") from exc
        else:
            raise PricingError(
                f"unsupported pricing manifest extension {suffix!r} "
                "(use .json or .yaml)"
            )
        if not isinstance(data, dict):
            raise PricingError(
                f"pricing manifest in {manifest_path} must be a JSON/YAML object"
            )
        return cls.from_dict(data)
