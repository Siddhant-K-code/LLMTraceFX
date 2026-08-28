"""Workload schema: categories, context tiers, and the ``Workload`` record.

Every workload is versioned and deterministic: the same ``workload_id`` +
``version`` always produces the same base prompt and evaluator
configuration, so matrix generation and prompt hashing are reproducible
across runs and machines.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

WORKLOAD_SCHEMA_VERSION = "1"


class WorkloadSchemaError(ValueError):
    """Raised when a workload definition is invalid."""


class WorkloadCategory(str, Enum):
    """The three workload categories this matrix covers."""

    CODE_COMPLETION = "code_completion"
    STRUCTURED_JSON = "structured_json"
    PROSE_REASONING = "prose_reasoning"


class ContextTier(str, Enum):
    """Target context sizes the matrix generates prompts for."""

    TIER_2K = "2k"
    TIER_8K = "8k"
    TIER_16K = "16k"


#: Target context length in tokens for each tier. These are *targets* the
#: materializer pads toward using an explicit, documented approximation
#: (see ``materialize.py``) -- not measured token counts. The actual
#: token count for a real run is measured by the runtime's own tokenizer
#: at collection time (``ExperimentRecord.tokens``).
CONTEXT_TIER_TARGET_TOKENS: dict[ContextTier, int] = {
    ContextTier.TIER_2K: 2048,
    ContextTier.TIER_8K: 8192,
    ContextTier.TIER_16K: 16384,
}


@dataclass(frozen=True)
class CodeCompletionSpec:
    """A small function stub plus a deterministic, executable test."""

    function_stub: str
    test_code: str
    """Python source appended after the candidate completion; must exit
    0 (via ``assert``) for the workload to pass."""
    entry_point: str
    """Name of the function the test code calls."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeCompletionSpec:
        try:
            return cls(
                function_stub=data["function_stub"],
                test_code=data["test_code"],
                entry_point=data["entry_point"],
            )
        except KeyError as exc:
            raise WorkloadSchemaError(
                f"CodeCompletionSpec is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class StructuredJSONSpec:
    """Required top-level fields and their expected JSON types."""

    required_fields: tuple[str, ...]
    field_types: dict[str, str]
    """Maps field name to one of 'str', 'int', 'float', 'bool', 'list',
    'dict' -- checked with a fixed, explicit type table, never ``eval``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_fields": list(self.required_fields),
            "field_types": dict(self.field_types),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredJSONSpec:
        try:
            return cls(
                required_fields=tuple(data["required_fields"]),
                field_types=dict(data["field_types"]),
            )
        except KeyError as exc:
            raise WorkloadSchemaError(
                f"StructuredJSONSpec is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class ProseReasoningSpec:
    """A deterministic expected-answer regex, matched case-insensitively."""

    expected_answer_pattern: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProseReasoningSpec:
        try:
            return cls(expected_answer_pattern=data["expected_answer_pattern"])
        except KeyError as exc:
            raise WorkloadSchemaError(
                f"ProseReasoningSpec is missing required field: {exc}"
            ) from exc


WorkloadSpec = CodeCompletionSpec | StructuredJSONSpec | ProseReasoningSpec

_SPEC_TYPES_BY_CATEGORY: dict[WorkloadCategory, type[WorkloadSpec]] = {
    WorkloadCategory.CODE_COMPLETION: CodeCompletionSpec,
    WorkloadCategory.STRUCTURED_JSON: StructuredJSONSpec,
    WorkloadCategory.PROSE_REASONING: ProseReasoningSpec,
}


class PaddingPlacement(str, Enum):
    """Where context-padding filler must be inserted in a materialized prompt."""

    APPEND_AFTER_BASE_PROMPT = "append_after_base_prompt"
    """Filler goes after the full (already self-contained) base prompt.
    Valid for workloads whose base prompt has no dangling continuation
    point, e.g. structured JSON extraction or prose reasoning."""

    BEFORE_CONTINUATION_STUB = "before_continuation_stub"
    """Filler is inserted between the base prompt's instructions and its
    fixed ``continuation_stub``, so the stub remains the very last text
    the model sees. Required for code completion, where the model must
    continue writing code immediately after an open function
    signature/docstring; appending filler after that stub would corrupt
    the completion point."""


@dataclass(frozen=True)
class Workload:
    """One versioned, deterministic workload definition."""

    workload_id: str
    version: str
    category: WorkloadCategory
    title: str
    base_prompt: str
    spec: WorkloadSpec
    continuation_stub: str = ""
    """When non-empty, the exact trailing content of ``base_prompt`` that
    must remain the very last text the model sees (e.g. an open function
    signature/docstring for code completion). Context padding is inserted
    *before* this stub rather than appended after the full base prompt,
    so the model still continues immediately from the stub. Empty for
    workloads (structured JSON, prose reasoning) whose ``base_prompt`` is
    already a complete, self-contained instruction with no dangling
    continuation point -- for those, padding appended after the full
    base prompt does not change the task's meaning."""

    def __post_init__(self) -> None:
        if not self.workload_id:
            raise WorkloadSchemaError("workload_id must be non-empty")
        if not self.version:
            raise WorkloadSchemaError("version must be non-empty")
        if not self.base_prompt:
            raise WorkloadSchemaError("base_prompt must be non-empty")
        expected_type = _SPEC_TYPES_BY_CATEGORY[self.category]
        if not isinstance(self.spec, expected_type):
            raise WorkloadSchemaError(
                f"workload '{self.workload_id}' has category "
                f"{self.category.value} but spec type "
                f"{type(self.spec).__name__}, expected {expected_type.__name__}"
            )
        if self.continuation_stub and not self.base_prompt.endswith(
            self.continuation_stub
        ):
            raise WorkloadSchemaError(
                f"workload '{self.workload_id}' has a continuation_stub that "
                "is not a suffix of base_prompt; padding placement requires "
                "the stub to be the exact trailing content of the prompt"
            )

    @property
    def padding_placement(self) -> PaddingPlacement:
        """Where context-padding filler must be inserted for this workload."""
        return (
            PaddingPlacement.BEFORE_CONTINUATION_STUB
            if self.continuation_stub
            else PaddingPlacement.APPEND_AFTER_BASE_PROMPT
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "workload_id": self.workload_id,
            "version": self.version,
            "category": self.category.value,
            "title": self.title,
            "base_prompt": self.base_prompt,
            "spec": self.spec.to_dict(),
            "continuation_stub": self.continuation_stub,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Workload:
        try:
            category = WorkloadCategory(data["category"])
            spec_type = _SPEC_TYPES_BY_CATEGORY[category]
            return cls(
                workload_id=data["workload_id"],
                version=data["version"],
                category=category,
                title=data["title"],
                base_prompt=data["base_prompt"],
                spec=spec_type.from_dict(data["spec"]),
                continuation_stub=data.get("continuation_stub", ""),
            )
        except KeyError as exc:
            raise WorkloadSchemaError(
                f"Workload is missing required field: {exc}"
            ) from exc
