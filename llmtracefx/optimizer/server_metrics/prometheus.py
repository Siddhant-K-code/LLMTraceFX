"""Strict parser for the Prometheus text exposition format.

Handles the classic ``text/plain; version=0.0.4`` format that vLLM, SGLang,
llama.cpp and TGI serve, plus the parts of OpenMetrics text a server may
send instead (``# UNIT``, ``# EOF``, exemplars). Anything that does not
parse is an error rather than a skipped line: a silently dropped sample
would turn into a silently wrong delta.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_EXPOSITION_SAMPLES = 250_000

METRIC_TYPES = frozenset(
    {"counter", "gauge", "histogram", "summary", "untyped", "unknown", "info"}
)
_SUFFIXES: dict[str, tuple[str, ...]] = {
    "counter": ("_total", "_created"),
    "histogram": ("_bucket", "_sum", "_count", "_created"),
    "summary": ("_sum", "_count", "_created"),
    "info": ("_info",),
}
_METRIC_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_NUMBER = re.compile(
    r"[+-]?(?:[Ii]nf(?:inity)?|NaN|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
)


class ExpositionParseError(ValueError):
    """Raised when a metrics body is not valid Prometheus text."""


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


@dataclass(frozen=True)
class MetricFamily:
    """One metric family.

    ``name`` is canonical: a counter is keyed without its ``_total`` suffix
    whichever way the server declared it, so ``vllm:prompt_tokens`` and
    ``llamacpp:prompt_tokens`` look the same to callers.
    """

    name: str
    type: str
    help: str | None
    samples: tuple[Sample, ...]


def canonical_family_name(name: str, metric_type: str) -> str:
    if metric_type == "counter" and name.endswith("_total"):
        return name[: -len("_total")]
    return name


def parse_number(token: str) -> float:
    if not _NUMBER.fullmatch(token):
        raise ExpositionParseError(f"invalid sample value {token!r}")
    return float(token)


def _skip_blanks(text: str, index: int) -> int:
    while index < len(text) and text[index] in " \t":
        index += 1
    return index


def _parse_label_value(text: str, index: int, line_number: int) -> tuple[str, int]:
    """Parse an escaped label value whose opening quote precedes ``index``."""
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(chars), index + 1
        if char == "\\":
            escaped = text[index + 1 : index + 2]
            if escaped == "n":
                chars.append("\n")
            elif escaped in ('"', "\\"):
                chars.append(escaped)
            else:
                raise ExpositionParseError(
                    f"line {line_number}: invalid escape in label value"
                )
            index += 2
            continue
        chars.append(char)
        index += 1
    raise ExpositionParseError(f"line {line_number}: unterminated label value")


def _parse_labels(
    text: str, start: int, line_number: int
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Parse ``{a="b",...}`` starting at ``text[start] == '{'``."""
    labels: dict[str, str] = {}
    index = start + 1
    while True:
        index = _skip_blanks(text, index)
        if text[index : index + 1] == "}":
            return tuple(sorted(labels.items())), index + 1
        match = _LABEL_NAME.match(text, index)
        if match is None:
            raise ExpositionParseError(f"line {line_number}: invalid label name")
        name = match.group(0)
        index = _skip_blanks(text, match.end())
        if text[index : index + 2] != '="':
            raise ExpositionParseError(
                f"line {line_number}: expected '=\"' after label {name!r}"
            )
        value, index = _parse_label_value(text, index + 2, line_number)
        if name in labels:
            raise ExpositionParseError(f"line {line_number}: duplicate label {name!r}")
        labels[name] = value
        index = _skip_blanks(text, index)
        separator = text[index : index + 1]
        if separator == ",":
            index += 1
        elif separator != "}":
            raise ExpositionParseError(f"line {line_number}: expected ',' or '}}'")


def _parse_sample(line: str, line_number: int) -> Sample:
    match = _METRIC_NAME.match(line)
    if match is None:
        raise ExpositionParseError(f"line {line_number}: invalid metric name")
    name = match.group(0)
    index = match.end()
    labels: tuple[tuple[str, str], ...] = ()
    if line[index : index + 1] == "{":
        labels, index = _parse_labels(line, index, line_number)
    rest = line[index:]
    if rest and rest[0] not in " \t":
        raise ExpositionParseError(f"line {line_number}: unexpected character")
    # An OpenMetrics exemplar follows the value as `# {...} v [ts]`.
    if " # " in rest:
        rest = rest.split(" # ", 1)[0]
    tokens = rest.split()
    if not tokens or len(tokens) > 2:
        raise ExpositionParseError(
            f"line {line_number}: expected a value and optional timestamp"
        )
    value = parse_number(tokens[0])
    if len(tokens) == 2:
        parse_number(tokens[1])
    return Sample(name=name, labels=labels, value=value)


def _family_for(
    sample_name: str, declared: dict[str, tuple[str, str]]
) -> tuple[str, str] | None:
    """Return ``(canonical_name, type)`` for a sample, if declared."""
    if sample_name in declared:
        return declared[sample_name]
    for metric_type, suffixes in _SUFFIXES.items():
        for suffix in suffixes:
            if not sample_name.endswith(suffix):
                continue
            base = sample_name[: -len(suffix)]
            for candidate in (base, f"{base}_total"):
                entry = declared.get(candidate)
                if entry is not None and entry[1] == metric_type:
                    return entry
    return None


def parse_exposition(
    text: str, *, max_samples: int = MAX_EXPOSITION_SAMPLES
) -> dict[str, MetricFamily]:
    """Parse an exposition body into families keyed by canonical name."""
    # Declared name, as written after ``# TYPE``, to (canonical, type).
    declared: dict[str, tuple[str, str]] = {}
    helps: dict[str, str] = {}
    types: dict[str, str] = {}
    samples: dict[str, list[Sample]] = {}
    seen_series: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    total = 0
    eof = False

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if eof:
            raise ExpositionParseError(f"line {line_number}: content after # EOF")
        if line.startswith("#"):
            parts = line[1:].strip().split(None, 2)
            if parts == ["EOF"]:
                eof = True
                continue
            if len(parts) < 2 or parts[0] not in ("HELP", "TYPE", "UNIT"):
                continue
            keyword, declared_name = parts[0], parts[1]
            if not _METRIC_NAME.fullmatch(declared_name):
                raise ExpositionParseError(
                    f"line {line_number}: invalid metric name in # {keyword}"
                )
            argument = parts[2] if len(parts) == 3 else ""
            if keyword == "TYPE":
                metric_type = argument.strip()
                if metric_type not in METRIC_TYPES:
                    raise ExpositionParseError(
                        f"line {line_number}: unknown metric type {metric_type!r}"
                    )
                canonical = canonical_family_name(declared_name, metric_type)
                if declared_name in declared or canonical in types:
                    raise ExpositionParseError(
                        f"line {line_number}: duplicate # TYPE for {declared_name}"
                    )
                declared[declared_name] = (canonical, metric_type)
                types[canonical] = metric_type
                if declared_name in helps and declared_name != canonical:
                    helps[canonical] = helps.pop(declared_name)
            elif keyword == "HELP":
                entry = declared.get(declared_name)
                helps[entry[0] if entry is not None else declared_name] = argument
            continue

        sample = _parse_sample(line, line_number)
        total += 1
        if total > max_samples:
            raise ExpositionParseError(
                f"exposition exceeds the {max_samples}-sample limit"
            )
        series = (sample.name, sample.labels)
        if series in seen_series:
            raise ExpositionParseError(
                f"line {line_number}: duplicate series {sample.name}"
            )
        seen_series.add(series)
        family = _family_for(sample.name, declared)
        if family is None:
            canonical = sample.name
            types.setdefault(canonical, "untyped")
        else:
            canonical = family[0]
        samples.setdefault(canonical, []).append(sample)

    families: dict[str, MetricFamily] = {}
    for canonical in sorted(types):
        families[canonical] = MetricFamily(
            name=canonical,
            type=types[canonical],
            help=helps.get(canonical),
            samples=tuple(
                sorted(
                    samples.get(canonical, []),
                    key=lambda item: (item.name, item.labels),
                )
            ),
        )
    return families


def encode_value(value: float) -> float | str:
    """JSON-safe value: non-finite floats become their exposition spelling."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def decode_value(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("sample value must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value in ("NaN", "+Inf", "-Inf"):
        return float(value)
    raise ValueError(f"invalid sample value {value!r}")
