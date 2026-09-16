"""Canonical Stage-1 quadruple contract (``canonical-quad-json/v1``).

This module is the single authority for adapting source annotations, validating
model JSON, and producing the byte-stable wire representation used by Stage 1.
The strict parser intentionally does not accept legacy pipe-delimited triples or
quadruples.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


SCHEMA_VERSION = "canonical-quad-json/v1"
QUAD_KEYS = ("target", "argument", "targeted_group", "hateful")
GROUP_ORDER = ("Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate")
HATEFUL_LABELS = ("hate", "non-hate")

_GROUP_INDEX = {label: index for index, label in enumerate(GROUP_ORDER)}
_GROUP_ALIASES = {
    "racism": "Racism",
    "region": "Region",
    "lgbtq": "LGBTQ",
    "sexism": "Sexism",
    "gender": "Sexism",
    "others": "others",
    "other": "others",
    "non-hate": "non-hate",
    "non_hate": "non-hate",
    "non hate": "non-hate",
    "nonhate": "non-hate",
}
_HATEFUL_ALIASES = {
    "hate": "hate",
    "non-hate": "non-hate",
    "non_hate": "non-hate",
    "non hate": "non-hate",
    "nonhate": "non-hate",
}


@dataclass(frozen=True)
class Quadruple:
    """Normalized in-memory representation of one annotation."""

    target: str | None
    argument: str | None
    targeted_group: tuple[str, ...]
    hateful: Literal["hate", "non-hate"]


@dataclass(frozen=True)
class ParseIssue:
    """Machine-readable validation or recovery diagnostic."""

    code: str
    message: str
    path: str = "$"

    def __str__(self) -> str:
        return f"{self.path}: {self.code}: {self.message}"


@dataclass(frozen=True)
class ParseResult:
    """Result of strict validation and, optionally, diagnostic recovery."""

    raw: str
    quadruples: list[Quadruple]
    syntax_valid: bool
    schema_valid: bool
    strict_format_valid: bool
    recoverable_parse_valid: bool
    canonical_wire_equal: bool
    canonical_text: str | None
    errors: list[ParseIssue]
    warnings: list[ParseIssue]

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.errors)

    @property
    def warning_codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.warnings)


class QuadrupleValidationError(ValueError):
    """Raised when source or programmatic input violates the contract."""

    def __init__(self, issues: Sequence[ParseIssue]):
        self.issues = tuple(issues)
        self.errors = self.issues
        message = "; ".join(str(issue) for issue in self.issues)
        super().__init__(message or "invalid quadruple")


class _JSONObjectPairs(list[tuple[str, Any]]):
    """Marker type used to retain duplicate JSON object keys."""


def _issue(code: str, message: str, path: str = "$") -> ParseIssue:
    return ParseIssue(code=code, message=message, path=path)


def _normalize_text(value: Any, field: str, path: str) -> tuple[str | None, list[ParseIssue]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return None, [_issue("field_type", f"{field} must be a string or null", path)]
    normalized = unicodedata.normalize("NFC", value).strip()
    if not normalized:
        return None, [_issue("empty_text", f"{field} cannot be empty", path)]
    if normalized == "NULL":
        return None, [_issue("legacy_null_sentinel", f"{field} must use JSON null", path)]
    return normalized, []


def _normalize_groups(
    value: Any,
    path: str,
    *,
    allow_aliases: bool,
    warnings: list[ParseIssue] | None = None,
) -> tuple[tuple[str, ...] | None, list[ParseIssue]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None, [_issue("field_type", "targeted_group must be a JSON array of strings", path)]
    if not value:
        return None, [_issue("empty_group", "targeted_group cannot be empty", path)]

    labels: list[str] = []
    errors: list[ParseIssue] = []
    for index, label in enumerate(value):
        item_path = f"{path}[{index}]"
        if not isinstance(label, str):
            errors.append(_issue("field_type", "group labels must be strings", item_path))
            continue
        normalized = unicodedata.normalize("NFC", label).strip()
        canonical = normalized
        if allow_aliases:
            canonical = _GROUP_ALIASES.get(normalized.lower(), normalized)
            if canonical != normalized and warnings is not None:
                warnings.append(
                    _issue("alias_normalized", f"normalized group alias {normalized!r} to {canonical!r}", item_path)
                )
        if canonical not in _GROUP_INDEX:
            errors.append(_issue("unknown_group", f"unknown group label {normalized!r}", item_path))
            continue
        labels.append(canonical)

    if len(labels) != len(set(labels)):
        errors.append(_issue("duplicate_group", "targeted_group contains duplicate labels", path))
    if "non-hate" in labels and len(labels) > 1:
        errors.append(_issue("mixed_non_hate_group", "non-hate cannot coexist with another group", path))
    if errors:
        return None, errors
    return tuple(sorted(labels, key=_GROUP_INDEX.__getitem__)), []


def _normalize_hateful(
    value: Any,
    path: str,
    *,
    allow_aliases: bool,
    warnings: list[ParseIssue] | None = None,
) -> tuple[Literal["hate", "non-hate"] | None, list[ParseIssue]]:
    if allow_aliases and value is None:
        return None, [
            _issue(
                "unknown_annotation",
                "missing source hateful annotation must be adjudicated",
                path,
            )
        ]
    if not isinstance(value, str):
        return None, [_issue("field_type", "hateful must be a string", path)]
    normalized = unicodedata.normalize("NFC", value).strip()
    if normalized == "NULL":
        if not allow_aliases:
            return None, [
                _issue(
                    "legacy_null_sentinel",
                    "hateful must be hate or non-hate; string 'NULL' is forbidden",
                    path,
                )
            ]
        return None, [
            _issue(
                "unknown_annotation",
                "hateful='NULL' is an unknown annotation and must be adjudicated",
                path,
            )
        ]
    canonical = normalized
    if allow_aliases:
        canonical = _HATEFUL_ALIASES.get(normalized.lower(), normalized)
        if canonical != normalized and warnings is not None:
            warnings.append(
                _issue("alias_normalized", f"normalized hateful alias {normalized!r} to {canonical!r}", path)
            )
    if canonical not in HATEFUL_LABELS:
        return None, [_issue("invalid_hateful", f"invalid hateful label {normalized!r}", path)]
    return canonical, []  # type: ignore[return-value]


def _strict_mapping(
    value: Mapping[str, Any],
    path: str,
    *,
    allow_aliases: bool = False,
    warnings: list[ParseIssue] | None = None,
) -> tuple[Quadruple | None, list[ParseIssue]]:
    errors: list[ParseIssue] = []
    actual_keys = set(value)
    expected_keys = set(QUAD_KEYS)
    for key in QUAD_KEYS:
        if key not in actual_keys:
            errors.append(_issue("missing_key", f"missing required key {key!r}", path))
    for key in sorted(actual_keys - expected_keys):
        errors.append(_issue("unknown_key", f"unknown key {key!r}", path))
    if errors:
        return None, errors

    target_value = value["target"]
    argument_value = value["argument"]
    if allow_aliases:
        target_value = _adapt_source_nullable(target_value, "target", f"{path}.target", warnings)
        argument_value = _adapt_source_nullable(argument_value, "argument", f"{path}.argument", warnings)

    target, field_errors = _normalize_text(target_value, "target", f"{path}.target")
    errors.extend(field_errors)
    argument, field_errors = _normalize_text(argument_value, "argument", f"{path}.argument")
    errors.extend(field_errors)
    groups, field_errors = _normalize_groups(
        value["targeted_group"],
        f"{path}.targeted_group",
        allow_aliases=allow_aliases,
        warnings=warnings,
    )
    errors.extend(field_errors)
    hateful, field_errors = _normalize_hateful(
        value["hateful"],
        f"{path}.hateful",
        allow_aliases=allow_aliases,
        warnings=warnings,
    )
    errors.extend(field_errors)
    if errors or groups is None or hateful is None:
        return None, errors
    return Quadruple(target=target, argument=argument, targeted_group=groups, hateful=hateful), []


def _adapt_source_nullable(
    value: Any,
    field: str,
    path: str,
    warnings: list[ParseIssue] | None,
) -> Any:
    if isinstance(value, str) and value.strip() == "NULL":
        if warnings is not None:
            warnings.append(_issue("legacy_null_normalized", f"normalized {field}='NULL' to null", path))
        return None
    return value


def _adapt_source_group(value: Any, path: str, warnings: list[ParseIssue] | None) -> Any:
    if isinstance(value, str):
        if warnings is not None:
            warnings.append(_issue("source_group_split", "split source group string into an array", path))
        return [part.strip() for part in re.split(r"[,，]", value) if part.strip()]
    return value


def adapt_source_quad(value: Mapping[str, Any]) -> Quadruple:
    """Adapt one legacy/source annotation into the canonical in-memory form.

    The adapter is deliberately one-way.  It accepts legacy ``NULL`` only for
    target/argument, a comma-separated group string, and a small documented
    label-alias set.  Unknown hateful annotations raise instead of being inferred.
    """

    if not isinstance(value, Mapping):
        raise QuadrupleValidationError([_issue("item_type", "source quadruple must be a mapping")])
    adapted = {
        "target": value.get("target"),
        "argument": value.get("argument"),
        "targeted_group": _adapt_source_group(value.get("targeted_group"), "$.targeted_group", None),
        "hateful": value.get("hateful"),
    }
    adapted["target"] = _adapt_source_nullable(adapted["target"], "target", "$.target", None)
    adapted["argument"] = _adapt_source_nullable(adapted["argument"], "argument", "$.argument", None)
    quadruple, errors = _strict_mapping(adapted, "$", allow_aliases=True)
    if errors or quadruple is None:
        raise QuadrupleValidationError(errors)
    return quadruple


def _quadruple_mapping(value: Quadruple | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Quadruple):
        return {
            "target": value.target,
            "argument": value.argument,
            "targeted_group": value.targeted_group,
            "hateful": value.hateful,
        }
    if isinstance(value, Mapping):
        return value
    raise QuadrupleValidationError([_issue("item_type", "quadruple must be a Quadruple or mapping")])


def canonicalize_quadruple(value: Quadruple | Mapping[str, Any]) -> Quadruple:
    """Validate and normalize one canonical (non-source) quadruple."""

    mapping = _quadruple_mapping(value)
    quadruple, errors = _strict_mapping(mapping, "$")
    if errors or quadruple is None:
        raise QuadrupleValidationError(errors)
    return quadruple


def canonicalize_quadruples(values: Sequence[Quadruple | Mapping[str, Any]]) -> list[Quadruple]:
    """Validate canonical programmatic input and return normalized quadruples."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise QuadrupleValidationError([_issue("top_level_type", "quadruples must be a sequence")])
    quadruples: list[Quadruple] = []
    errors: list[ParseIssue] = []
    for index, value in enumerate(values):
        try:
            mapping = _quadruple_mapping(value)
        except QuadrupleValidationError as exc:
            errors.extend(
                ParseIssue(issue.code, issue.message, f"$[{index}]") for issue in exc.issues
            )
            continue
        quadruple, item_errors = _strict_mapping(mapping, f"$[{index}]")
        errors.extend(item_errors)
        if quadruple is not None:
            quadruples.append(quadruple)
    if errors:
        raise QuadrupleValidationError(errors)
    return quadruples


def _as_json_object(quadruple: Quadruple) -> dict[str, Any]:
    return {
        "target": quadruple.target,
        "argument": quadruple.argument,
        "targeted_group": list(quadruple.targeted_group),
        "hateful": quadruple.hateful,
    }


def serialize_quadruples(values: Sequence[Quadruple | Mapping[str, Any]]) -> str:
    """Serialize canonical quadruples to compact deterministic JSON."""

    quadruples = canonicalize_quadruples(values)
    return json.dumps(
        [_as_json_object(quadruple) for quadruple in quadruples],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


ValueSpanMap = dict[tuple[int, str], tuple[int, int]]


def _json_literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def serialize_with_spans(
    values: Sequence[Quadruple | Mapping[str, Any]],
) -> tuple[str, ValueSpanMap]:
    """Serialize while recording half-open spans of every complete value literal."""

    quadruples = canonicalize_quadruples(values)
    chunks: list[str] = ["["]
    length = 1
    spans: ValueSpanMap = {}
    for tuple_index, quadruple in enumerate(quadruples):
        if tuple_index:
            chunks.append(",")
            length += 1
        chunks.append("{")
        length += 1
        obj = _as_json_object(quadruple)
        for field_index, field in enumerate(QUAD_KEYS):
            if field_index:
                chunks.append(",")
                length += 1
            prefix = _json_literal(field) + ":"
            chunks.append(prefix)
            length += len(prefix)
            literal = _json_literal(obj[field])
            start = length
            chunks.append(literal)
            length += len(literal)
            spans[(tuple_index, field)] = (start, length)
        chunks.append("}")
        length += 1
    chunks.append("]")
    text = "".join(chunks)
    return text, spans


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_json(raw: str) -> tuple[Any | None, list[ParseIssue]]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_JSONObjectPairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, [_issue("invalid_json", str(exc))]
    return value, []


def _pairs_to_plain(value: Any, path: str = "$") -> tuple[Any, list[ParseIssue]]:
    if isinstance(value, _JSONObjectPairs):
        result: dict[str, Any] = {}
        errors: list[ParseIssue] = []
        for key, item in value:
            if key in result:
                errors.append(_issue("duplicate_key", f"duplicate object key {key!r}", path))
            converted, child_errors = _pairs_to_plain(item, f"{path}.{key}")
            errors.extend(child_errors)
            if key not in result:
                result[key] = converted
        return result, errors
    if isinstance(value, list):
        converted_items: list[Any] = []
        errors: list[ParseIssue] = []
        for index, item in enumerate(value):
            converted, child_errors = _pairs_to_plain(item, f"{path}[{index}]")
            converted_items.append(converted)
            errors.extend(child_errors)
        return converted_items, errors
    return value, []


def _semantic_warnings(quadruples: Sequence[Quadruple]) -> list[ParseIssue]:
    warnings: list[ParseIssue] = []
    for index, quadruple in enumerate(quadruples):
        groups_are_non_hate = quadruple.targeted_group == ("non-hate",)
        labels_disagree = (quadruple.hateful == "hate" and groups_are_non_hate) or (
            quadruple.hateful == "non-hate" and not groups_are_non_hate
        )
        if labels_disagree:
            warnings.append(
                _issue(
                    "group_hateful_mismatch",
                    "targeted_group and hateful disagree; values were preserved",
                    f"$[{index}]",
                )
            )
    return warnings


def _validate_loaded(
    loaded: Any,
    *,
    allow_aliases: bool,
) -> tuple[list[Quadruple], list[ParseIssue], list[ParseIssue]]:
    plain, errors = _pairs_to_plain(loaded)
    warnings: list[ParseIssue] = []
    if not isinstance(plain, list):
        errors.append(_issue("top_level_type", "top-level JSON value must be an array"))
        return [], errors, warnings

    quadruples: list[Quadruple] = []
    for index, item in enumerate(plain):
        path = f"$[{index}]"
        if not isinstance(item, Mapping):
            errors.append(_issue("item_type", "each array item must be an object", path))
            continue
        quadruple, item_errors = _strict_mapping(
            item,
            path,
            allow_aliases=allow_aliases,
            warnings=warnings,
        )
        errors.extend(item_errors)
        if quadruple is not None:
            quadruples.append(quadruple)
    if errors:
        return [], errors, warnings
    warnings.extend(_semantic_warnings(quadruples))
    return quadruples, [], warnings


def _strict_parse(raw: str) -> ParseResult:
    loaded, syntax_errors = _load_json(raw)
    if syntax_errors:
        return ParseResult(
            raw=raw,
            quadruples=[],
            syntax_valid=False,
            schema_valid=False,
            strict_format_valid=False,
            recoverable_parse_valid=False,
            canonical_wire_equal=False,
            canonical_text=None,
            errors=syntax_errors,
            warnings=[],
        )
    quadruples, schema_errors, warnings = _validate_loaded(loaded, allow_aliases=False)
    if schema_errors:
        return ParseResult(
            raw=raw,
            quadruples=[],
            syntax_valid=True,
            schema_valid=False,
            strict_format_valid=False,
            recoverable_parse_valid=False,
            canonical_wire_equal=False,
            canonical_text=None,
            errors=schema_errors,
            warnings=warnings,
        )
    canonical_text = serialize_quadruples(quadruples)
    return ParseResult(
        raw=raw,
        quadruples=quadruples,
        syntax_valid=True,
        schema_valid=True,
        strict_format_valid=True,
        recoverable_parse_valid=True,
        canonical_wire_equal=raw == canonical_text,
        canonical_text=canonical_text,
        errors=[],
        warnings=warnings,
    )


def _extract_first_json_array(text: str) -> str | None:
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            if start is None:
                start = index
            depth += 1
        elif char == "]" and start is not None:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _recovery_candidate(raw: str) -> tuple[str | None, list[ParseIssue]]:
    candidate = raw.strip()
    warnings: list[ParseIssue] = []
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        candidate = fence.group(1).strip()
        warnings.append(_issue("recovered_json_fence", "removed a Markdown JSON fence"))
    extracted = _extract_first_json_array(candidate)
    if extracted is None:
        return None, warnings
    if extracted != candidate.strip():
        warnings.append(_issue("recovered_array_extraction", "extracted the first complete JSON array"))
    return extracted, warnings


def parse_quadruples(raw: str, *, mode: Literal["strict", "recover"] = "strict") -> ParseResult:
    """Parse canonical Stage-1 JSON.

    ``strict`` validates the untouched response. ``recover`` additionally
    performs diagnostic-only fence/array extraction and source-alias
    normalization. Recovery never upgrades the strict validity flags.
    """

    if mode not in {"strict", "recover"}:
        raise ValueError("mode must be 'strict' or 'recover'")
    if not isinstance(raw, str):
        raw = str(raw)
    strict_result = _strict_parse(raw)
    if mode == "strict" or strict_result.strict_format_valid:
        return strict_result

    candidate, recovery_warnings = _recovery_candidate(raw)
    if candidate is None:
        return ParseResult(
            raw=raw,
            quadruples=[],
            syntax_valid=strict_result.syntax_valid,
            schema_valid=strict_result.schema_valid,
            strict_format_valid=False,
            recoverable_parse_valid=False,
            canonical_wire_equal=False,
            canonical_text=None,
            errors=strict_result.errors,
            warnings=recovery_warnings,
        )
    loaded, recovery_syntax_errors = _load_json(candidate)
    if recovery_syntax_errors:
        return ParseResult(
            raw=raw,
            quadruples=[],
            syntax_valid=strict_result.syntax_valid,
            schema_valid=strict_result.schema_valid,
            strict_format_valid=False,
            recoverable_parse_valid=False,
            canonical_wire_equal=False,
            canonical_text=None,
            errors=[*strict_result.errors, *recovery_syntax_errors],
            warnings=recovery_warnings,
        )
    quadruples, recovery_errors, alias_warnings = _validate_loaded(loaded, allow_aliases=True)
    if recovery_errors:
        return ParseResult(
            raw=raw,
            quadruples=[],
            syntax_valid=strict_result.syntax_valid,
            schema_valid=strict_result.schema_valid,
            strict_format_valid=False,
            recoverable_parse_valid=False,
            canonical_wire_equal=False,
            canonical_text=None,
            errors=[*strict_result.errors, *recovery_errors],
            warnings=[*recovery_warnings, *alias_warnings],
        )
    return ParseResult(
        raw=raw,
        quadruples=quadruples,
        syntax_valid=strict_result.syntax_valid,
        schema_valid=strict_result.schema_valid,
        strict_format_valid=False,
        recoverable_parse_valid=True,
        canonical_wire_equal=False,
        canonical_text=serialize_quadruples(quadruples),
        errors=strict_result.errors,
        warnings=[*recovery_warnings, *alias_warnings],
    )


__all__ = [
    "SCHEMA_VERSION",
    "QUAD_KEYS",
    "GROUP_ORDER",
    "HATEFUL_LABELS",
    "Quadruple",
    "ParseIssue",
    "ParseResult",
    "QuadrupleValidationError",
    "ValueSpanMap",
    "adapt_source_quad",
    "canonicalize_quadruple",
    "canonicalize_quadruples",
    "serialize_quadruples",
    "serialize_with_spans",
    "parse_quadruples",
]
