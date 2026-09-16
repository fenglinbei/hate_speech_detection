"""Deterministic matcher for the reviewed annotated-lexicon repair line.

The matcher deliberately separates candidate generation from candidate
eligibility.  Canonical terms, explicit variants, and the frozen Unicode
normalization policy are the only span generators.  Per-entry regular
expressions can accept or reject an existing candidate, but can never create
or resize one.
"""

from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


MATCHER_POLICY_VERSION = "annotated-lexicon-controlled-longest/v1"
NORMALIZATION_VERSION = "unicode-nfc-fullwidth-ascii-ascii-casefold/v1"
CONTEXT_WINDOW = 16
MAX_PATTERN_CHARS = 256
MAX_RULES_PER_ENTRY = 8
MATCH_TIMEOUT_SECONDS = 0.005
QUERY_REGEX_BUDGET_SECONDS = 0.100
_RULE_TARGETS = frozenset({"left", "right", "surface", "context"})


class ControlledLexiconError(RuntimeError):
    """Fail-closed controlled matcher error."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_surface(value: str) -> str:
    """Apply only the normalization transformations frozen for repair v1."""

    normalized = unicodedata.normalize("NFC", value)
    output: list[str] = []
    for character in normalized:
        codepoint = ord(character)
        if codepoint == 0x3000:
            character = " "
        elif 0xFF01 <= codepoint <= 0xFF5E:
            character = chr(codepoint - 0xFEE0)
        if "A" <= character <= "Z":
            character = character.lower()
        output.append(character)
    return "".join(output)


def _normalized_text_with_boundaries(value: str) -> tuple[str, list[int]]:
    """Return normalized text and a normalized-boundary to raw-boundary map.

    NFC may collapse a base character and combining mark.  Prefix lengths
    provide an unambiguous right-edge mapping without introducing fuzzy span
    recovery.  Inputs are capped by the calling experiment at 8k tokens, so
    the simple prefix construction is intentionally preferred over a second
    normalization implementation with subtly different semantics.
    """

    normalized = normalize_surface(value)
    raw_by_normalized: dict[int, int] = {0: 0}
    for raw_end in range(1, len(value) + 1):
        normalized_end = len(normalize_surface(value[:raw_end]))
        raw_by_normalized[normalized_end] = raw_end
    try:
        boundaries = [raw_by_normalized[index] for index in range(len(normalized) + 1)]
    except KeyError as exc:
        raise ControlledLexiconError(
            "normalization produced an unmappable internal boundary"
        ) from exc
    return normalized, boundaries


def _regex_module() -> Any:
    try:
        import regex  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised by deployment check
        raise ControlledLexiconError(
            "match policies require the repair runtime pin regex==2026.4.4"
        ) from exc
    return regex


@dataclass(frozen=True)
class _CompiledRule:
    rule_id: str
    target: str
    pattern_text: str
    pattern: Any


class ControlledLexiconMatcher:
    """Generate, filter, and globally select non-overlapping lexicon spans."""

    def __init__(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        lexicon_sha256: str,
        policy_sha256: str | None = None,
    ) -> None:
        if not lexicon_sha256 or len(lexicon_sha256) != 64:
            raise ControlledLexiconError("lexicon_sha256 must be a SHA-256 hex digest")
        self.entries = [dict(entry) for entry in entries]
        self.lexicon_sha256 = lexicon_sha256
        self.compiled_policies: dict[int, dict[str, list[_CompiledRule]]] = {}
        surface_owners: dict[str, tuple[int, str]] = {}
        policy_identity: list[dict[str, Any]] = []

        for entry_index, entry in enumerate(self.entries):
            lexicon_id = str(entry.get("lexicon_id", "")).strip()
            term = str(entry.get("term", "")).strip()
            if not lexicon_id or not term:
                raise ControlledLexiconError("every entry requires lexicon_id and term")
            variants = entry.get("variants", []) or []
            if not isinstance(variants, list) or any(
                not isinstance(value, str) or not value.strip() for value in variants
            ):
                raise ControlledLexiconError(f"{lexicon_id}: variants must be non-empty strings")
            surfaces = [(term, "term"), *[(str(value), "variant") for value in variants]]
            for surface, source in surfaces:
                normalized = normalize_surface(surface)
                if not normalized:
                    raise ControlledLexiconError(f"{lexicon_id}: normalized surface is empty")
                prior = surface_owners.get(normalized)
                if prior is not None and prior[0] != entry_index:
                    other_id = self.entries[prior[0]].get("lexicon_id")
                    raise ControlledLexiconError(
                        f"normalized surface collision {surface!r}: {other_id} vs {lexicon_id}"
                    )
                surface_owners[normalized] = (entry_index, source)

            raw_policy = entry.get("match_policy") or {}
            compiled = self._compile_policy(lexicon_id, raw_policy)
            self.compiled_policies[entry_index] = compiled
            policy_identity.append(
                {
                    "lexicon_id": lexicon_id,
                    "match_policy": raw_policy,
                    "variants": variants,
                }
            )

        observed_policy_sha256 = _canonical_sha256(
            {
                "matcher_policy_version": MATCHER_POLICY_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "entries": policy_identity,
            }
        )
        if policy_sha256 is not None and policy_sha256 != observed_policy_sha256:
            raise ControlledLexiconError("matcher policy hash differs from lexicon declaration")
        self.policy_sha256 = observed_policy_sha256
        self.cache_identity = {
            "lexicon_sha256": self.lexicon_sha256,
            "matcher_policy_sha256": self.policy_sha256,
            "matcher_policy_version": MATCHER_POLICY_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        }

    @staticmethod
    def _compile_policy(
        lexicon_id: str, raw_policy: Any
    ) -> dict[str, list[_CompiledRule]]:
        if not isinstance(raw_policy, dict) or set(raw_policy) - {"require_any", "exclude_any"}:
            raise ControlledLexiconError(f"{lexicon_id}: match_policy fields differ")
        raw_groups = {
            "require_any": raw_policy.get("require_any", []) or [],
            "exclude_any": raw_policy.get("exclude_any", []) or [],
        }
        if any(not isinstance(value, list) for value in raw_groups.values()):
            raise ControlledLexiconError(f"{lexicon_id}: policy groups must be lists")
        if sum(len(value) for value in raw_groups.values()) > MAX_RULES_PER_ENTRY:
            raise ControlledLexiconError(f"{lexicon_id}: policy exceeds {MAX_RULES_PER_ENTRY} rules")
        if not any(raw_groups.values()):
            return {"require_any": [], "exclude_any": []}
        regex = _regex_module()
        seen_ids: set[str] = set()
        compiled: dict[str, list[_CompiledRule]] = {"require_any": [], "exclude_any": []}
        for group, rows in raw_groups.items():
            for row in rows:
                if not isinstance(row, dict) or set(row) != {"rule_id", "target", "pattern"}:
                    raise ControlledLexiconError(f"{lexicon_id}: policy rule fields differ")
                rule_id = str(row["rule_id"]).strip()
                target = str(row["target"])
                pattern_text = str(row["pattern"])
                if not rule_id or rule_id in seen_ids:
                    raise ControlledLexiconError(f"{lexicon_id}: rule_id is missing or duplicated")
                if target not in _RULE_TARGETS:
                    raise ControlledLexiconError(f"{lexicon_id}/{rule_id}: invalid rule target")
                if not pattern_text or len(pattern_text) > MAX_PATTERN_CHARS:
                    raise ControlledLexiconError(f"{lexicon_id}/{rule_id}: pattern length is invalid")
                try:
                    pattern = regex.compile(pattern_text)
                except Exception as exc:
                    raise ControlledLexiconError(
                        f"{lexicon_id}/{rule_id}: regex compilation failed"
                    ) from exc
                seen_ids.add(rule_id)
                compiled[group].append(_CompiledRule(rule_id, target, pattern_text, pattern))
        return compiled

    def _candidate_is_eligible(
        self,
        *,
        entry_index: int,
        normalized_query: str,
        normalized_start: int,
        normalized_end: int,
        started_at: float,
    ) -> tuple[bool, dict[str, Any]]:
        left = normalized_query[max(0, normalized_start - CONTEXT_WINDOW) : normalized_start]
        surface = normalized_query[normalized_start:normalized_end]
        right = normalized_query[normalized_end : normalized_end + CONTEXT_WINDOW]
        targets = {
            "left": left,
            "right": right,
            "surface": surface,
            "context": left + surface + right,
        }
        groups = self.compiled_policies[entry_index]
        matched: dict[str, list[str]] = {"require_any": [], "exclude_any": []}
        for group in ("exclude_any", "require_any"):
            for rule in groups[group]:
                if time.monotonic() - started_at > QUERY_REGEX_BUDGET_SECONDS:
                    raise ControlledLexiconError("query regex budget exceeded")
                try:
                    if rule.pattern.search(
                        targets[rule.target], timeout=MATCH_TIMEOUT_SECONDS
                    ) is not None:
                        matched[group].append(rule.rule_id)
                except TimeoutError as exc:
                    raise ControlledLexiconError(
                        f"regex timeout for rule {rule.rule_id}"
                    ) from exc
        eligible = not matched["exclude_any"] and (
            not groups["require_any"] or bool(matched["require_any"])
        )
        return eligible, {
            "context_window": CONTEXT_WINDOW,
            "normalized_left": left,
            "normalized_surface": surface,
            "normalized_right": right,
            "matched_exclude_rule_ids": matched["exclude_any"],
            "matched_require_rule_ids": matched["require_any"],
        }

    def match(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str):
            raise ControlledLexiconError("query must be a string")
        normalized_query, boundaries = _normalized_text_with_boundaries(query)
        started_at = time.monotonic()
        candidates_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}

        for entry_index, entry in enumerate(self.entries):
            surfaces = [(str(entry["term"]), "term")]
            surfaces.extend((str(value), "variant") for value in entry.get("variants", []) or [])
            for declared_surface, source in surfaces:
                normalized_surface = normalize_surface(declared_surface)
                cursor = 0
                while True:
                    position = normalized_query.find(normalized_surface, cursor)
                    if position < 0:
                        break
                    normalized_end = position + len(normalized_surface)
                    raw_start = boundaries[position]
                    raw_end = boundaries[normalized_end]
                    key = (entry_index, raw_start, raw_end)
                    candidate = candidates_by_key.get(key)
                    source_record = {
                        "kind": source,
                        "declared_surface": declared_surface,
                        "normalization_applied": declared_surface != query[raw_start:raw_end],
                    }
                    if candidate is None:
                        candidate = {
                            "candidate_id": "candidate-" + _canonical_sha256(
                                [entry["lexicon_id"], raw_start, raw_end]
                            )[:16],
                            "entry_index": entry_index,
                            "lexicon_id": str(entry["lexicon_id"]),
                            "term": str(entry["term"]),
                            "raw_surface": query[raw_start:raw_end],
                            "span": [raw_start, raw_end],
                            "normalized_span": [position, normalized_end],
                            "generation_sources": [],
                        }
                        candidates_by_key[key] = candidate
                    if source_record not in candidate["generation_sources"]:
                        candidate["generation_sources"].append(source_record)
                    cursor = position + 1

        exact_span_owners: dict[tuple[int, int], set[str]] = defaultdict(set)
        for candidate in candidates_by_key.values():
            exact_span_owners[tuple(candidate["span"])].add(candidate["lexicon_id"])
        collisions = {
            span: owners for span, owners in exact_span_owners.items() if len(owners) > 1
        }
        if collisions:
            raise ControlledLexiconError(f"same-span cross-entry collision: {collisions}")

        candidates = list(candidates_by_key.values())
        for candidate in candidates:
            eligible, policy_trace = self._candidate_is_eligible(
                entry_index=int(candidate["entry_index"]),
                normalized_query=normalized_query,
                normalized_start=int(candidate["normalized_span"][0]),
                normalized_end=int(candidate["normalized_span"][1]),
                started_at=started_at,
            )
            candidate["policy"] = policy_trace
            candidate["eligible"] = eligible
            candidate["selection"] = "eligible" if eligible else "policy_excluded"

        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        eligible.sort(
            key=lambda row: (
                -(int(row["span"][1]) - int(row["span"][0])),
                int(row["span"][0]),
                str(row["lexicon_id"]),
            )
        )
        selected: list[dict[str, Any]] = []
        for candidate in eligible:
            overlap = next(
                (
                    winner
                    for winner in selected
                    if int(candidate["span"][0]) < int(winner["span"][1])
                    and int(winner["span"][0]) < int(candidate["span"][1])
                ),
                None,
            )
            if overlap is None:
                candidate["selection"] = "selected"
                selected.append(candidate)
            else:
                candidate["selection"] = "overlap_lost"
                candidate["lost_to_candidate_id"] = overlap["candidate_id"]

        selected.sort(key=lambda row: (int(row["span"][0]), int(row["span"][1])))
        candidates.sort(
            key=lambda row: (
                int(row["span"][0]),
                -(int(row["span"][1]) - int(row["span"][0])),
                str(row["lexicon_id"]),
            )
        )
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for candidate in selected:
            grouped[int(candidate["entry_index"])].append(candidate)
        selected_hits: list[dict[str, Any]] = []
        for entry_index, rows in grouped.items():
            entry = dict(self.entries[entry_index])
            entry["match_spans"] = [row["span"] for row in rows]
            entry["matched_surfaces"] = [row["raw_surface"] for row in rows]
            entry["candidate_ids"] = [row["candidate_id"] for row in rows]
            selected_hits.append(entry)
        selected_hits.sort(
            key=lambda row: (
                int(row["match_spans"][0][0]),
                -len(str(row["term"])),
                str(row["lexicon_id"]),
            )
        )
        for rank, row in enumerate(selected_hits):
            row["rank"] = rank

        return {
            "query": query,
            "normalized_query": normalized_query,
            "cache_identity": dict(self.cache_identity),
            "candidates": candidates,
            "selected_spans": [
                {
                    key: candidate[key]
                    for key in (
                        "candidate_id",
                        "lexicon_id",
                        "term",
                        "raw_surface",
                        "span",
                    )
                }
                for candidate in selected
            ],
            "selected_hits": selected_hits,
        }


__all__ = [
    "CONTEXT_WINDOW",
    "ControlledLexiconError",
    "ControlledLexiconMatcher",
    "MATCHER_POLICY_VERSION",
    "MATCH_TIMEOUT_SECONDS",
    "MAX_PATTERN_CHARS",
    "MAX_RULES_PER_ENTRY",
    "NORMALIZATION_VERSION",
    "QUERY_REGEX_BUDGET_SECONDS",
    "normalize_surface",
]
