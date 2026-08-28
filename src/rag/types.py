"""Structured, serialization-friendly retrieval types for Stage 1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def content_sha256(content: str) -> str:
    """Hash source text after the only permitted newline normalization."""

    return sha256_text(str(content).replace("\r\n", "\n"))


def stable_demo_id(source_record_id: str, content_hash: str, gold_hash: str) -> str:
    payload = {
        "source_record_id": str(source_record_id),
        "content_sha256": content_hash,
        "gold_sha256": gold_hash,
    }
    return "demo:v1:" + sha256_text(canonical_json(payload))


def stable_lexicon_id(
    term: str,
    category: str,
    definition: str,
    variants: list[str] | tuple[str, ...] | None = None,
) -> str:
    payload = {
        "term": str(term),
        "category": str(category),
        "definition": str(definition),
        "variants": list(variants or []),
    }
    return "lex:v1:" + sha256_text(canonical_json(payload))


def stable_term_evidence_id(
    term: str,
    definition: str,
    variants: list[str] | tuple[str, ...] | None = None,
    usage_notes: str = "",
    ambiguity_notes: str = "",
) -> str:
    """Identify category-free terminology evidence by model-visible semantics."""

    payload = {
        "term": str(term),
        "definition": str(definition),
        "variants": list(variants or []),
        "usage_notes": str(usage_notes),
        "ambiguity_notes": str(ambiguity_notes),
    }
    return "lex:v2:" + sha256_text(canonical_json(payload))


@dataclass(frozen=True)
class RetrievalHit:
    """One raw retrieval result before global selection.

    ``score`` remains the backend/raw score. The selector computes and records
    the half-even eight-decimal written value separately.
    """

    id: str
    source_record_id: str
    content: str
    output: str | None
    content_sha256: str
    gold_sha256: str | None
    score: float | None
    rank: int
    source_class: str
    method: str
    provenance: Mapping[str, Any]

    @property
    def demo_id(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_record_id": self.source_record_id,
            "content": self.content,
            "output": self.output,
            "content_sha256": self.content_sha256,
            "gold_sha256": self.gold_sha256,
            "score": self.score,
            "rank": self.rank,
            "source_class": self.source_class,
            "method": self.method,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class RetrievalEvidence:
    source_class: str
    source_class_index: int
    method: str
    source_rank: int
    raw_similarity: float | None
    written_similarity: float | None
    eligible: bool
    match_spans: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_class": self.source_class,
            "source_class_index": self.source_class_index,
            "method": self.method,
            "source_rank": self.source_rank,
            "raw_similarity": self.raw_similarity,
            "written_similarity": self.written_similarity,
            "eligible": self.eligible,
            "match_spans": [list(span) for span in self.match_spans],
        }


@dataclass(frozen=True)
class DemoCandidate:
    demo_id: str
    source_record_id: str
    content: str
    output: str | None
    content_sha256: str
    gold_sha256: str | None
    selection_score: float
    tie_source_class: str
    tie_source_class_index: int
    tie_source_rank: int
    evidence: tuple[RetrievalEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "source_record_id": self.source_record_id,
            "content": self.content,
            "output": self.output,
            "content_sha256": self.content_sha256,
            "gold_sha256": self.gold_sha256,
            "selection_score": self.selection_score,
            "tie_source_class": self.tie_source_class,
            "tie_source_class_index": self.tie_source_class_index,
            "tie_source_rank": self.tie_source_rank,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class QuotaAssignment:
    demo_id: str
    assigned_quota_class: str
    quota_round: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "demo_id": self.demo_id,
            "assigned_quota_class": self.assigned_quota_class,
            "quota_round": self.quota_round,
        }


@dataclass(frozen=True)
class LexiconCandidate:
    lexicon_id: str
    content: str
    content_sha256: str
    source_class: str
    evidence: tuple[RetrievalEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexicon_id": self.lexicon_id,
            "content": self.content,
            "content_sha256": self.content_sha256,
            "source_class": self.source_class,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class LexiconSelectionTrace:
    policy: str
    candidates: tuple[LexiconCandidate, ...]
    selected_ids: tuple[str, ...]
    prompt_order: tuple[str, ...]
    duplicate_occurrences_merged: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_ids": list(self.selected_ids),
            "prompt_order": list(self.prompt_order),
            "duplicate_occurrences_merged": self.duplicate_occurrences_merged,
        }


@dataclass(frozen=True)
class DemoSelectionTrace:
    policy: str
    source_class_order: tuple[str, ...]
    allocated_class_top_k: Mapping[str, int]
    similarity_threshold: float
    candidate_multiplier: int
    candidates: tuple[DemoCandidate, ...]
    class_candidate_ids: Mapping[str, tuple[str, ...]]
    selected_ids: tuple[str, ...]
    quota_assignments: tuple[QuotaAssignment, ...]
    prompt_order: tuple[str, ...]
    excluded: tuple[Mapping[str, str], ...]
    duplicate_occurrences_merged: int

    @property
    def prompt_order_before_budget(self) -> tuple[str, ...]:
        return self.prompt_order

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "source_class_order": list(self.source_class_order),
            "allocated_class_top_k": dict(self.allocated_class_top_k),
            "similarity_threshold": self.similarity_threshold,
            "candidate_multiplier": self.candidate_multiplier,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "class_candidate_ids": {
                key: list(value) for key, value in self.class_candidate_ids.items()
            },
            "selected_ids": list(self.selected_ids),
            "quota_assignments": [assignment.to_dict() for assignment in self.quota_assignments],
            "prompt_order": list(self.prompt_order),
            "excluded": [dict(item) for item in self.excluded],
            "duplicate_occurrences_merged": self.duplicate_occurrences_merged,
        }


__all__ = [
    "RetrievalHit",
    "RetrievalEvidence",
    "DemoCandidate",
    "LexiconCandidate",
    "QuotaAssignment",
    "DemoSelectionTrace",
    "LexiconSelectionTrace",
    "canonical_json",
    "sha256_text",
    "content_sha256",
    "stable_demo_id",
    "stable_lexicon_id",
]
