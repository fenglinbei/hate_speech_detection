"""Pure deterministic Stage-1 demo selection and audit trace construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any

from rag.types import (
    DemoCandidate,
    DemoSelectionTrace,
    LexiconCandidate,
    LexiconSelectionTrace,
    QuotaAssignment,
    RetrievalEvidence,
    RetrievalHit,
)


SELECTION_POLICY = "class-quota-round-robin-global-dedup/v1"
SOURCE_RANK_POLICY = "round-filter-sort-rerank/v1"
SELECTION_SCORE_POLICY = "eligible-written-cosine-max/v1"
LEXICON_SELECTION_POLICY = "exact-first-semantic-global-dedup/v1"
_SCORE_QUANTUM = Decimal("0.00000001")


class ContextSelectionError(ValueError):
    pass


class NonFiniteScoreError(ContextSelectionError):
    pass


class CandidateIdentityError(ContextSelectionError):
    pass


class QuotaUnsatisfiedError(ContextSelectionError):
    def __init__(self, source_class: str, quota_round: int, quota: int, candidate_count: int):
        self.source_class = source_class
        self.quota_round = quota_round
        self.quota = quota
        self.candidate_count = candidate_count
        super().__init__(
            f"cannot fill quota slot class={source_class!r} round={quota_round}; "
            f"quota={quota}, retained_candidates={candidate_count}"
        )


def round_similarity_half_even(score: float, digits: int = 8) -> float:
    """Return the persisted score used by every threshold and tie-break."""

    try:
        numeric = float(score)
    except (TypeError, ValueError) as exc:
        raise NonFiniteScoreError(f"retrieval score is not numeric: {score!r}") from exc
    if not math.isfinite(numeric):
        raise NonFiniteScoreError(f"retrieval score is non-finite: {score!r}")
    quantum = _SCORE_QUANTUM if digits == 8 else Decimal(1).scaleb(-digits)
    try:
        rounded = Decimal(str(score)).quantize(quantum, rounding=ROUND_HALF_EVEN)
    except (InvalidOperation, ValueError) as exc:
        raise NonFiniteScoreError(f"retrieval score cannot be rounded: {score!r}") from exc
    value = float(rounded)
    return 0.0 if value == 0 else value


def _identity(hit: RetrievalHit) -> tuple[str, str, str | None]:
    return hit.source_record_id, hit.content_sha256, hit.gold_sha256


def _validate_same_identity(first: RetrievalHit, second: RetrievalHit) -> None:
    if _identity(first) != _identity(second):
        raise CandidateIdentityError(
            f"retrieval ID {first.id!r} maps to conflicting source/content/gold hashes"
        )


def _validate_config(
    source_class_order: Sequence[str],
    allocated_class_top_k: Mapping[str, int],
    candidate_multiplier: int,
) -> tuple[str, ...]:
    order = tuple(str(value) for value in source_class_order)
    if not order or len(order) != len(set(order)):
        raise ContextSelectionError("source_class_order must be non-empty and unique")
    if not isinstance(candidate_multiplier, int) or candidate_multiplier < 1:
        raise ContextSelectionError("candidate_multiplier must be a positive integer")
    missing = [source_class for source_class in order if source_class not in allocated_class_top_k]
    if missing:
        raise ContextSelectionError(f"missing allocated quota for classes: {missing}")
    for source_class in order:
        quota = allocated_class_top_k[source_class]
        if not isinstance(quota, int) or quota < 0:
            raise ContextSelectionError(f"quota for {source_class!r} must be a non-negative integer")
    return order


def select_demos(
    hits_by_class: Mapping[str, Sequence[RetrievalHit]],
    *,
    source_class_order: Sequence[str],
    allocated_class_top_k: Mapping[str, int],
    similarity_threshold: float = 0.0,
    candidate_multiplier: int = 3,
    query_source_record_id: str | None = None,
    query_content_sha256: str | None = None,
) -> DemoSelectionTrace:
    """Select demos with global dedupe, quota refill, and independent ordering."""

    order = _validate_config(source_class_order, allocated_class_top_k, candidate_multiplier)
    try:
        threshold = float(similarity_threshold)
    except (TypeError, ValueError) as exc:
        raise ContextSelectionError("similarity_threshold must be finite") from exc
    if not math.isfinite(threshold):
        raise ContextSelectionError("similarity_threshold must be finite")
    class_index = {source_class: index for index, source_class in enumerate(order)}
    merged_hits: dict[str, RetrievalHit] = {}
    evidence_by_id: dict[str, list[RetrievalEvidence]] = {}
    class_candidate_ids: dict[str, tuple[str, ...]] = {}
    excluded: list[dict[str, str]] = []
    duplicate_occurrences = 0

    for source_class in order:
        quota = allocated_class_top_k[source_class]
        unique_for_class: dict[str, tuple[RetrievalHit, float]] = {}
        for hit in hits_by_class.get(source_class, ()):
            if not isinstance(hit, RetrievalHit):
                raise ContextSelectionError(f"{source_class!r} contains a non-RetrievalHit value")
            if hit.score is None:
                raise NonFiniteScoreError(f"demo hit {hit.id!r} has no cosine score")
            written_score = round_similarity_half_even(hit.score)
            if query_source_record_id is not None and hit.source_record_id == str(query_source_record_id):
                excluded.append({"demo_id": hit.id, "source_class": source_class, "reason": "source_record_id_overlap"})
                continue
            if query_content_sha256 is not None and hit.content_sha256 == query_content_sha256:
                excluded.append({"demo_id": hit.id, "source_class": source_class, "reason": "content_sha256_overlap"})
                continue
            previous = unique_for_class.get(hit.id)
            if previous is not None:
                _validate_same_identity(previous[0], hit)
                duplicate_occurrences += 1
                if written_score > previous[1]:
                    unique_for_class[hit.id] = (hit, written_score)
                continue
            unique_for_class[hit.id] = (hit, written_score)

        eligible = [
            pair for pair in unique_for_class.values() if pair[1] >= threshold
        ]
        eligible.sort(key=lambda pair: (-pair[1], pair[0].id))
        reranked = eligible[: quota * candidate_multiplier]
        retained_ids: list[str] = []
        for source_rank, (hit, written_score) in enumerate(reranked):
            retained_ids.append(hit.id)
            prior = merged_hits.get(hit.id)
            if prior is not None:
                _validate_same_identity(prior, hit)
                duplicate_occurrences += 1
            else:
                merged_hits[hit.id] = hit
            evidence_by_id.setdefault(hit.id, []).append(
                RetrievalEvidence(
                    source_class=source_class,
                    source_class_index=class_index[source_class],
                    method=hit.method,
                    source_rank=source_rank,
                    raw_similarity=float(hit.score),
                    written_similarity=written_score,
                    eligible=True,
                )
            )
        class_candidate_ids[source_class] = tuple(retained_ids)

    candidates_by_id: dict[str, DemoCandidate] = {}
    for demo_id, hit in merged_hits.items():
        evidence = tuple(
            sorted(
                evidence_by_id[demo_id],
                key=lambda item: (item.source_class_index, item.source_rank),
            )
        )
        selection_score = max(
            item.written_similarity for item in evidence if item.written_similarity is not None
        )
        tied = [item for item in evidence if item.written_similarity == selection_score]
        tie = min(tied, key=lambda item: (item.source_class_index, item.source_rank))
        candidates_by_id[demo_id] = DemoCandidate(
            demo_id=demo_id,
            source_record_id=hit.source_record_id,
            content=hit.content,
            output=hit.output,
            content_sha256=hit.content_sha256,
            gold_sha256=hit.gold_sha256,
            selection_score=selection_score,
            tie_source_class=tie.source_class,
            tie_source_class_index=tie.source_class_index,
            tie_source_rank=tie.source_rank,
            evidence=evidence,
        )

    selected: set[str] = set()
    assignments: list[QuotaAssignment] = []
    cursors = {source_class: 0 for source_class in order}
    max_quota = max(allocated_class_top_k[source_class] for source_class in order)
    for quota_round in range(max_quota):
        for source_class in order:
            quota = allocated_class_top_k[source_class]
            if quota_round >= quota:
                continue
            candidate_ids = class_candidate_ids[source_class]
            cursor = cursors[source_class]
            while cursor < len(candidate_ids) and candidate_ids[cursor] in selected:
                cursor += 1
            if cursor >= len(candidate_ids):
                raise QuotaUnsatisfiedError(source_class, quota_round, quota, len(candidate_ids))
            demo_id = candidate_ids[cursor]
            cursors[source_class] = cursor + 1
            selected.add(demo_id)
            assignments.append(
                QuotaAssignment(
                    demo_id=demo_id,
                    assigned_quota_class=source_class,
                    quota_round=quota_round,
                )
            )

    def prompt_key(demo_id: str) -> tuple[Any, ...]:
        candidate = candidates_by_id[demo_id]
        return (
            -candidate.selection_score,
            candidate.tie_source_class_index,
            candidate.tie_source_rank,
            candidate.demo_id,
        )

    candidate_order = tuple(sorted(candidates_by_id, key=prompt_key))
    selected_ids = tuple(assignment.demo_id for assignment in assignments)
    prompt_order = tuple(sorted(selected, key=prompt_key))
    return DemoSelectionTrace(
        policy=SELECTION_POLICY,
        source_class_order=order,
        allocated_class_top_k={source_class: allocated_class_top_k[source_class] for source_class in order},
        similarity_threshold=threshold,
        candidate_multiplier=candidate_multiplier,
        candidates=tuple(candidates_by_id[demo_id] for demo_id in candidate_order),
        class_candidate_ids=class_candidate_ids,
        selected_ids=selected_ids,
        quota_assignments=tuple(assignments),
        prompt_order=prompt_order,
        excluded=tuple(excluded),
        duplicate_occurrences_merged=duplicate_occurrences,
    )


def select_lexicons(
    exact_hits: Sequence[RetrievalHit],
    semantic_hits: Sequence[RetrievalHit],
    *,
    exact_top_k: int = -1,
    semantic_top_k: int = -1,
    similarity_threshold: float | None = None,
) -> LexiconSelectionTrace:
    """Merge exact and semantic lexicon evidence, rendering exact hits first."""

    if exact_top_k < -1 or semantic_top_k < -1:
        raise ContextSelectionError("lexicon top_k values must be -1 or non-negative")
    try:
        threshold = float(similarity_threshold) if similarity_threshold is not None else None
    except (TypeError, ValueError) as exc:
        raise ContextSelectionError("lexicon similarity_threshold must be finite") from exc
    if threshold is not None and not math.isfinite(threshold):
        raise ContextSelectionError("lexicon similarity_threshold must be finite")
    identity_by_id: dict[str, RetrievalHit] = {}
    evidence_by_id: dict[str, list[RetrievalEvidence]] = {}
    exact_order: list[str] = []
    semantic_order: list[str] = []
    duplicates = 0

    ordered_exact = sorted(exact_hits, key=lambda hit: (hit.rank, hit.id))
    if exact_top_k >= 0:
        ordered_exact = ordered_exact[:exact_top_k]
    for hit in ordered_exact:
        if hit.score is not None:
            raise ContextSelectionError(f"exact lexicon hit {hit.id!r} must use score=None")
        prior = identity_by_id.get(hit.id)
        if prior is not None:
            _validate_same_identity(prior, hit)
            duplicates += 1
        else:
            identity_by_id[hit.id] = hit
            exact_order.append(hit.id)
        raw_spans = hit.provenance.get("match_spans", ()) if isinstance(hit.provenance, Mapping) else ()
        spans = tuple(
            (int(span[0]), int(span[1]))
            for span in raw_spans
            if isinstance(span, (list, tuple)) and len(span) == 2
        )
        evidence_by_id.setdefault(hit.id, []).append(
            RetrievalEvidence(
                source_class=hit.source_class,
                source_class_index=-1,
                method=hit.method,
                source_rank=hit.rank,
                raw_similarity=None,
                written_similarity=None,
                eligible=True,
                match_spans=spans,
            )
        )

    semantic_rows: list[tuple[RetrievalHit, float]] = []
    for hit in semantic_hits:
        if hit.score is None:
            raise NonFiniteScoreError(f"semantic lexicon hit {hit.id!r} has no score")
        written = round_similarity_half_even(hit.score)
        if threshold is not None and written < threshold:
            continue
        semantic_rows.append((hit, written))
    # Upstream ranks are derived from category-free, model-visible evidence. Keep
    # that ordering for rounded-score ties so private audit metadata embedded in
    # an internal ID cannot influence which evidence reaches the prompt.
    semantic_rows.sort(key=lambda pair: (-pair[1], pair[0].rank, pair[0].id))
    semantic_added = 0
    for semantic_rank, (hit, written) in enumerate(semantic_rows):
        prior = identity_by_id.get(hit.id)
        if prior is not None:
            _validate_same_identity(prior, hit)
            duplicates += 1
        elif semantic_top_k >= 0 and semantic_added >= semantic_top_k:
            continue
        else:
            identity_by_id[hit.id] = hit
            semantic_order.append(hit.id)
            semantic_added += 1
        evidence_by_id.setdefault(hit.id, []).append(
            RetrievalEvidence(
                source_class=hit.source_class,
                source_class_index=-1,
                method=hit.method,
                source_rank=semantic_rank,
                raw_similarity=float(hit.score),
                written_similarity=written,
                eligible=True,
            )
        )

    selected_ids = tuple([*exact_order, *semantic_order])
    candidates = tuple(
        LexiconCandidate(
            lexicon_id=lexicon_id,
            content=identity_by_id[lexicon_id].content,
            content_sha256=identity_by_id[lexicon_id].content_sha256,
            source_class=identity_by_id[lexicon_id].source_class,
            evidence=tuple(evidence_by_id[lexicon_id]),
        )
        for lexicon_id in selected_ids
    )
    return LexiconSelectionTrace(
        policy=LEXICON_SELECTION_POLICY,
        candidates=candidates,
        selected_ids=selected_ids,
        prompt_order=selected_ids,
        duplicate_occurrences_merged=duplicates,
    )


select_demo_context = select_demos


__all__ = [
    "SELECTION_POLICY",
    "SOURCE_RANK_POLICY",
    "SELECTION_SCORE_POLICY",
    "LEXICON_SELECTION_POLICY",
    "ContextSelectionError",
    "NonFiniteScoreError",
    "CandidateIdentityError",
    "QuotaUnsatisfiedError",
    "round_similarity_half_even",
    "select_demos",
    "select_demo_context",
    "select_lexicons",
]
