"""Deterministic Stage-1 counterfactual proposal and finalization primitives."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from metrics.stage1_margin import replace_one_field
from utils.quadruple import GROUP_ORDER, Quadruple, canonicalize_quadruples, serialize_quadruples


PROPOSAL_ROW_SCHEMA = "stage1-cf-proposal-row/v1"
REVIEW_ROW_SCHEMA = "stage1-cf-review-row/v1"
CF_ROW_SCHEMA = "stage1-cf/v1"
CANDIDATE_POLICY_VERSION = "query-local-candidate-policy/v1"
SELECTION_POLICY = "query-local-valid-first/v1"
FIELDS = ("target", "argument", "targeted_group", "hateful")
REVIEW_FIELDS = ("target", "argument")
FAMILY_ORDER = (
    "boundary-edit",
    "multi-tuple-binding-swap",
    "query-local-distractor-span",
)
RUBRIC_META_SCHEMA = "stage1-cf-rubric-meta/v1"
REVIEWER_DECLARATION_SCHEMA = "stage1-cf-reviewer-declaration/v1"
PROPOSAL_META_SCHEMA = "stage1-cf-proposal/v1"
PROPOSAL_SUMMARY_SCHEMA = "stage1-cf-proposal-summary/v1"
REVIEW_META_SCHEMA = "stage1-cf-review/v1"
FINAL_META_SCHEMA = "stage1-cf-manifest/v1"
PROPOSAL_ARTIFACT_KIND = "cf-proposal"
REVIEW_ARTIFACT_KIND = "cf-review"
FINAL_ARTIFACT_KIND = "counterfactual"
FOIL_POLICY_SCHEMA = "stage1-cf-foil-policy/v1"
RESOLVED_CONFIG_SCHEMA = "stage1-cf-resolved-config/v1"
ENGINEERING_CONTEXT_SCHEMA = "stage1-engineering-context-fixture/v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROPOSAL_ID_RE = re.compile(r"^cfp-[0-9a-f]{64}$")
REVIEW_ID_RE = re.compile(r"^review:v1:[0-9a-f]{64}$")
FINAL_ID_RE = re.compile(r"^cf-[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = REPOSITORY_ROOT / "schemas"

PROPOSAL_ROW_KEYS = frozenset(
    {
        "schema_version",
        "cf_proposal_id",
        "candidate_id",
        "context_record_sha256",
        "query_id",
        "query_content",
        "gold_sha256",
        "gold_quadruples",
        "tuple_index",
        "field",
        "gold_value",
        "candidate_value",
        "candidate_value_sha256",
        "family",
        "source",
        "source_id",
        "review_required",
    }
)
REVIEW_ROW_KEYS = frozenset(
    {
        "schema_version",
        "cf_proposal_id",
        "candidate_id",
        "decision",
        "reason_code",
        "note",
        "reviewer_id",
    }
)
DECLARATION_KEYS = frozenset(
    {
        "schema_version",
        "cf_proposal_id",
        "cf_blind_review_dependency",
        "human_queue_rows_sha256",
        "auto_review_rows_sha256",
        "reviewer_id",
        "rubric_body_sha256",
        "rubric_meta_sha256",
        "completed_review_rows_sha256",
        "saw_condition_outputs",
        "saw_model_scores",
        "attestation_confirmed",
    }
)


class CounterfactualError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_value(field: str, value: Any) -> Any:
    if field not in FIELDS:
        raise CounterfactualError(f"unknown field: {field}")
    probe = {
        "target": "probe",
        "argument": "probe",
        "targeted_group": ["others"],
        "hateful": "hate",
    }
    probe[field] = value
    try:
        quad = canonicalize_quadruples([probe])[0]
    except Exception as exc:
        raise CounterfactualError(f"invalid candidate for {field}: {exc}") from exc
    normalized = getattr(quad, field)
    return list(normalized) if field == "targeted_group" else normalized


def _quad_mappings(quads: Sequence[Quadruple]) -> list[dict[str, Any]]:
    return [
        {
            "target": quad.target,
            "argument": quad.argument,
            "targeted_group": list(quad.targeted_group),
            "hateful": quad.hateful,
        }
        for quad in quads
    ]


def candidate_id(
    *,
    query_id: str,
    gold_sha256: str,
    tuple_index: int,
    field: str,
    candidate_value: Any,
    family: str,
    source: str,
    source_id: str,
) -> str:
    payload = {
        "query_id": str(query_id),
        "gold_sha256": gold_sha256,
        "tuple_index": tuple_index,
        "field": field,
        "candidate_value": canonical_value(field, candidate_value),
        "family": family,
        "source": source,
        "source_id": str(source_id),
    }
    return "cf:v1:" + canonical_sha256(payload)


def _record_query(record: Mapping[str, Any]) -> tuple[str, str, list[Quadruple], str]:
    query = record.get("query", record)
    if not isinstance(query, Mapping):
        raise CounterfactualError("context/data record lacks query object")
    query_id = str(query.get("id", ""))
    content = query.get("content")
    gold_raw = query.get("gold", query.get("quadruples"))
    if not query_id or not isinstance(content, str) or not content:
        raise CounterfactualError("query id/content is invalid")
    try:
        gold = canonicalize_quadruples(gold_raw)
    except Exception as exc:
        raise CounterfactualError(f"query {query_id} gold is invalid: {exc}") from exc
    context_hash = record.get("record_sha256")
    if not isinstance(context_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", context_hash):
        # Pure fixtures may not yet be wrapped in a frozen context target.  The
        # deterministic projection is explicit and cannot be used as a formal ref.
        context_hash = canonical_sha256(record)
    return query_id, content, gold, context_hash


def group_combination_frequencies(train_records: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, ...]]:
    frequencies: Counter[tuple[str, ...]] = Counter()
    for record in train_records:
        _, _, quads, _ = _record_query(record)
        frequencies.update(quad.targeted_group for quad in quads)
    if len(frequencies) < 2:
        raise CounterfactualError("train-only group catalog must contain at least two combinations")
    return frequencies


def _best_group_foil(
    gold: tuple[str, ...],
    frequencies: Mapping[tuple[str, ...], int],
) -> list[str]:
    order = {label: index for index, label in enumerate(GROUP_ORDER)}
    candidates = [combo for combo in frequencies if combo != gold]
    if not candidates:
        raise CounterfactualError("no non-gold group combination is available")
    selected = min(
        candidates,
        key=lambda combo: (
            len(set(gold) ^ set(combo)),
            abs(len(gold) - len(combo)),
            -int(frequencies[combo]),
            tuple(order[label] for label in combo),
        ),
    )
    return list(selected)


def _candidate_row(
    *,
    cf_proposal_id: str,
    context_record_sha256: str,
    query_id: str,
    query_content: str,
    gold_sha256: str,
    tuple_index: int,
    field: str,
    gold_value: Any,
    candidate_value: Any,
    family: str,
    source: str,
    source_id: str,
    review_required: bool,
    gold_quads: Sequence[Quadruple],
) -> dict[str, Any]:
    value = canonical_value(field, candidate_value)
    normalized_gold = canonical_value(field, gold_value)
    if value == normalized_gold:
        raise CounterfactualError("canonical candidate equals gold")
    # This is the executable one-field-only invariant used again by the scorer.
    replace_one_field(
        gold_quads,
        tuple_index=tuple_index,
        field=field,
        candidate_value=value,
    )
    identifier = candidate_id(
        query_id=query_id,
        gold_sha256=gold_sha256,
        tuple_index=tuple_index,
        field=field,
        candidate_value=value,
        family=family,
        source=source,
        source_id=source_id,
    )
    return {
        "schema_version": PROPOSAL_ROW_SCHEMA,
        "cf_proposal_id": cf_proposal_id,
        "candidate_id": identifier,
        "context_record_sha256": context_record_sha256,
        "query_id": query_id,
        "query_content": query_content,
        "gold_sha256": gold_sha256,
        "gold_quadruples": _quad_mappings(gold_quads),
        "tuple_index": tuple_index,
        "field": field,
        "gold_value": normalized_gold,
        "candidate_value": value,
        "candidate_value_sha256": canonical_sha256(value),
        "family": family,
        "source": source,
        "source_id": source_id,
        "review_required": review_required,
    }


def _boundary_values(value: str | None, content: str) -> list[tuple[str, str]]:
    if not isinstance(value, str) or len(value) <= 1:
        return []
    candidates = [(value[1:], "drop-left-1"), (value[:-1], "drop-right-1")]
    return [(candidate, source_id) for candidate, source_id in candidates if candidate and candidate in content]


def _query_segments(content: str) -> list[str]:
    segments = [part.strip() for part in re.split(r"[\s，。！？；、,.!?;:：]+", content) if part.strip()]
    # Long punctuation-free Chinese sentences are not blindly decomposed into
    # arbitrary n-grams; boundary edits and binding swaps remain available.
    return list(dict.fromkeys(segment for segment in segments if 1 <= len(segment) <= 80))


def _review_candidates_for_unit(
    *,
    field: str,
    tuple_index: int,
    content: str,
    gold_quads: Sequence[Quadruple],
) -> list[tuple[Any, str, str, str]]:
    gold_value = getattr(gold_quads[tuple_index], field)
    candidates: list[tuple[Any, str, str, str]] = []
    for value, edit_id in _boundary_values(gold_value, content):
        candidates.append((value, "boundary-edit", "same-query", edit_id))
    for other_index, other in enumerate(gold_quads):
        value = getattr(other, field)
        if other_index != tuple_index and value is not None and value != gold_value:
            candidates.append(
                (value, "multi-tuple-binding-swap", "same-query", f"tuple-{other_index}")
            )
    for segment_index, segment in enumerate(_query_segments(content)):
        if segment != gold_value and segment not in {value for value, *_ in candidates}:
            candidates.append(
                (segment, "query-local-distractor-span", "same-query", f"segment-{segment_index}")
            )
    # Canonical-value/family/source identity, preserving family order.
    unique: dict[str, tuple[Any, str, str, str]] = {}
    for item in candidates:
        key = canonical_sha256({"value": item[0], "family": item[1], "source_id": item[3]})
        unique.setdefault(key, item)
    family_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return sorted(unique.values(), key=lambda item: (family_index[item[1]], canonical_sha256(item)))


def _stratum(row: Mapping[str, Any], lookup: Mapping[tuple[str, int], Quadruple], multi: Mapping[str, bool]) -> tuple[Any, ...]:
    quad = lookup[(str(row["query_id"]), int(row["tuple_index"]))]
    return (
        row["field"],
        quad.hateful,
        quad.targeted_group,
        multi[str(row["query_id"])],
    )


def stratified_candidate_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    field_quota: Mapping[str, int],
    stable_hash_seed: int,
    quad_lookup: Mapping[tuple[str, int], Quadruple],
    multi_lookup: Mapping[str, bool],
) -> list[dict[str, Any]]:
    """Deterministic round-robin across frozen field/label/structure strata."""

    selected: list[dict[str, Any]] = []
    for field in REVIEW_FIELDS:
        quota = int(field_quota.get(field, 0))
        if quota < 0:
            raise CounterfactualError("field quota must be non-negative")
        groups: defaultdict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for raw in rows:
            if raw["field"] == field:
                row = dict(raw)
                groups[_stratum(row, quad_lookup, multi_lookup)].append(row)
        queues: dict[tuple[Any, ...], deque[dict[str, Any]]] = {}
        for key, values in groups.items():
            values.sort(
                key=lambda row: hashlib.sha256(
                    f"{stable_hash_seed}:{row['candidate_id']}".encode("utf-8")
                ).hexdigest()
            )
            queues[key] = deque(values)
        ordered_strata = sorted(groups, key=canonical_sha256)
        while len([row for row in selected if row["field"] == field]) < quota:
            progressed = False
            for key in ordered_strata:
                if queues[key]:
                    selected.append(queues[key].popleft())
                    progressed = True
                    if len([row for row in selected if row["field"] == field]) == quota:
                        break
            if not progressed:
                break
    selected.sort(key=lambda row: row["candidate_id"])
    return selected


def build_proposal_rows(
    *,
    cf_proposal_id: str,
    context_records: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
    field_quota: Mapping[str, int] | None = None,
    stable_hash_seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build all automatic label foils plus a ~400-row blind-review cohort."""

    if field_quota is None:
        field_quota = {"target": 200, "argument": 200}
    frequencies = group_combination_frequencies(train_records)
    automatic: list[dict[str, Any]] = []
    review_pool: list[dict[str, Any]] = []
    quad_lookup: dict[tuple[str, int], Quadruple] = {}
    multi_lookup: dict[str, bool] = {}
    for record in context_records:
        query_id, content, gold_quads, context_hash = _record_query(record)
        gold_text = serialize_quadruples(gold_quads)
        gold_hash = hashlib.sha256(gold_text.encode("utf-8")).hexdigest()
        multi_lookup[query_id] = len(gold_quads) > 1
        for tuple_index, quad in enumerate(gold_quads):
            quad_lookup[(query_id, tuple_index)] = quad
            automatic.append(
                _candidate_row(
                    cf_proposal_id=cf_proposal_id,
                    context_record_sha256=context_hash,
                    query_id=query_id,
                    query_content=content,
                    gold_sha256=gold_hash,
                    tuple_index=tuple_index,
                    field="hateful",
                    gold_value=quad.hateful,
                    candidate_value="non-hate" if quad.hateful == "hate" else "hate",
                    family="binary-label-flip",
                    source="deterministic-label-catalog",
                    source_id="hateful-binary-v1",
                    review_required=False,
                    gold_quads=gold_quads,
                )
            )
            automatic.append(
                _candidate_row(
                    cf_proposal_id=cf_proposal_id,
                    context_record_sha256=context_hash,
                    query_id=query_id,
                    query_content=content,
                    gold_sha256=gold_hash,
                    tuple_index=tuple_index,
                    field="targeted_group",
                    gold_value=list(quad.targeted_group),
                    candidate_value=_best_group_foil(quad.targeted_group, frequencies),
                    family="train-group-combination",
                    source="train-only-canonical-catalog",
                    source_id=canonical_sha256(dict(sorted(("|".join(key), value) for key, value in frequencies.items()))),
                    review_required=False,
                    gold_quads=gold_quads,
                )
            )
            for field in REVIEW_FIELDS:
                for value, family, source, source_id in _review_candidates_for_unit(
                    field=field,
                    tuple_index=tuple_index,
                    content=content,
                    gold_quads=gold_quads,
                ):
                    try:
                        row = _candidate_row(
                            cf_proposal_id=cf_proposal_id,
                            context_record_sha256=context_hash,
                            query_id=query_id,
                            query_content=content,
                            gold_sha256=gold_hash,
                            tuple_index=tuple_index,
                            field=field,
                            gold_value=getattr(quad, field),
                            candidate_value=value,
                            family=family,
                            source=source,
                            source_id=source_id,
                            review_required=True,
                            gold_quads=gold_quads,
                        )
                    except CounterfactualError:
                        continue
                    review_pool.append(row)
    sampled = stratified_candidate_sample(
        review_pool,
        field_quota=field_quota,
        stable_hash_seed=stable_hash_seed,
        quad_lookup=quad_lookup,
        multi_lookup=multi_lookup,
    )
    rows = [*automatic, *sampled]
    rows.sort(key=lambda row: row["candidate_id"])
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise CounterfactualError("candidate IDs are not unique")
    meta = {
        "schema_version": PROPOSAL_SUMMARY_SCHEMA,
        "cf_proposal_id": cf_proposal_id,
        "automatic_candidate_count": len(automatic),
        "review_candidate_count": len(sampled),
        "review_candidate_count_by_field": {
            field: sum(row["field"] == field for row in sampled) for field in REVIEW_FIELDS
        },
        "review_pool_count": len(review_pool),
        "query_count": len(context_records),
        "gold_tuple_count": len(automatic) // 2,
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
    }
    return rows, meta


def review_template(rows: Sequence[Mapping[str, Any]], *, reviewer_id: str = "") -> list[dict[str, Any]]:
    templates = []
    for row in rows:
        automatic = not bool(row["review_required"])
        templates.append(
            {
                "schema_version": REVIEW_ROW_SCHEMA,
                "cf_proposal_id": row["cf_proposal_id"],
                "candidate_id": row["candidate_id"],
                "decision": "not_required" if automatic else "",
                "reason_code": "deterministic-label-foil" if automatic else "",
                "note": "",
                "reviewer_id": reviewer_id,
            }
        )
    return templates


def finalize_rows(
    proposal_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    cf_build_id: str,
    review_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposals = {str(row["candidate_id"]): dict(row) for row in proposal_rows}
    reviews = {str(row.get("candidate_id")): dict(row) for row in review_rows}
    if len(proposals) != len(proposal_rows) or len(reviews) != len(review_rows):
        raise CounterfactualError("duplicate candidate ID in proposal/review")
    if set(proposals) != set(reviews):
        raise CounterfactualError("review candidate set must exactly equal proposal")
    accepted: defaultdict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for identifier, proposal in proposals.items():
        review = reviews[identifier]
        if set(review) != REVIEW_ROW_KEYS:
            raise CounterfactualError("review row has non-canonical fields")
        if review.get("schema_version") != REVIEW_ROW_SCHEMA:
            raise CounterfactualError("invalid review row schema")
        if review.get("cf_proposal_id") != proposal["cf_proposal_id"]:
            raise CounterfactualError("review proposal ID mismatch")
        decision = review.get("decision")
        if proposal["review_required"]:
            if decision not in {"pass", "reject"}:
                raise CounterfactualError("review-required candidate must be pass/reject")
            if decision == "pass" and review.get("reason_code") != "valid-local-foil":
                raise CounterfactualError("passed candidate has invalid reason code")
            if decision == "reject" and review.get("reason_code") not in {
                "equivalent-to-gold",
                "invalid-boundary-fragment",
                "unsupported-by-query",
                "changes-more-than-one-field",
                "malformed-candidate",
            }:
                raise CounterfactualError("rejected candidate has invalid reason code")
        else:
            if decision != "not_required" or review.get("reason_code") != "deterministic-label-foil":
                raise CounterfactualError("automatic candidate must use deterministic not_required decision")
        if decision in {"pass", "not_required"}:
            accepted[(proposal["query_id"], proposal["tuple_index"], proposal["field"])].append(proposal)

    family_index = {family: index for index, family in enumerate(FAMILY_ORDER)}
    # Every gold tuple contributes all four fields to construction coverage,
    # including target/argument units for which no valid local candidate could
    # be proposed.  Automatic group/hate candidates anchor every gold tuple.
    tuple_anchors: dict[tuple[str, int], dict[str, Any]] = {}
    for proposal in proposals.values():
        key = (str(proposal["query_id"]), int(proposal["tuple_index"]))
        existing = tuple_anchors.setdefault(key, proposal)
        for frozen_key in (
            "context_record_sha256",
            "gold_sha256",
            "gold_quadruples",
            "query_content",
        ):
            if proposal[frozen_key] != existing[frozen_key]:
                raise CounterfactualError(f"proposal rows disagree on {frozen_key}")
    final: list[dict[str, Any]] = []
    units = [(query_id, tuple_index, field) for query_id, tuple_index in tuple_anchors for field in FIELDS]
    for unit in sorted(units, key=lambda value: (int(value[0]), value[1], FIELDS.index(value[2]))):
        candidates = accepted.get(unit, [])
        selected = min(
            candidates,
            key=lambda row: (family_index.get(row["family"], -1), row["candidate_id"]),
        ) if candidates else None
        source_rows = [dict(row) for row in proposal_rows if (row["query_id"], row["tuple_index"], row["field"]) == unit]
        anchor = source_rows[0] if source_rows else tuple_anchors[(unit[0], unit[1])]
        gold_quads = canonicalize_quadruples(anchor["gold_quadruples"])
        gold_value = getattr(gold_quads[unit[1]], unit[2])
        if unit[2] == "targeted_group":
            gold_value = list(gold_value)
        row = {
            "schema_version": CF_ROW_SCHEMA,
            "cf_build_id": cf_build_id,
            "cf_proposal_id": anchor["cf_proposal_id"],
            "review_id": review_id,
            "context_record_sha256": anchor["context_record_sha256"],
            "query_id": unit[0],
            "gold_sha256": anchor["gold_sha256"],
            "tuple_index": unit[1],
            "field": unit[2],
            "gold_value": gold_value,
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "value": candidate["candidate_value"],
                    "family": candidate["family"],
                    "source": candidate["source"],
                    "source_id": candidate["source_id"],
                    "quality_status": "blind-review-pass" if candidate["review_required"] else "deterministic-validated",
                }
                for candidate in sorted(candidates, key=lambda item: item["candidate_id"])
            ],
            "selected_cf_id": selected["candidate_id"] if selected else None,
            "selection_policy": SELECTION_POLICY,
            "construction_status": "ok" if selected else "unavailable-no-passed-query-local-foil",
            "reference_model_sha256": None,
        }
        row["record_sha256"] = canonical_sha256(row)
        final.append(row)
    fields = {
        field: {
            "denominator": sum(row["field"] == field for row in final),
            "constructed": sum(row["field"] == field and row["construction_status"] == "ok" for row in final),
        }
        for field in FIELDS
    }
    for counts in fields.values():
        counts["coverage"] = counts["constructed"] / counts["denominator"] if counts["denominator"] else None
    return final, {
        "schema_version": "stage1-cf-summary/v1",
        "cf_build_id": cf_build_id,
        "review_id": review_id,
        "field_construction": fields,
        "group_hate_coverage_gate": fields["targeted_group"]["coverage"] == 1.0 and fields["hateful"]["coverage"] == 1.0,
    }


__all__ = [
    "CANDIDATE_POLICY_VERSION",
    "CF_ROW_SCHEMA",
    "CounterfactualError",
    "PROPOSAL_ROW_SCHEMA",
    "REVIEW_ROW_SCHEMA",
    "SELECTION_POLICY",
    "build_proposal_rows",
    "candidate_id",
    "canonical_value",
    "finalize_rows",
    "group_combination_frequencies",
    "review_template",
    "stratified_candidate_sample",
]
