"""Frozen Stage 1 PL/PD placebo controls.

The module has three deliberately separate responsibilities:

* deterministically match train-only low-relevance replacements against an
  already frozen context record and its catalogs;
* render PL/PD by changing exactly one evidence block in an existing frozen
  condition; and
* publish/validate an independent content-addressed control artifact.

It never opens prediction, evaluation, margin, or model-output artifacts.  The
only numeric scores it consumes are the written eight-decimal retrieval
similarities already frozen in the context record.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any

from data.context_manifest import (
    chat_prompt_text,
    text_sha256,
    token_count,
    validate_context_record,
)
from model.stage1_registry import (
    ModelRegistryError,
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
    verified_model_source_lease,
)


CONTROL_ARTIFACT_SCHEMA = "stage1-control/v1"
CONTROL_RECORD_SCHEMA = "stage1-control-record/v1"
CONTROL_META_SCHEMA = "stage1-control-manifest/v1"
CONTROL_PROVENANCE_SCHEMA = "stage1-control-provenance/v1"
CONTROL_CONFIG_SCHEMA = "stage1-control-config/v1"
CONTROL_POLICY = "placebo-match/v1"
LEXICAL_POLICY = "lexical-overlap-v1"
QUANTILE_POLICY = "nearest-rank-ceil-score-boundary/v1"
MATCHING_LOSS = "absolute-token-delta-then-similarity-sum-then-id-list/v1"
PAYLOAD_MANIFEST_SCHEMA = "stage1-payload-manifest/v1"
LOCATOR_REF_SCHEMA = "stage1-locator-ref/v1"
DEPENDENCY_REF_SCHEMA = "stage1-dependency-ref/v1"
CONTROL_ARTIFACT_KIND = "control"
TEST_CONTROL_ARTIFACT_KIND = "test-control"
FROZEN_CONTROL_POLICY_SCHEMA = "stage1-frozen-control-policy-ref/v1"
TOKENIZER_SOURCE_IDENTITY_SCHEMA = "stage1-control-tokenizer-source-identity/v1"
TOKENIZER_CONSTRUCTOR_POLICY = {
    "backend": "transformers-auto-tokenizer/v1",
    "local_files_only": True,
    "trust_remote_code": False,
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTROL_CONDITIONS = ("PL", "PD")
DEFAULT_TIERS = (10, 20, 30)
DEFAULT_MAX_TOKEN_DELTA = Decimal("0.01")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LEXICON_ID_RE = re.compile(r"^lex:v2:[0-9a-f]{64}$")
DEMO_ID_RE = re.compile(r"^demo:v1:[0-9a-f]{64}$")


class ControlManifestError(ValueError):
    """Raised when a control lifecycle or scientific invariant is violated."""


class ControlUnavailableError(ControlManifestError):
    """Raised when a formal dev control cannot satisfy its frozen constraints."""


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_written_similarity(value: Any) -> Decimal:
    """Return the frozen eight-place half-even relevance score."""

    try:
        score = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ControlManifestError(f"invalid retrieval similarity: {value!r}") from exc
    if not score.is_finite():
        raise ControlManifestError("retrieval similarity must be finite")
    return score.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN)


def lexical_normalize(text: str) -> str:
    """Implement ``lexical-overlap-v1`` exactly.

    NFKC is applied first, only ASCII A-Z is lower-cased, and all Unicode
    punctuation/separator code points are removed.  Other Unicode letters are
    intentionally left unchanged.
    """

    if not isinstance(text, str):
        raise ControlManifestError("lexical normalization requires text")
    normalized = unicodedata.normalize("NFKC", text)
    output: list[str] = []
    for character in normalized:
        if unicodedata.category(character)[0] in {"P", "Z"}:
            continue
        if "A" <= character <= "Z":
            character = chr(ord(character) + 32)
        output.append(character)
    return "".join(output)


def lexical_grams(text: str, width: int = 4) -> frozenset[str]:
    normalized = lexical_normalize(text)
    if not normalized:
        return frozenset()
    if len(normalized) < width:
        return frozenset({normalized})
    return frozenset(normalized[index : index + width] for index in range(len(normalized) - width + 1))


def lexicon_query_overlap(query: str, item: Mapping[str, Any]) -> bool:
    query_normalized = lexical_normalize(query)
    values = [item.get("term", ""), *(item.get("variants", []) or [])]
    for value in values:
        term = lexical_normalize(str(value))
        if not term:
            raise ControlManifestError("lexicon term/variant normalizes to empty text")
        if term in query_normalized:
            return True
    return False


def demo_query_overlap(query: str, item: Mapping[str, Any]) -> bool:
    query_normalized = lexical_normalize(query)
    demo_normalized = lexical_normalize(str(item.get("content", "")))
    if not query_normalized or not demo_normalized:
        return False
    if len(query_normalized) < 4 or len(demo_normalized) < 4:
        shorter, longer = sorted((query_normalized, demo_normalized), key=len)
        return shorter in longer
    return bool(lexical_grams(query) & lexical_grams(str(item.get("content", ""))))


_POST_OUTCOME_KEY_RE = re.compile(
    r"(?:prediction|evaluation|eval_result|metric|margin|model_score|model_output|generation_output)",
    re.IGNORECASE,
)


def _reject_post_outcome_fields(value: Any, *, where: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if _POST_OUTCOME_KEY_RE.search(key_text):
                raise ControlManifestError(
                    f"{where} contains forbidden post-outcome field at {path}.{key_text}"
                )
            _reject_post_outcome_fields(child, where=where, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_post_outcome_fields(child, where=where, path=f"{path}[{index}]")


_DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": CONTROL_CONFIG_SCHEMA,
    "policy_version": CONTROL_POLICY,
    "lexical_overlap_policy": LEXICAL_POLICY,
    "quantile_policy": QUANTILE_POLICY,
    "candidate_expansion_tiers_percent": list(DEFAULT_TIERS),
    "tier_boundary_comparator": "<=",
    "score_round_digits": 8,
    "score_rounding": "half-even",
    "low_similarity_threshold": None,
    "max_block_token_delta_fraction": 0.01,
    "matching_loss": MATCHING_LOSS,
    "replacement_order_policy": "target-quota-slot-then-stable-id/v1",
    "require_complete_train_pool_scores": True,
    "dev_nonempty_required": True,
    "max_exact_search_states": 1_000_000,
    "source_class_order": [
        "terminology",
        "non-hate",
        "Region",
        "Racism",
        "Sexism",
        "LGBTQ",
        "others",
    ],
}


def resolve_control_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve and validate a standalone or nested control configuration."""

    if not isinstance(config, Mapping):
        raise ControlManifestError("control config must be an object")
    _reject_post_outcome_fields(config, where="control config")
    section: Mapping[str, Any]
    nested = False
    if isinstance(config.get("controls"), Mapping):
        section = config["controls"]
        nested = True
    elif isinstance(config.get("placebo_controls"), Mapping):
        section = config["placebo_controls"]
        nested = True
    else:
        section = config
    standalone = config.get("schema_version") == CONTROL_CONFIG_SCHEMA
    if not nested and not standalone:
        section = {}
    resolved = copy.deepcopy(_DEFAULT_POLICY)
    allowed = set(_DEFAULT_POLICY) | {"tokenizer", "profile_name"}
    if nested or standalone:
        ignored_runtime_keys = {"artifact_root"} if not nested else set()
        unknown = set(section) - allowed - ignored_runtime_keys
        if unknown:
            raise ControlManifestError(f"unknown control config fields: {sorted(unknown)}")
    scientific = {key: value for key, value in section.items() if key in allowed}
    resolved.update(copy.deepcopy(scientific))

    if resolved.get("schema_version") != CONTROL_CONFIG_SCHEMA:
        raise ControlManifestError("unsupported control config schema")
    if resolved.get("policy_version") != CONTROL_POLICY:
        raise ControlManifestError("unsupported placebo matching policy")
    if resolved.get("lexical_overlap_policy") != LEXICAL_POLICY:
        raise ControlManifestError("unsupported lexical-overlap policy")
    if resolved.get("quantile_policy") != QUANTILE_POLICY:
        raise ControlManifestError("unsupported quantile policy")
    if resolved.get("tier_boundary_comparator") != "<=":
        raise ControlManifestError("bottom-tier boundary must use <=")
    if resolved.get("score_round_digits") != 8 or resolved.get("score_rounding") != "half-even":
        raise ControlManifestError("control relevance must use eight-place half-even scores")
    tiers = resolved.get("candidate_expansion_tiers_percent")
    if tiers != list(DEFAULT_TIERS):
        raise ControlManifestError("candidate tiers must be exactly [10, 20, 30]")
    try:
        max_delta = Decimal(str(resolved.get("max_block_token_delta_fraction")))
    except InvalidOperation as exc:
        raise ControlManifestError("invalid max block token delta") from exc
    if max_delta != DEFAULT_MAX_TOKEN_DELTA:
        raise ControlManifestError("maximum block token delta must be exactly 0.01")
    threshold = resolved.get("low_similarity_threshold")
    if threshold is not None:
        resolved["low_similarity_threshold"] = float(round_written_similarity(threshold))
    if resolved.get("matching_loss") != MATCHING_LOSS:
        raise ControlManifestError("unsupported matching loss")
    if resolved.get("replacement_order_policy") != "target-quota-slot-then-stable-id/v1":
        raise ControlManifestError("unsupported replacement order policy")
    if resolved.get("require_complete_train_pool_scores") is not True:
        raise ControlManifestError("formal control matching requires complete train-pool scores")
    if not isinstance(resolved.get("dev_nonempty_required"), bool):
        raise ControlManifestError("dev_nonempty_required must be boolean")
    classes = resolved.get("source_class_order")
    if not isinstance(classes, list) or not classes or any(not isinstance(value, str) or not value for value in classes):
        raise ControlManifestError("source_class_order must be a non-empty string array")
    if len(classes) != len(set(classes)):
        raise ControlManifestError("source_class_order contains duplicates")
    limit = resolved.get("max_exact_search_states")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ControlManifestError("max_exact_search_states must be a positive integer")
    tokenizer = resolved.get("tokenizer")
    if tokenizer is not None:
        if not isinstance(tokenizer, Mapping) or not isinstance(tokenizer.get("revision"), str):
            raise ControlManifestError("tokenizer config requires a revision")
        logical_path = tokenizer.get("logical_path", tokenizer.get("path"))
        if logical_path is not None and (not isinstance(logical_path, str) or Path(logical_path).is_absolute()):
            raise ControlManifestError("tokenizer path must be logical/repository-relative")
        resolved["tokenizer"] = {
            "revision": tokenizer["revision"],
            **({"logical_path": logical_path} if logical_path is not None else {}),
        }
    return resolved


def _item_id_key(kind: str) -> str:
    return "lexicon_id" if kind == "lexicon" else "demo_id"


def _id_pattern(kind: str) -> re.Pattern[str]:
    return LEXICON_ID_RE if kind == "lexicon" else DEMO_ID_RE


def _catalog_map(catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    key = _item_id_key(kind)
    if isinstance(catalog, Mapping):
        rows = []
        for item_id, raw in catalog.items():
            if not isinstance(raw, Mapping):
                raise ControlManifestError(f"{kind} catalog row must be an object")
            row = dict(raw)
            row.setdefault(key, str(item_id))
            rows.append(row)
    elif isinstance(catalog, Sequence) and not isinstance(catalog, (str, bytes)):
        rows = [dict(row) for row in catalog if isinstance(row, Mapping)]
        if len(rows) != len(catalog):
            raise ControlManifestError(f"{kind} catalog contains a non-object row")
    else:
        raise ControlManifestError(f"{kind} catalog must be an object or array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get(key)
        if not isinstance(item_id, str) or not _id_pattern(kind).fullmatch(item_id):
            raise ControlManifestError(f"invalid {kind} unit ID: {item_id!r}")
        if item_id in result:
            raise ControlManifestError(f"duplicate {kind} catalog ID: {item_id}")
        block = row.get("rendered_block")
        if not isinstance(block, str) or not block:
            raise ControlManifestError(f"{kind} {item_id} lacks rendered_block")
        block_hash = text_sha256(block)
        if row.get("rendered_block_sha256") not in {None, block_hash}:
            raise ControlManifestError(f"{kind} {item_id} rendered block hash mismatch")
        row["rendered_block_sha256"] = block_hash
        source_split = row.get("source_split", row.get("split"))
        if source_split is not None and source_split != "train":
            raise ControlManifestError(f"{kind} placebo catalog contains non-train row {item_id}")
        if row.get("train_only") is False:
            raise ControlManifestError(f"{kind} placebo catalog row is not train-only: {item_id}")
        if kind == "lexicon":
            if not isinstance(row.get("term"), str) or not row.get("term"):
                raise ControlManifestError(f"lexicon {item_id} lacks term")
            if "category" in row or "categories" in row:
                raise ControlManifestError(
                    f"terminology evidence {item_id} contains a task category"
                )
            if row.get("evidence_kind") != "terminology":
                raise ControlManifestError(
                    f"lexicon {item_id} is not terminology evidence"
                )
            variants = row.get("variants", []) or []
            if not isinstance(variants, list) or any(not isinstance(value, str) for value in variants):
                raise ControlManifestError(f"lexicon {item_id} variants must be strings")
            row["variants"] = variants
            row["quota_class"] = "terminology"
            row["content_sha256"] = str(row.get("content_sha256", block_hash))
        else:
            if not isinstance(row.get("source_record_id"), str) or not row.get("source_record_id"):
                raise ControlManifestError(f"demo {item_id} lacks source_record_id")
            if not isinstance(row.get("content"), str) or not row.get("content"):
                raise ControlManifestError(f"demo {item_id} lacks content")
            content_hash = text_sha256(row["content"])
            if row.get("content_sha256") not in {None, content_hash}:
                raise ControlManifestError(f"demo {item_id} content hash mismatch")
            row["content_sha256"] = content_hash
            quota_class = row.get("output_label", row.get("source_class", row.get("quota_class")))
            if not isinstance(quota_class, str) or not quota_class:
                raise ControlManifestError(f"demo {item_id} lacks output-label quota class")
            row["quota_class"] = quota_class
        if not SHA256_RE.fullmatch(row["content_sha256"]):
            raise ControlManifestError(f"{kind} {item_id} has invalid content hash")
        result[item_id] = row
    return result


def _selected_ids(context_record: Mapping[str, Any], kind: str) -> list[str]:
    key = "lexicons" if kind == "lexicon" else "demos"
    value = context_record.get("selection", {}).get(key, {}).get("prompt_order_final")
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ControlManifestError(f"context record lacks final {kind} order")
    if len(value) != len(set(value)):
        raise ControlManifestError(f"context final {kind} IDs are duplicated")
    return list(value)


def _relevance_rows(context_record: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    control = context_record.get("control_relevance")
    aliases = ("lexicons", "lexicon_candidates") if kind == "lexicon" else ("demos", "demo_candidates")
    if isinstance(control, Mapping):
        for alias in aliases:
            if isinstance(control.get(alias), list):
                return control[alias]
    retrieval = context_record.get("retrieval")
    if isinstance(retrieval, Mapping):
        keys = (
            ("control_lexicon_candidates", "lexicon_control_candidates", "lexicon_candidates")
            if kind == "lexicon"
            else ("control_demo_candidates", "demo_control_candidates", "demo_candidates")
        )
        for key in keys:
            if isinstance(retrieval.get(key), list):
                return retrieval[key]
    raise ControlManifestError(f"context record lacks frozen all-pool {kind} relevance rows")


def _extract_score(row: Mapping[str, Any]) -> Decimal:
    for key in ("written_similarity", "similarity", "selection_score"):
        if row.get(key) is not None:
            return round_written_similarity(row[key])
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        scores = [
            round_written_similarity(item["written_similarity"])
            for item in evidence
            if isinstance(item, Mapping) and item.get("written_similarity") is not None
        ]
        if scores:
            return max(scores)
    raise ControlManifestError("control relevance row lacks a written cosine score")


def _scored_pool(
    context_record: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]], kind: str
) -> dict[str, dict[str, Any]]:
    key = _item_id_key(kind)
    result: dict[str, dict[str, Any]] = {}
    for raw in _relevance_rows(context_record, kind):
        if not isinstance(raw, Mapping):
            raise ControlManifestError(f"{kind} relevance row must be an object")
        item_id = raw.get(key)
        if not isinstance(item_id, str) or item_id not in catalog:
            raise ControlManifestError(f"{kind} relevance row refers to unknown train unit {item_id!r}")
        if item_id in result:
            raise ControlManifestError(f"duplicate {kind} relevance row: {item_id}")
        item = catalog[item_id]
        source_class = raw.get("source_class", item["quota_class"])
        if source_class != item["quota_class"]:
            raise ControlManifestError(f"{kind} relevance class disagrees with catalog: {item_id}")
        result[item_id] = {
            "item_id": item_id,
            "source_class": source_class,
            "written_similarity": _extract_score(raw),
            "catalog": item,
        }
    if set(result) != set(catalog):
        missing = sorted(set(catalog) - set(result))
        extra = sorted(set(result) - set(catalog))
        raise ControlManifestError(
            f"{kind} relevance is not the complete train catalog: missing={missing[:5]} extra={extra[:5]}"
        )
    return result


def _class_rankings(
    scored_pool: Mapping[str, Mapping[str, Any]], class_order: Sequence[str]
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, int]]:
    known = set(class_order)
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in class_order}
    for candidate in scored_pool.values():
        source_class = candidate["source_class"]
        if source_class not in known:
            raise ControlManifestError(f"candidate class absent from source_class_order: {source_class}")
        grouped[source_class].append(candidate)
    ranks: dict[str, int] = {}
    for source_class in class_order:
        grouped[source_class].sort(key=lambda item: (item["written_similarity"], item["item_id"]))
        ranks.update({item["item_id"]: rank for rank, item in enumerate(grouped[source_class])})
    return grouped, ranks


def _tier_membership(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], tier_percent: int
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    included: set[str] = set()
    boundaries: dict[str, dict[str, Any]] = {}
    for source_class, rows in grouped.items():
        if not rows:
            boundaries[source_class] = {
                "pool_count": 0,
                "nearest_rank_count": 0,
                "included_count": 0,
                "boundary_written_similarity": None,
                "boundary_unit_id": None,
            }
            continue
        count = max(1, math.ceil(len(rows) * tier_percent / 100))
        boundary = rows[count - 1]
        boundary_score = boundary["written_similarity"]
        chosen = [
            row
            for row in rows
            if row["written_similarity"] <= boundary_score
        ]
        included.update(row["item_id"] for row in chosen)
        boundaries[source_class] = {
            "pool_count": len(rows),
            "nearest_rank_count": count,
            "included_count": len(chosen),
            "boundary_written_similarity": float(boundary["written_similarity"]),
            "boundary_unit_id": boundary["item_id"],
        }
    return included, boundaries


def _block_text(ids: Sequence[str], catalog: Mapping[str, Mapping[str, Any]]) -> str:
    return "\n\n".join(catalog[item_id]["rendered_block"] for item_id in ids)


def _token_ratio(replacement_tokens: int, target_tokens: int) -> Decimal:
    return Decimal(abs(replacement_tokens - target_tokens)) / Decimal(max(target_tokens, 1))


def _batched_token_counts(texts: Sequence[str], tokenizer: Any) -> list[int]:
    """Tokenize an exact candidate batch without padding or truncation."""

    if not texts:
        return []
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                list(texts),
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            if len(encoded) == len(texts) and all(hasattr(row, "__len__") for row in encoded):
                return [len(row) for row in encoded]
        except (KeyError, TypeError, ValueError):
            pass
    return [token_count(text, tokenizer) for text in texts]


def _prepare_token_costs(
    catalog: Mapping[str, Mapping[str, Any]], tokenizer: Any
) -> dict[str, tuple[int, int]]:
    """Prove and cache exact ``block + \n\n`` token composition costs.

    Qwen's pre-tokenizer isolates the double-newline boundary, but the newline
    may merge with the preceding punctuation token.  Therefore non-final
    blocks use ``encode(block + separator)`` while the final block uses
    ``encode(block)``.  The two star checks fail closed if a tokenizer/catalog
    does not obey this separable composition contract.
    """

    item_ids = sorted(catalog)
    if not item_ids:
        return {}
    blocks = [str(catalog[item_id]["rendered_block"]) for item_id in item_ids]
    separator = "\n\n"
    standalone = _batched_token_counts(blocks, tokenizer)
    with_separator = _batched_token_counts(
        [block + separator for block in blocks], tokenizer
    )
    representative = blocks[0]
    representative_standalone = standalone[0]
    representative_with_separator = with_separator[0]
    left_star = _batched_token_counts(
        [block + separator + representative for block in blocks], tokenizer
    )
    right_star = _batched_token_counts(
        [representative + separator + block for block in blocks], tokenizer
    )
    for index, item_id in enumerate(item_ids):
        if left_star[index] != with_separator[index] + representative_standalone:
            raise ControlManifestError(
                f"tokenizer separator composition fails after catalog item {item_id}"
            )
        if right_star[index] != representative_with_separator + standalone[index]:
            raise ControlManifestError(
                f"tokenizer separator composition fails before catalog item {item_id}"
            )
    return {
        item_id: (standalone[index], with_separator[index])
        for index, item_id in enumerate(item_ids)
    }


def _ordered_slot_ids(
    selected_by_class: Mapping[str, Sequence[str]],
    target_class_slots: Sequence[str],
) -> list[str]:
    offsets: Counter[str] = Counter()
    result: list[str] = []
    for source_class in target_class_slots:
        values = selected_by_class.get(source_class)
        if values is None:
            continue
        offset = offsets[source_class]
        if offset >= len(values):
            raise ControlManifestError("control DP class assignment is incomplete")
        result.append(values[offset])
        offsets[source_class] += 1
    return result


def _search_match_dp(
    *,
    by_class: Mapping[str, Sequence[Mapping[str, Any]]],
    quotas: Mapping[str, int],
    catalog: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    token_costs: Mapping[str, tuple[int, int]],
    target_tokens: int,
    max_delta: Decimal,
    max_states: int,
    class_order: Sequence[str],
    target_class_slots: Sequence[str],
    search_space: int,
) -> dict[str, Any]:
    """Exact quota/token optimization via bounded token-sum dynamic programming."""

    candidate_hashes = [
        candidate["catalog"]["content_sha256"]
        for source_class in class_order
        for candidate in by_class[source_class]
        if quotas.get(source_class, 0)
    ]
    if len(candidate_hashes) != len(set(candidate_hashes)):
        return {
            "status": "duplicate-content-search-space",
            "search_space_upper_bound": search_space,
            "evaluated_states": 0,
            "best_observed_token_delta": None,
        }
    allowed_delta = int(max_delta * Decimal(max(target_tokens, 1)))
    lower = max(0, target_tokens - allowed_delta)
    upper = target_tokens + allowed_delta
    positions = {
        source_class: [
            index
            for index, slot_class in enumerate(target_class_slots)
            if slot_class == source_class
        ]
        for source_class in class_order
    }
    class_states: dict[str, list[tuple[int, Decimal, tuple[str, ...]]]] = {}
    peak_states = 1
    for source_class in class_order:
        quota = int(quotas.get(source_class, 0))
        if quota == 0:
            continue
        # Keep one sparse token-sum table per selected-item count.  Updating
        # counts in descending order preserves 0/1 semantics while avoiding a
        # full copy of every reachable state for every catalog candidate.  A
        # flat ``updated = dict(states)`` is asymptotically equivalent, but is
        # prohibitively expensive for the real 5.7k-demo train catalog.
        states_by_count: list[dict[int, tuple[Decimal, tuple[str, ...]]]] = [
            {} for _ in range(quota + 1)
        ]
        states_by_count[0][0] = (Decimal("0"), ())
        for candidate_index, candidate in enumerate(by_class[source_class]):
            highest_source_count = min(quota - 1, candidate_index)
            for count in range(highest_source_count, -1, -1):
                item_id = candidate["item_id"]
                standalone_cost, separator_cost = token_costs[item_id]
                position = positions[source_class][count]
                cost = (
                    standalone_cost
                    if position == len(target_class_slots) - 1
                    else separator_cost
                )
                destination = states_by_count[count + 1]
                for total, (similarity_sum, ids) in states_by_count[count].items():
                    new_total = total + cost
                    if new_total > upper:
                        continue
                    value = (
                        similarity_sum + candidate["written_similarity"],
                        (*ids, item_id),
                    )
                    previous = destination.get(new_total)
                    if previous is None or value < previous:
                        destination[new_total] = value
            state_count = sum(len(values) for values in states_by_count)
            peak_states = max(peak_states, state_count)
            if peak_states > max_states:
                return {
                    "status": "search-space-limit",
                    "search_space_upper_bound": search_space,
                    "evaluated_states": peak_states,
                    "best_observed_token_delta": None,
                }
        terminal = [
            (total, similarity_sum, ids)
            for total, (similarity_sum, ids) in states_by_count[quota].items()
        ]
        if not terminal:
            return {
                "status": "no-token-match",
                "search_space_upper_bound": search_space,
                "evaluated_states": peak_states,
                "best_observed_token_delta": None,
            }
        class_states[source_class] = terminal

    combined: dict[int, tuple[Decimal, dict[str, tuple[str, ...]]]] = {
        0: (Decimal("0"), {})
    }
    for source_class in class_order:
        if source_class not in class_states:
            continue
        updated_combined: dict[int, tuple[Decimal, dict[str, tuple[str, ...]]]] = {}
        for base_total, (base_similarity, base_selection) in combined.items():
            for class_total, class_similarity, class_ids in class_states[source_class]:
                total = base_total + class_total
                if total > upper:
                    continue
                selection = {**base_selection, source_class: class_ids}
                value = (base_similarity + class_similarity, selection)
                previous = updated_combined.get(total)
                if previous is None:
                    updated_combined[total] = value
                    continue
                previous_key = (
                    previous[0],
                    tuple(_ordered_slot_ids(previous[1], target_class_slots)),
                )
                value_key = (
                    value[0],
                    tuple(_ordered_slot_ids(value[1], target_class_slots)),
                )
                if value_key < previous_key:
                    updated_combined[total] = value
        combined = updated_combined
        peak_states = max(peak_states, len(combined))
        if not combined or peak_states > max_states:
            return {
                "status": "search-space-limit" if peak_states > max_states else "no-token-match",
                "search_space_upper_bound": search_space,
                "evaluated_states": peak_states,
                "best_observed_token_delta": None,
            }

    nearest_delta = min((abs(total - target_tokens) for total in combined), default=None)
    eligible: list[tuple[tuple[Any, ...], list[str], int, Decimal]] = []
    for total, (similarity_sum, selection) in combined.items():
        if total < lower or _token_ratio(total, target_tokens) > max_delta:
            continue
        ids = _ordered_slot_ids(selection, target_class_slots)
        loss = (abs(total - target_tokens), similarity_sum, tuple(ids))
        eligible.append((loss, ids, total, similarity_sum))
    if not eligible:
        return {
            "status": "no-token-match",
            "search_space_upper_bound": search_space,
            "evaluated_states": peak_states,
            "best_observed_token_delta": nearest_delta,
        }
    _, ids, replacement_tokens, similarity_sum = min(eligible, key=lambda row: row[0])
    actual_tokens = token_count(_block_text(ids, catalog), tokenizer)
    if actual_tokens != replacement_tokens:
        raise ControlManifestError(
            "tokenizer separator composition changed during exact control replay"
        )
    return {
        "status": "ok",
        "replacement_ids": ids,
        "replacement_tokens": replacement_tokens,
        "similarity_sum": float(similarity_sum),
        "search_space_upper_bound": search_space,
        "evaluated_states": peak_states,
        "best_observed_token_delta": nearest_delta,
    }


def _search_match(
    *,
    candidates: Sequence[Mapping[str, Any]],
    quotas: Mapping[str, int],
    catalog: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    token_costs: Mapping[str, tuple[int, int]],
    target_tokens: int,
    max_delta: Decimal,
    max_states: int,
    class_order: Sequence[str],
    target_class_slots: Sequence[str],
) -> dict[str, Any]:
    by_class: dict[str, list[Mapping[str, Any]]] = {name: [] for name in class_order}
    for candidate in candidates:
        by_class[candidate["source_class"]].append(candidate)
    active_classes = [name for name in class_order if quotas.get(name, 0)]
    for name in active_classes:
        by_class[name].sort(key=lambda item: item["item_id"])
        if len(by_class[name]) < quotas[name]:
            return {
                "status": "no-class-quota-match",
                "search_space_upper_bound": 0,
                "evaluated_states": 0,
                "best_observed_token_delta": None,
            }
    search_space = 1
    for name in active_classes:
        search_space *= math.comb(len(by_class[name]), quotas[name])
    if search_space > max_states:
        return _search_match_dp(
            by_class=by_class,
            quotas=quotas,
            catalog=catalog,
            tokenizer=tokenizer,
            token_costs=token_costs,
            target_tokens=target_tokens,
            max_delta=max_delta,
            max_states=max_states,
            class_order=class_order,
            target_class_slots=target_class_slots,
            search_space=search_space,
        )

    best: tuple[tuple[Any, ...], list[str], int, Decimal] | None = None
    nearest_delta: int | None = None
    evaluated = 0
    pending: list[tuple[list[str], Decimal]] = []

    def flush_pending() -> None:
        nonlocal best, nearest_delta
        if not pending:
            return
        texts = [_block_text(ids, catalog) for ids, _ in pending]
        counts = _batched_token_counts(texts, tokenizer)
        if len(counts) != len(pending):
            raise ControlManifestError("tokenizer returned the wrong control-candidate batch size")
        for (ids, similarity_sum), replacement_tokens in zip(pending, counts, strict=True):
            delta = abs(replacement_tokens - target_tokens)
            nearest_delta = delta if nearest_delta is None else min(nearest_delta, delta)
            ratio = _token_ratio(replacement_tokens, target_tokens)
            if ratio > max_delta:
                continue
            loss = (delta, similarity_sum, tuple(ids))
            if best is None or loss < best[0]:
                best = (loss, ids, replacement_tokens, similarity_sum)
        pending.clear()

    def visit(class_index: int, selected: list[Mapping[str, Any]]) -> None:
        nonlocal best, nearest_delta, evaluated
        if class_index == len(active_classes):
            evaluated += 1
            content_hashes = [candidate["catalog"]["content_sha256"] for candidate in selected]
            if len(content_hashes) != len(set(content_hashes)):
                return
            selected_by_class: dict[str, list[str]] = {name: [] for name in class_order}
            for candidate in selected:
                selected_by_class[candidate["source_class"]].append(candidate["item_id"])
            for ids_for_class in selected_by_class.values():
                ids_for_class.sort()
            ids = [selected_by_class[source_class].pop(0) for source_class in target_class_slots]
            similarity_sum = sum(
                (candidate["written_similarity"] for candidate in selected), Decimal("0")
            )
            pending.append((ids, similarity_sum))
            if len(pending) >= 2048:
                flush_pending()
            return
        source_class = active_classes[class_index]
        quota = quotas[source_class]
        for combination in itertools.combinations(by_class[source_class], quota):
            visit(class_index + 1, [*selected, *combination])

    visit(0, [])
    flush_pending()
    base = {
        "search_space_upper_bound": search_space,
        "evaluated_states": evaluated,
        "best_observed_token_delta": nearest_delta,
    }
    if best is None:
        return {"status": "no-token-match", **base}
    _, ids, replacement_tokens, similarity_sum = best
    return {
        "status": "ok",
        "replacement_ids": ids,
        "replacement_tokens": replacement_tokens,
        "similarity_sum": float(similarity_sum),
        **base,
    }


def _target_classes(
    context_record: Mapping[str, Any],
    target_ids: Sequence[str],
    catalog: Mapping[str, Mapping[str, Any]],
    kind: str,
) -> tuple[list[str], str]:
    if kind == "lexicon":
        return [catalog[item_id]["quota_class"] for item_id in target_ids], "catalog-evidence-kind"
    assignments = context_record.get("selection", {}).get("demos", {}).get("quota_assignments")
    if isinstance(assignments, list):
        by_id: dict[str, str] = {}
        for row in assignments:
            if not isinstance(row, Mapping):
                raise ControlManifestError("demo quota assignment must be an object")
            item_id = row.get("demo_id")
            quota_class = row.get("assigned_quota_class")
            if item_id in by_id or not isinstance(item_id, str) or not isinstance(quota_class, str):
                raise ControlManifestError("invalid or duplicate demo quota assignment")
            by_id[item_id] = quota_class
        if all(item_id in by_id for item_id in target_ids):
            return [by_id[item_id] for item_id in target_ids], "context-assigned-quota-class"
        if target_ids:
            raise ControlManifestError("demo quota assignments do not cover every final demo")
    return [catalog[item_id]["quota_class"] for item_id in target_ids], "catalog-output-label"


def _empty_shape(
    *,
    target_ids: Sequence[str],
    target_block: str,
    target_tokens: int,
    quotas: Mapping[str, int],
    quota_source: str,
) -> dict[str, Any]:
    return {
        "target_ids": list(target_ids),
        "target_quota": dict(quotas),
        "quota_source": quota_source,
        "candidate_evidence": [],
        "tier_attempts": [],
        "used_tier_percent": None,
        "replacement_ids": [],
        "replacement_order": [],
        "target_rendered_block_sha256": text_sha256(target_block),
        "rendered_block_sha256": text_sha256(""),
        "target_block_tokens": target_tokens,
        "replacement_block_tokens": 0,
        "token_delta_ratio": 0.0,
        "similarity_sum": 0.0,
        "status": "empty",
        "failure_reason": None,
        "effective_treatment": False,
        "degenerate_reason": "source-empty",
    }


def _build_control_shape(
    context_record: Mapping[str, Any],
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    kind: str,
    tokenizer: Any,
    token_costs: Mapping[str, tuple[int, int]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    target_ids = _selected_ids(context_record, kind)
    unknown = [item_id for item_id in target_ids if item_id not in catalog]
    if unknown:
        raise ControlManifestError(f"frozen {kind} target IDs are absent from catalog: {unknown}")
    target_hashes = [catalog[item_id]["content_sha256"] for item_id in target_ids]
    if len(target_hashes) != len(set(target_hashes)):
        raise ControlManifestError(f"frozen {kind} target contains duplicate content")
    target_classes, quota_source = _target_classes(context_record, target_ids, catalog, kind)
    class_order = list(config["source_class_order"])
    unknown_classes = sorted(set(target_classes) - set(class_order))
    if unknown_classes:
        raise ControlManifestError(f"target quota classes absent from source_class_order: {unknown_classes}")
    quotas = Counter(target_classes)
    ordered_quotas = {name: quotas[name] for name in class_order if quotas[name]}
    target_block = _block_text(target_ids, catalog)
    target_tokens = token_count(target_block, tokenizer)
    if not target_ids:
        return _empty_shape(
            target_ids=target_ids,
            target_block=target_block,
            target_tokens=target_tokens,
            quotas=ordered_quotas,
            quota_source=quota_source,
        )

    scored = _scored_pool(context_record, catalog, kind)
    grouped, class_ranks = _class_rankings(scored, class_order)
    query = context_record.get("query", {})
    query_id = str(query.get("id", ""))
    query_content = query.get("content")
    if not isinstance(query_content, str) or not query_content:
        raise ControlManifestError("context query content is required for lexical controls")
    query_content_hash = query.get("content_sha256", text_sha256(query_content))
    if query_content_hash != text_sha256(query_content):
        raise ControlManifestError("context query content hash mismatch")
    target_id_set = set(target_ids)
    target_hash_set = set(target_hashes)
    threshold = config.get("low_similarity_threshold")
    threshold_decimal = round_written_similarity(threshold) if threshold is not None else None

    all_tier_ids: dict[int, set[str]] = {}
    all_boundaries: dict[int, dict[str, dict[str, Any]]] = {}
    for tier in config["candidate_expansion_tiers_percent"]:
        included, boundaries = _tier_membership(grouped, tier)
        all_tier_ids[tier] = included
        all_boundaries[tier] = boundaries

    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item_id, candidate in scored.items():
        item = candidate["catalog"]
        lexical_overlap = (
            lexicon_query_overlap(query_content, item)
            if kind == "lexicon"
            else demo_query_overlap(query_content, item)
        )
        exclusions: list[str] = []
        if item_id in target_id_set:
            exclusions.append("target-unit-id")
        if item["content_sha256"] in target_hash_set:
            exclusions.append("target-content-sha256")
        if lexical_overlap:
            exclusions.append("query-lexical-overlap")
        if threshold_decimal is not None and candidate["written_similarity"] > threshold_decimal:
            exclusions.append("above-low-similarity-threshold")
        if kind == "demo":
            if item["source_record_id"] == query_id:
                exclusions.append("query-source-record-id")
            if item["content_sha256"] == query_content_hash:
                exclusions.append("query-content-sha256")
        eligible_tiers = [
            tier
            for tier in config["candidate_expansion_tiers_percent"]
            if item_id in all_tier_ids[tier] and not exclusions
        ]
        evidence = {
            _item_id_key(kind): item_id,
            "source_class": candidate["source_class"],
            "written_similarity": float(candidate["written_similarity"]),
            "class_rank": class_ranks[item_id],
            "class_pool_count": len(grouped[candidate["source_class"]]),
            "lexical_overlap": lexical_overlap,
            "content_sha256": item["content_sha256"],
            "eligible_tiers_percent": eligible_tiers,
            "exclusion_reasons": exclusions,
            "selected": False,
        }
        if kind == "demo":
            evidence["source_record_id"] = item["source_record_id"]
        evidence_by_id[item_id] = evidence

    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_tier: int | None = None
    last_failure = "pre-registered-tiers-exhausted"
    for tier in config["candidate_expansion_tiers_percent"]:
        eligible = [
            candidate
            for item_id, candidate in scored.items()
            if tier in evidence_by_id[item_id]["eligible_tiers_percent"]
            and candidate["source_class"] in quotas
        ]
        outcome = _search_match(
            candidates=eligible,
            quotas=ordered_quotas,
            catalog=catalog,
            tokenizer=tokenizer,
            token_costs=token_costs,
            target_tokens=target_tokens,
            max_delta=Decimal(str(config["max_block_token_delta_fraction"])),
            max_states=config["max_exact_search_states"],
            class_order=class_order,
            target_class_slots=target_classes,
        )
        attempt = {
            "tier_percent": tier,
            "class_boundaries": all_boundaries[tier],
            "eligible_candidate_count": len(eligible),
            **outcome,
        }
        attempts.append(attempt)
        if outcome["status"] == "ok":
            selected = outcome
            selected_tier = tier
            break
        last_failure = str(outcome["status"])

    candidate_evidence = sorted(
        evidence_by_id.values(),
        key=lambda row: (
            class_order.index(row["source_class"]),
            Decimal(str(row["written_similarity"])),
            row[_item_id_key(kind)],
        ),
    )
    if selected is None:
        return {
            "target_ids": target_ids,
            "target_quota": ordered_quotas,
            "quota_source": quota_source,
            "candidate_evidence": candidate_evidence,
            "tier_attempts": attempts,
            "used_tier_percent": None,
            "replacement_ids": [],
            "replacement_order": [],
            "target_rendered_block_sha256": text_sha256(target_block),
            "rendered_block_sha256": text_sha256(""),
            "target_block_tokens": target_tokens,
            "replacement_block_tokens": 0,
            "token_delta_ratio": None,
            "similarity_sum": None,
            "status": "unavailable",
            "failure_reason": last_failure,
            "effective_treatment": False,
            "degenerate_reason": "no-pre-registered-match",
        }
    replacement_ids = list(selected["replacement_ids"])
    replacement_hashes = [catalog[item_id]["content_sha256"] for item_id in replacement_ids]
    if set(replacement_ids) & target_id_set or set(replacement_hashes) & target_hash_set:
        raise ControlManifestError("placebo replacement overlaps its target")
    if len(replacement_ids) != len(set(replacement_ids)) or len(replacement_hashes) != len(set(replacement_hashes)):
        raise ControlManifestError("placebo replacement contains duplicate unit/content")
    replacement_block = _block_text(replacement_ids, catalog)
    replacement_tokens = int(selected["replacement_tokens"])
    ratio = _token_ratio(replacement_tokens, target_tokens)
    if ratio > DEFAULT_MAX_TOKEN_DELTA:
        raise ControlManifestError("selected placebo exceeds the frozen token tolerance")
    selected_set = set(replacement_ids)
    for row in candidate_evidence:
        row["selected"] = row[_item_id_key(kind)] in selected_set
    return {
        "target_ids": target_ids,
        "target_quota": ordered_quotas,
        "quota_source": quota_source,
        "candidate_evidence": candidate_evidence,
        "tier_attempts": attempts,
        "used_tier_percent": selected_tier,
        "replacement_ids": replacement_ids,
        "replacement_order": replacement_ids,
        "target_rendered_block_sha256": text_sha256(target_block),
        "rendered_block_sha256": text_sha256(replacement_block),
        "target_block_tokens": target_tokens,
        "replacement_block_tokens": replacement_tokens,
        "token_delta_ratio": float(ratio),
        "similarity_sum": selected["similarity_sum"],
        "status": "ok",
        "failure_reason": None,
        "effective_treatment": True,
        "degenerate_reason": None,
    }


def _build_control_record_unchecked(
    context_record: Mapping[str, Any],
    *,
    lexicon_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    demo_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tokenizer: Any,
    resolved_config: Mapping[str, Any],
    control_build_id: str,
    token_costs: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    validate_context_record(context_record)
    _reject_post_outcome_fields(context_record, where="context record")
    if not re.fullmatch(r"ctl-[0-9a-f]{64}", control_build_id):
        raise ControlManifestError("invalid control build ID")
    lexicons = _catalog_map(lexicon_catalog, "lexicon")
    demos = _catalog_map(demo_catalog, "demo")
    costs = dict(token_costs or {})
    if "lexicon" not in costs:
        costs["lexicon"] = _prepare_token_costs(lexicons, tokenizer)
    if "demo" not in costs:
        costs["demo"] = _prepare_token_costs(demos, tokenizer)
    query = context_record.get("query", {})
    query_id = query.get("id")
    if not isinstance(query_id, str) or not query_id:
        raise ControlManifestError("context query lacks canonical ID")
    record = {
        "schema_version": CONTROL_RECORD_SCHEMA,
        "control_build_id": control_build_id,
        "context_build_id": context_record.get("context_build_id"),
        "context_record_sha256": context_record.get("record_sha256"),
        "query_id": query_id,
        "PL": _build_control_shape(
            context_record,
            catalog=lexicons,
            kind="lexicon",
            tokenizer=tokenizer,
            token_costs=costs["lexicon"],
            config=resolved_config,
        ),
        "PD": _build_control_shape(
            context_record,
            catalog=demos,
            kind="demo",
            tokenizer=tokenizer,
            token_costs=costs["demo"],
            config=resolved_config,
        ),
    }
    record["availability_mask"] = {
        condition: record[condition]["status"] != "unavailable"
        for condition in CONTROL_CONDITIONS
    }
    record["nondegenerate_mask"] = {
        condition: record[condition]["status"] == "ok"
        for condition in CONTROL_CONDITIONS
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def build_control_record(
    context_record: Mapping[str, Any],
    *,
    lexicon_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    demo_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Mapping[str, Any],
    control_build_id: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic PL/PD record without doing any file I/O."""

    resolved = resolve_control_config(config)
    if control_build_id is None:
        control_build_id = "ctl-" + canonical_sha256(
            {
                "mode": "direct-record",
                "context_record_sha256": context_record.get("record_sha256"),
                "resolved_config": resolved,
                "lexicon_catalog_sha256": canonical_sha256(_catalog_map(lexicon_catalog, "lexicon")),
                "demo_catalog_sha256": canonical_sha256(_catalog_map(demo_catalog, "demo")),
            }
        )
    return _build_control_record_unchecked(
        context_record,
        lexicon_catalog=lexicon_catalog,
        demo_catalog=demo_catalog,
        tokenizer=tokenizer,
        resolved_config=resolved,
        control_build_id=control_build_id,
    )


def validate_control_record(
    record: Mapping[str, Any],
    context_record: Mapping[str, Any],
    *,
    lexicon_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    demo_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tokenizer: Any,
    config: Mapping[str, Any],
    allow_unavailable: bool = True,
    token_costs: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
) -> None:
    """Fully recompute and byte-compare one frozen control record."""

    if record.get("schema_version") != CONTROL_RECORD_SCHEMA:
        raise ControlManifestError("wrong control record schema")
    recorded_hash = record.get("record_sha256")
    if not isinstance(recorded_hash, str) or recorded_hash != canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    ):
        raise ControlManifestError("control record hash mismatch")
    expected = _build_control_record_unchecked(
        context_record,
        lexicon_catalog=lexicon_catalog,
        demo_catalog=demo_catalog,
        tokenizer=tokenizer,
        resolved_config=resolve_control_config(config),
        control_build_id=str(record.get("control_build_id", "")),
        token_costs=token_costs,
    )
    if canonical_json_bytes(record) != canonical_json_bytes(expected):
        raise ControlManifestError("control record cannot be reproduced from frozen inputs")
    if not allow_unavailable and any(record[name]["status"] == "unavailable" for name in CONTROL_CONDITIONS):
        raise ControlUnavailableError("non-empty formal dev placebo is unavailable")


def _control_record_integrity(record: Mapping[str, Any], context_record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CONTROL_RECORD_SCHEMA:
        raise ControlManifestError("wrong control record schema")
    recorded_hash = record.get("record_sha256")
    if not isinstance(recorded_hash, str) or recorded_hash != canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    ):
        raise ControlManifestError("control record hash mismatch")
    if record.get("context_build_id") != context_record.get("context_build_id"):
        raise ControlManifestError("control record refers to another context build")
    if record.get("context_record_sha256") != context_record.get("record_sha256"):
        raise ControlManifestError("control record refers to another context record")
    if record.get("query_id") != context_record.get("query", {}).get("id"):
        raise ControlManifestError("control/context query IDs disagree")


def _replace_frozen_block(
    messages: Sequence[Mapping[str, str]], *, old_block: str, new_block: str
) -> list[dict[str, str]]:
    rendered = [dict(message) for message in messages]
    if not old_block:
        if new_block:
            raise ControlManifestError("cannot add a non-empty placebo to an empty target")
        return rendered
    occurrences = [
        (index, message["content"].count(old_block))
        for index, message in enumerate(rendered)
        if isinstance(message.get("content"), str) and old_block in message["content"]
    ]
    if sum(count for _, count in occurrences) != 1:
        raise ControlManifestError(
            "frozen target block is not a unique message span; pure replacement is unsafe"
        )
    index = next(index for index, count in occurrences if count)
    rendered[index]["content"] = rendered[index]["content"].replace(old_block, new_block, 1)
    return rendered


def render_control_condition_item(
    context_record: Mapping[str, Any],
    control_record: Mapping[str, Any],
    *,
    lexicon_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    demo_catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    tokenizer: Any,
    condition: str,
    base_condition: str | None = None,
) -> dict[str, Any]:
    """Purely replace one frozen evidence block and preserve everything else.

    Artifact adapters use CL as PL's base and CD as PD's base.  ``base_condition``
    is exposed for diagnostic compositions such as replacing L inside CLD; the
    untouched source IDs/messages remain byte-identical.
    """

    validate_context_record(context_record)
    _control_record_integrity(control_record, context_record)
    if condition not in CONTROL_CONDITIONS:
        raise ControlManifestError(f"unknown placebo condition: {condition}")
    if base_condition is None:
        base_condition = "CL" if condition == "PL" else "CD"
    conditions = context_record.get("conditions", {})
    if base_condition not in conditions:
        raise ControlManifestError(f"unknown frozen base condition: {base_condition}")
    shape = control_record[condition]
    if shape.get("status") == "unavailable":
        raise ControlUnavailableError(f"{condition} is unavailable and cannot be rendered")
    kind = "lexicon" if condition == "PL" else "demo"
    catalog = _catalog_map(lexicon_catalog if kind == "lexicon" else demo_catalog, kind)
    expected_target = _selected_ids(context_record, kind)
    if shape.get("target_ids") != expected_target:
        raise ControlManifestError(f"{condition} target differs from frozen final context")
    base = conditions[base_condition]
    id_field = "lexicon_ids" if kind == "lexicon" else "demo_ids"
    other_id_field = "demo_ids" if kind == "lexicon" else "lexicon_ids"
    if base.get(id_field) != expected_target:
        raise ControlManifestError(
            f"base condition {base_condition} does not contain the frozen {kind} target"
        )
    replacement_ids = shape.get("replacement_order")
    if not isinstance(replacement_ids, list) or any(item_id not in catalog for item_id in replacement_ids):
        raise ControlManifestError(f"{condition} replacement order is invalid")
    old_block = _block_text(expected_target, catalog)
    new_block = _block_text(replacement_ids, catalog)
    if text_sha256(old_block) != shape.get("target_rendered_block_sha256"):
        raise ControlManifestError(f"{condition} target block hash mismatch")
    if text_sha256(new_block) != shape.get("rendered_block_sha256"):
        raise ControlManifestError(f"{condition} replacement block hash mismatch")
    if token_count(old_block, tokenizer) != shape.get("target_block_tokens"):
        raise ControlManifestError(f"{condition} target block token mismatch")
    if token_count(new_block, tokenizer) != shape.get("replacement_block_tokens"):
        raise ControlManifestError(f"{condition} replacement block token mismatch")
    messages = _replace_frozen_block(base.get("messages", []), old_block=old_block, new_block=new_block)
    chat_text = chat_prompt_text(messages, tokenizer)
    lexicon_ids = replacement_ids if condition == "PL" else list(base.get("lexicon_ids", []))
    demo_ids = replacement_ids if condition == "PD" else list(base.get("demo_ids", []))
    if list(base.get(other_id_field, [])) != (demo_ids if kind == "lexicon" else lexicon_ids):
        raise ControlManifestError("pure placebo rendering changed the untouched evidence source")
    query = context_record["query"]
    return {
        "id": str(query["id"]),
        "content": query["content"],
        "gt_quadruples": copy.deepcopy(query["gold"]),
        "messages_list": [messages],
        "context_manifest": {
            "context_build_id": context_record.get("context_build_id"),
            "record_sha256": context_record["record_sha256"],
            "base_condition": base_condition,
            "lexicon_ids": lexicon_ids,
            "demo_ids": demo_ids,
        },
        "control_manifest": {
            "control_build_id": control_record["control_build_id"],
            "record_sha256": control_record["record_sha256"],
            "condition": condition,
            "target_ids": shape["target_ids"],
            "replacement_ids": replacement_ids,
            "status": shape["status"],
            "effective_treatment": shape["effective_treatment"],
            "chat_prompt_tokens": token_count(chat_text, tokenizer),
            "chat_prompt_sha256": text_sha256(chat_text),
        },
    }


render_control_item = render_control_condition_item


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlManifestError(f"cannot read JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ControlManifestError(f"{path}:{line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlManifestError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)


def _payload_manifest(target: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*"), key=lambda value: value.relative_to(target).as_posix()):
        if path.is_symlink():
            raise ControlManifestError("artifact payload cannot contain symlinks")
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(target).as_posix(),
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
            }
        )
    return {"schema_version": PAYLOAD_MANIFEST_SCHEMA, "files": files}


def _verify_payload_manifest(target: Path) -> str:
    path = target / "payload_manifest.json"
    stored = _load_json(path)
    if stored != _payload_manifest(target):
        raise ControlManifestError(f"payload manifest mismatch: {target}")
    return sha256_file(path)


def _resolve_locator(ref_path: str | Path, *, kinds: set[str]) -> tuple[dict[str, Any], Path]:
    locator = _load_json(Path(ref_path))
    if locator.get("schema_version") != LOCATOR_REF_SCHEMA or locator.get("artifact_kind") not in kinds:
        raise ControlManifestError(f"locator is not a Stage 1 {sorted(kinds)} ref")
    artifact_id = locator.get("artifact_id")
    target_raw = locator.get("target_path")
    if not isinstance(artifact_id, str) or not isinstance(target_raw, str):
        raise ControlManifestError("locator lacks artifact ID/target path")
    target = Path(target_raw)
    if not target.is_absolute() or not target.is_dir() or target.name != artifact_id:
        raise ControlManifestError("locator target is not an absolute matching artifact directory")
    payload_hash = _verify_payload_manifest(target)
    if locator.get("payload_manifest_sha256") != payload_hash:
        raise ControlManifestError("locator payload hash mismatch")
    if set(locator) != {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "target_path",
        "payload_manifest_sha256",
    }:
        raise ControlManifestError("locator contains unsupported fields")
    return locator, target


def _portable_dependency(locator: Mapping[str, Any]) -> dict[str, Any]:
    directory = "test_contexts" if locator["artifact_kind"] == "test-context" else "contexts"
    return {
        "schema_version": DEPENDENCY_REF_SCHEMA,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "logical_repo_path": f"{directory}/{locator['artifact_id']}",
    }


def _validate_dependency(value: Mapping[str, Any], *, kinds: set[str]) -> None:
    required = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
        "logical_repo_path",
    }
    if set(value) != required or value.get("schema_version") != DEPENDENCY_REF_SCHEMA:
        raise ControlManifestError("embedded context ref is not a portable dependency")
    if value.get("artifact_kind") not in kinds or not isinstance(value.get("artifact_id"), str):
        raise ControlManifestError("embedded context dependency kind/ID is invalid")
    if not SHA256_RE.fullmatch(str(value.get("payload_manifest_sha256", ""))):
        raise ControlManifestError("embedded context dependency payload hash is invalid")
    logical = value.get("logical_repo_path")
    if not isinstance(logical, str) or not logical or Path(logical).is_absolute() or ".." in Path(logical).parts:
        raise ControlManifestError("embedded dependency path is not portable")


def _dependency_identity(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise ControlManifestError("embedded dependency identity is malformed")
    kind = value.get("artifact_kind")
    artifact_id = value.get("artifact_id")
    payload_hash = value.get("payload_manifest_sha256")
    if (
        not isinstance(kind, str)
        or not kind
        or not isinstance(artifact_id, str)
        or not artifact_id
        or not isinstance(payload_hash, str)
        or not SHA256_RE.fullmatch(payload_hash)
    ):
        raise ControlManifestError("embedded dependency identity is malformed")
    return kind, artifact_id, payload_hash


def _catalog_file(target: Path, kind: str) -> Path:
    candidates = (
        ("catalogs/lexicon_pool.jsonl", "catalogs/lexicon_catalog.jsonl")
        if kind == "lexicon"
        else ("catalogs/demo_pool.train.jsonl", "catalogs/demo_catalog.train.jsonl")
    )
    found = [target / name for name in candidates if (target / name).is_file()]
    if len(found) != 1:
        raise ControlManifestError(
            f"context target must contain exactly one supported {kind} catalog file"
        )
    return found[0]


def _context_snapshot(target: Path, *, split: str, expected_id: str) -> dict[str, Any]:
    for path in target.rglob("*"):
        if path.is_file() and _POST_OUTCOME_KEY_RE.search(path.relative_to(target).as_posix()):
            raise ControlManifestError(
                f"context target is contaminated by a post-outcome file: {path.name}"
            )
    records_path = target / f"context_manifest.{split}.jsonl"
    meta_path = target / f"context_manifest.{split}.meta.json"
    if not records_path.is_file() or not meta_path.is_file():
        raise ControlManifestError(f"context target lacks frozen {split} manifest/meta")
    meta = _load_json(meta_path)
    if not isinstance(meta, Mapping) or meta.get("context_build_id") != expected_id:
        raise ControlManifestError("context meta build ID mismatch")
    sources = meta.get("sources")
    if not isinstance(sources, Mapping):
        raise ControlManifestError("context meta lacks train-pool lineage")
    demo_source = sources.get("demo_pool")
    lexicon_source = sources.get("lexicon_pool")
    if not isinstance(demo_source, Mapping) or demo_source.get("split") != "train":
        raise ControlManifestError("context demo catalog is not proven train-only")
    if not isinstance(lexicon_source, Mapping) or lexicon_source.get("train_only_verified") is not True:
        raise ControlManifestError("context lexicon catalog is not proven train-only")
    scientific_eligible = meta.get("scientific_eligible")
    if scientific_eligible is True:
        if demo_source.get("partition") != "fit":
            raise ControlManifestError(
                "scientific context demo catalog is not proven fit-only"
            )
        if lexicon_source.get("partition") != "fit":
            raise ControlManifestError(
                "scientific context lexicon catalog is not proven fit-only"
            )
        partition_ref_path = target / "train_partition_ref.json"
        if not partition_ref_path.is_file():
            raise ControlManifestError(
                "scientific context lacks its immutable train-partition dependency"
            )
        partition_dependency = _load_json(partition_ref_path)
        if not isinstance(partition_dependency, Mapping):
            raise ControlManifestError(
                "scientific context train-partition dependency is malformed"
            )
        _validate_dependency(partition_dependency, kinds={"train-partition"})
        prepared_path = target / "prepared_bundle.meta.json"
        if not prepared_path.is_file():
            raise ControlManifestError(
                "scientific context lacks frozen retrieval provenance"
            )
        prepared = _load_json(prepared_path)
        retrieval = (
            prepared.get("retrieval_provenance")
            if isinstance(prepared, Mapping)
            else None
        )
        if (
            not isinstance(retrieval, Mapping)
            or retrieval.get("fit_only_demo_pool") is not True
            or retrieval.get("calibration_demo_excluded") is not True
            or _dependency_identity(retrieval.get("train_partition_dependency"))
            != _dependency_identity(partition_dependency)
        ):
            raise ControlManifestError(
                "scientific context retrieval is not bound to its fit-only partition"
            )
    elif scientific_eligible is False or scientific_eligible is None:
        # Pre-P0 engineering artifacts did not freeze a partition label.  They
        # remain useful for deterministic smoke replay, but may never claim
        # scientific eligibility.  New engineering artifacts explicitly use
        # the legacy-full-train label and are checked when it is present.
        for source in (demo_source, lexicon_source):
            label = source.get("partition")
            if label not in {None, "legacy-full-train"}:
                raise ControlManifestError(
                    "engineering context contains an unsupported pool partition"
                )
    else:
        raise ControlManifestError("context scientific eligibility flag is invalid")
    records = _load_jsonl(records_path)
    if not records:
        raise ControlManifestError("context manifest is empty")
    query_ids: list[str] = []
    for record in records:
        validate_context_record(record)
        _reject_post_outcome_fields(record, where="context record")
        if record.get("context_build_id") != expected_id:
            raise ControlManifestError("context record build ID mismatch")
        query_id = record.get("query", {}).get("id")
        if not isinstance(query_id, str) or query_id in query_ids:
            raise ControlManifestError("context query IDs are missing or duplicated")
        query_ids.append(query_id)
    if meta.get("records_sha256") not in {None, sha256_file(records_path)}:
        raise ControlManifestError("context records file hash disagrees with meta")
    if meta.get("record_count") not in {None, len(records)}:
        raise ControlManifestError("context record count disagrees with meta")
    lexicon_path = _catalog_file(target, "lexicon")
    demo_path = _catalog_file(target, "demo")
    lexicons = _catalog_map(_load_jsonl(lexicon_path), "lexicon")
    demos = _catalog_map(_load_jsonl(demo_path), "demo")
    return {
        "meta": dict(meta),
        "records": records,
        "query_ids": query_ids,
        "records_path": records_path,
        "lexicon_catalog": lexicons,
        "demo_catalog": demos,
        "lexicon_catalog_path": lexicon_path,
        "demo_catalog_path": demo_path,
        "train_pool_sha256": canonical_sha256(
            {
                "lexicon_catalog_sha256": canonical_sha256(lexicons),
                "demo_catalog_sha256": canonical_sha256(demos),
            }
        ),
    }


def _tokenizer_revision(
    resolved: dict[str, Any], context_meta: Mapping[str, Any], explicit: str | None
) -> str:
    configured = None
    if isinstance(resolved.get("tokenizer"), Mapping):
        configured = resolved["tokenizer"].get("revision")
    context_revision = context_meta.get("budget", {}).get("tokenizer_revision")
    values = [value for value in (configured, context_revision, explicit) if value is not None]
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ControlManifestError("a frozen tokenizer revision is required")
    if len(set(values)) != 1:
        raise ControlManifestError("control/context/CLI tokenizer revisions disagree")
    revision = values[0]
    tokenizer_config = dict(resolved.get("tokenizer", {}))
    tokenizer_config["revision"] = revision
    resolved["tokenizer"] = tokenizer_config
    return revision


def _logical_tokenizer_path(
    resolved: Mapping[str, Any], workspace_root: str | Path
) -> Path:
    tokenizer = resolved.get("tokenizer")
    logical_raw = tokenizer.get("logical_path") if isinstance(tokenizer, Mapping) else None
    logical = Path(logical_raw) if isinstance(logical_raw, str) else None
    if (
        not isinstance(logical_raw, str)
        or not logical_raw
        or logical is None
        or logical.is_absolute()
        or logical_raw != logical.as_posix()
        or logical in {Path("."), Path("")}
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise ControlManifestError(
            "scientific control requires a canonical frozen tokenizer.logical_path"
        )
    root = Path(workspace_root).resolve()
    target = (root / logical).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ControlManifestError("frozen tokenizer path escapes workspace_root") from exc
    return target


def _construct_control_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ControlManifestError(
            "transformers is required to load the frozen control tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ControlManifestError(f"frozen control tokenizer construction failed: {exc}") from exc


def _frozen_tokenizer_source_identity(
    resolved: Mapping[str, Any],
    context_meta: Mapping[str, Any],
    *,
    revision: str,
    workspace_root: str | Path,
    scientific: bool,
) -> tuple[dict[str, Any], ResolvedModelSourceContract]:
    """Resolve the exact tokenizer tree used for deterministic control replay."""

    target = _logical_tokenizer_path(resolved, workspace_root)
    try:
        inventory = inventory_regular_file_tree(
            target,
            workspace_root=workspace_root,
            label="control tokenizer",
            inventory_policy="all-regular-files/v1",
        )
    except ModelRegistryError as exc:
        raise ControlManifestError(f"control tokenizer inventory failed: {exc}") from exc
    identity = {
        "schema_version": TOKENIZER_SOURCE_IDENTITY_SCHEMA,
        "scientific_eligible": scientific,
        "declared_revision": revision,
        "inventory": inventory,
        "constructor_policy": dict(TOKENIZER_CONSTRUCTOR_POLICY),
    }
    if scientific:
        context_runtime = context_meta.get("runtime_source_identity")
        context_tokenizer = (
            context_runtime.get("tokenizer")
            if isinstance(context_runtime, Mapping)
            else None
        )
        expected_context_identity = {
            "declared_revision": revision,
            "inventory": inventory,
            "constructor_policy": dict(TOKENIZER_CONSTRUCTOR_POLICY),
        }
        if not isinstance(context_tokenizer, Mapping) or any(
            context_tokenizer.get(key) != value
            for key, value in expected_context_identity.items()
        ):
            raise ControlManifestError(
                "scientific control tokenizer identity differs from its frozen context"
            )
    root = Path(workspace_root).resolve()
    contract = ResolvedModelSourceContract(
        workspace_root=root,
        checkpoint_inventory=inventory,
        tokenizer_inventory=inventory,
        base_inventory=inventory,
    )
    return identity, contract


def _injected_engineering_tokenizer_identity(revision: str) -> dict[str, Any]:
    """Explicitly non-scientific identity retained for unit/smoke compatibility."""

    return {
        "schema_version": TOKENIZER_SOURCE_IDENTITY_SCHEMA,
        "scientific_eligible": False,
        "declared_revision": revision,
        "inventory": None,
        "constructor_policy": {
            "backend": "caller-injected-engineering-test-only/v1",
            "local_files_only": None,
            "trust_remote_code": None,
        },
    }


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ControlManifestError("control config root must be an object")
    return value


def _target_parent(
    raw_config: Mapping[str, Any],
    target_root: str | Path | None,
    *,
    sealed: bool = False,
) -> Path:
    if target_root is not None:
        return Path(target_root).resolve()
    artifact_root = raw_config.get("artifact_root")
    if not isinstance(artifact_root, str) or not artifact_root:
        raise ControlManifestError("control build requires target_root or config artifact_root")
    root = Path(artifact_root)
    if not root.is_absolute():
        repository_root = Path(__file__).resolve().parents[2]
        root = repository_root / root
    return (root / ("test_controls" if sealed else "controls")).resolve()


def _workspace_dependency(
    locator: Mapping[str, Any], target: Path, workspace_root: str | Path
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    try:
        logical = target.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ControlManifestError("sealed dependency must live below workspace_root") from exc
    return {
        "schema_version": DEPENDENCY_REF_SCHEMA,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "logical_repo_path": logical,
    }


def _resolve_workspace_dependency(
    dependency: Mapping[str, Any],
    workspace_root: str | Path,
    *,
    kinds: set[str],
) -> Path:
    _validate_dependency(dependency, kinds=kinds)
    root = Path(workspace_root).resolve()
    target = (root / str(dependency["logical_repo_path"])).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ControlManifestError("sealed dependency escapes workspace_root") from exc
    if not target.is_dir() or target.name != dependency["artifact_id"]:
        raise ControlManifestError("sealed dependency cannot be resolved")
    if _verify_payload_manifest(target) != dependency["payload_manifest_sha256"]:
        raise ControlManifestError("sealed dependency payload hash mismatch")
    return target


def _atomic_write_locator(path: Path, locator: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(locator) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _records_file_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    ).hexdigest()


def _runner_items(
    records: Sequence[Mapping[str, Any]],
    context_records: Sequence[Mapping[str, Any]],
    *,
    lexicon_catalog: Mapping[str, Any],
    demo_catalog: Mapping[str, Any],
    tokenizer: Any,
    condition: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for context_record, control_record in zip(context_records, records, strict=True):
        if control_record[condition]["status"] == "unavailable":
            continue
        result.append(
            render_control_condition_item(
                context_record,
                control_record,
                lexicon_catalog=lexicon_catalog,
                demo_catalog=demo_catalog,
                tokenizer=tokenizer,
                condition=condition,
            )
        )
    return result


def build_control_artifact(
    *,
    config: str | Path | Mapping[str, Any],
    context_ref: str | Path,
    write_ref: str | Path | None,
    split: str,
    tokenizer: Any | None = None,
    tokenizer_revision: str | None = None,
    target_root: str | Path | None = None,
    allow_unavailable: bool | None = None,
    sealed_lineage: Mapping[str, Any] | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build with a frozen tokenizer source; injection is engineering-only."""

    if split not in {"train", "dev", "test"}:
        raise ControlManifestError("control split must be train/dev/test")
    sealed = sealed_lineage is not None
    raw_config = _load_config(config)
    resolved = resolve_control_config(raw_config)
    context_locator, context_target = _resolve_locator(
        context_ref, kinds={"context", "test-context"}
    )
    snapshot = _context_snapshot(
        context_target, split=split, expected_id=context_locator["artifact_id"]
    )
    scientific = snapshot["meta"].get("scientific_eligible") is True or sealed
    if sealed and snapshot["meta"].get("scientific_eligible") is not True:
        raise ControlManifestError("sealed control requires a scientific test context")
    if scientific and (tokenizer is not None or tokenizer_revision is not None):
        raise ControlManifestError(
            "scientific control forbids caller-injected tokenizer/revision"
        )
    revision = _tokenizer_revision(
        resolved,
        snapshot["meta"],
        tokenizer_revision if not scientific else None,
    )
    if tokenizer is not None:
        source_identity = _injected_engineering_tokenizer_identity(revision)
        return _build_control_artifact_with_tokenizer(
            config=raw_config,
            context_ref=context_ref,
            write_ref=write_ref,
            split=split,
            tokenizer=tokenizer,
            tokenizer_revision=revision,
            tokenizer_source_identity=source_identity,
            target_root=target_root,
            allow_unavailable=allow_unavailable,
            sealed_lineage=sealed_lineage,
            workspace_root=workspace_root,
        )

    source_identity, contract = _frozen_tokenizer_source_identity(
        resolved,
        snapshot["meta"],
        revision=revision,
        workspace_root=workspace_root,
        scientific=scientific,
    )
    # Source leases monitor workspace ancestors.  Create unrelated output
    # directories before entering the lease, then keep construction, all token
    # counting/replay, validation, and publication inside the verified window.
    _target_parent(raw_config, target_root, sealed=sealed).mkdir(
        parents=True, exist_ok=True
    )
    if write_ref is not None:
        Path(write_ref).parent.mkdir(parents=True, exist_ok=True)
    try:
        with verified_model_source_lease(
            contract, source_names=("tokenizer",)
        ) as paths:
            frozen_tokenizer = _construct_control_tokenizer(paths.tokenizer_path)
            return _build_control_artifact_with_tokenizer(
                config=raw_config,
                context_ref=context_ref,
                write_ref=write_ref,
                split=split,
                tokenizer=frozen_tokenizer,
                tokenizer_revision=revision,
                tokenizer_source_identity=source_identity,
                target_root=target_root,
                allow_unavailable=allow_unavailable,
                sealed_lineage=sealed_lineage,
                workspace_root=workspace_root,
            )
    except ModelRegistryError as exc:
        raise ControlManifestError(f"control tokenizer source lease failed: {exc}") from exc


def _build_control_artifact_with_tokenizer(
    *,
    config: str | Path | Mapping[str, Any],
    context_ref: str | Path,
    write_ref: str | Path | None,
    split: str,
    tokenizer: Any,
    tokenizer_source_identity: Mapping[str, Any],
    tokenizer_revision: str | None = None,
    target_root: str | Path | None = None,
    allow_unavailable: bool | None = None,
    sealed_lineage: Mapping[str, Any] | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Private replay implementation; caller owns the complete source lease."""

    if split not in {"train", "dev", "test"}:
        raise ControlManifestError("control split must be train/dev/test")
    sealed = sealed_lineage is not None
    if split == "test" and not sealed:
        raise ControlManifestError("test controls can only be created through seal-test")
    if sealed and split != "test":
        raise ControlManifestError("sealed control build requires split=test")
    raw_config = _load_config(config)
    resolved = resolve_control_config(raw_config)
    context_locator, context_target = _resolve_locator(
        context_ref, kinds={"context", "test-context"}
    )
    if sealed and context_locator["artifact_kind"] != "test-context":
        raise ControlManifestError("sealed control requires a test-context dependency")
    if not sealed and context_locator["artifact_kind"] == "test-context":
        raise ControlManifestError("test-context controls require seal-test")
    snapshot = _context_snapshot(
        context_target, split=split, expected_id=context_locator["artifact_id"]
    )
    revision = _tokenizer_revision(resolved, snapshot["meta"], tokenizer_revision)
    scientific = snapshot["meta"].get("scientific_eligible") is True or sealed
    if tokenizer_source_identity.get("scientific_eligible") is not scientific:
        raise ControlManifestError("control tokenizer source eligibility mismatch")
    if scientific and not isinstance(tokenizer_source_identity.get("inventory"), Mapping):
        raise ControlManifestError("scientific control lacks a full tokenizer inventory")
    context_dependency = _portable_dependency(context_locator)
    builder_code_sha256 = sha256_file(__file__)
    id_inputs = {
        "schema_version": CONTROL_ARTIFACT_SCHEMA,
        "context_dependency": context_dependency,
        "split": split,
        "resolved_control_config": resolved,
        "train_pool_sha256": snapshot["train_pool_sha256"],
        "tokenizer_revision": revision,
        "tokenizer_source_identity": dict(tokenizer_source_identity),
        "control_builder_code_sha256": builder_code_sha256,
    }
    if sealed:
        frozen = dict(sealed_lineage or {})
        if frozen.get("schema_version") != FROZEN_CONTROL_POLICY_SCHEMA:
            raise ControlManifestError("invalid frozen control policy lineage")
        id_inputs["artifact_kind"] = TEST_CONTROL_ARTIFACT_KIND
        id_inputs["frozen_policy_ref"] = frozen
    control_build_id = "ctl-" + canonical_sha256(id_inputs)
    shared_token_costs = {
        "lexicon": _prepare_token_costs(
            _catalog_map(snapshot["lexicon_catalog"], "lexicon"), tokenizer
        ),
        "demo": _prepare_token_costs(
            _catalog_map(snapshot["demo_catalog"], "demo"), tokenizer
        ),
    }
    records = [
        _build_control_record_unchecked(
            record,
            lexicon_catalog=snapshot["lexicon_catalog"],
            demo_catalog=snapshot["demo_catalog"],
            tokenizer=tokenizer,
            resolved_config=resolved,
            control_build_id=control_build_id,
            token_costs=shared_token_costs,
        )
        for record in snapshot["records"]
    ]
    unavailable = {
        condition: [
            record["query_id"]
            for record in records
            if record[condition]["status"] == "unavailable"
        ]
        for condition in CONTROL_CONDITIONS
    }
    if allow_unavailable is None:
        allow_unavailable = not (split == "dev" and resolved["dev_nonempty_required"] is True)
    if split == "dev" and resolved["dev_nonempty_required"] is True and allow_unavailable:
        raise ControlManifestError(
            "formal dev config forbids --allow-unavailable; use an explicit engineering config"
        )
    if not allow_unavailable and any(unavailable.values()):
        raise ControlUnavailableError(
            "formal dev controls have unavailable non-empty queries: "
            + ", ".join(f"{key}={value[:5]}" for key, value in unavailable.items() if value)
        )

    runners = {
        condition: _runner_items(
            records,
            snapshot["records"],
            lexicon_catalog=snapshot["lexicon_catalog"],
            demo_catalog=snapshot["demo_catalog"],
            tokenizer=tokenizer,
            condition=condition,
        )
        for condition in CONTROL_CONDITIONS
    }
    status_counts = {
        condition: dict(sorted(Counter(record[condition]["status"] for record in records).items()))
        for condition in CONTROL_CONDITIONS
    }
    meta = {
        "schema_version": CONTROL_META_SCHEMA,
        "control_build_id": control_build_id,
        "context_build_id": context_locator["artifact_id"],
        "split": split,
        "record_count": len(records),
        "ordered_query_ids_sha256": canonical_sha256(snapshot["query_ids"]),
        "records_sha256": _records_file_hash(records),
        "status_counts": status_counts,
        "availability_mask": {
            condition: [
                record["query_id"]
                for record in records
                if record[condition]["status"] != "unavailable"
            ]
            for condition in CONTROL_CONDITIONS
        },
        "nondegenerate_masks": {
            "L_nondegenerate": [record["query_id"] for record in records if record["PL"]["status"] == "ok"],
            "D_nondegenerate": [record["query_id"] for record in records if record["PD"]["status"] == "ok"],
            "LD_nondegenerate": [
                record["query_id"]
                for record in records
                if record["PL"]["status"] == "ok" and record["PD"]["status"] == "ok"
            ],
        },
        "unavailable_query_ids": unavailable,
        "tokenizer_source_identity": dict(tokenizer_source_identity),
        "id_inputs": id_inputs,
    }
    if sealed:
        meta["artifact_kind"] = TEST_CONTROL_ARTIFACT_KIND
    provenance = {
        "schema_version": CONTROL_PROVENANCE_SCHEMA,
        "control_build_id": control_build_id,
        "context_dependency": context_dependency,
        "train_pool_sha256": snapshot["train_pool_sha256"],
        "context_records_sha256": sha256_file(snapshot["records_path"]),
        "lexicon_catalog_sha256": canonical_sha256(snapshot["lexicon_catalog"]),
        "demo_catalog_sha256": canonical_sha256(snapshot["demo_catalog"]),
        "tokenizer_revision": revision,
        "tokenizer_source_identity": dict(tokenizer_source_identity),
        "builder_code_sha256": builder_code_sha256,
        "saw_condition_predictions": False,
        "saw_post_outcome_model_scores": False,
    }
    if sealed:
        provenance["frozen_policy_ref"] = dict(sealed_lineage or {})

    parent = _target_parent(raw_config, target_root, sealed=sealed)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / control_build_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{control_build_id}.", dir=parent))
    try:
        _write_json(temporary / "config.resolved.json", resolved)
        _write_json(temporary / "context_ref.json", context_dependency)
        if sealed:
            _write_json(
                temporary / "frozen_policy_ref.json", dict(sealed_lineage or {})
            )
        _write_json(temporary / "provenance.json", provenance)
        _write_json(temporary / "control_manifest.meta.json", meta)
        _write_jsonl(temporary / f"control_manifest.{split}.jsonl", records)
        for condition, items in runners.items():
            _write_json(temporary / "conditions" / "runner" / condition / f"{split}.json", items)
        _write_json(temporary / "payload_manifest.json", _payload_manifest(temporary))
        _validate_control_target_with_tokenizer(
            temporary,
            tokenizer=tokenizer,
            tokenizer_source_identity=tokenizer_source_identity,
            context_target=context_target,
            require_directory_name=False,
            workspace_root=workspace_root,
        )
        if target.exists():
            _validate_control_target_with_tokenizer(
                target,
                tokenizer=tokenizer,
                tokenizer_source_identity=tokenizer_source_identity,
                context_target=context_target,
                workspace_root=workspace_root,
            )
            if _load_json(target / "payload_manifest.json") != _load_json(temporary / "payload_manifest.json"):
                raise ControlManifestError("same control build ID produced a different payload")
        else:
            os.replace(temporary, target)
        payload_hash = sha256_file(target / "payload_manifest.json")
        locator = {
            "schema_version": LOCATOR_REF_SCHEMA,
            "artifact_kind": (
                TEST_CONTROL_ARTIFACT_KIND if sealed else CONTROL_ARTIFACT_KIND
            ),
            "artifact_id": control_build_id,
            "target_path": str(target.resolve()),
            "payload_manifest_sha256": payload_hash,
        }
        if write_ref is not None:
            _atomic_write_locator(Path(write_ref), locator)
        return locator
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _resolve_embedded_context_target(control_target: Path, dependency: Mapping[str, Any]) -> Path:
    logical = Path(str(dependency["logical_repo_path"]))
    artifact_root = control_target.parent.parent
    candidate = (artifact_root / logical).resolve()
    if not candidate.is_dir() or candidate.name != dependency["artifact_id"]:
        raise ControlManifestError(
            "portable context dependency cannot be resolved beside the control artifact"
        )
    return candidate


def _verify_context_dependency(
    context_target: Path, dependency: Mapping[str, Any]
) -> None:
    if context_target.name != dependency["artifact_id"]:
        raise ControlManifestError("resolved context target ID mismatch")
    payload_hash = _verify_payload_manifest(context_target)
    if payload_hash != dependency["payload_manifest_sha256"]:
        raise ControlManifestError("resolved context payload hash mismatch")


def seal_test_control_artifact(
    *,
    test_context_ref: str | Path,
    frozen_control_ref: str | Path,
    write_ref: str | Path | None,
    tokenizer: Any | None = None,
    target_root: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Build sealed PL/PD using only the exact frozen dev config/code/tier policy."""

    if tokenizer is not None:
        raise ControlManifestError("scientific seal-test forbids caller-injected tokenizer")
    root = Path(workspace_root).resolve()
    frozen_locator, frozen_target = _resolve_locator(
        frozen_control_ref, kinds={CONTROL_ARTIFACT_KIND}
    )
    frozen_report = validate_control_target(frozen_target, workspace_root=root)
    if frozen_report.get("split") != "dev":
        raise ControlManifestError("seal-test requires a frozen dev control")
    frozen_meta = _load_json(frozen_target / "control_manifest.meta.json")
    frozen_inputs = frozen_meta.get("id_inputs")
    if not isinstance(frozen_inputs, Mapping) or frozen_inputs.get(
        "control_builder_code_sha256"
    ) != sha256_file(__file__):
        raise ControlManifestError("frozen dev control builder code differs from seal-test")
    frozen_config = _load_json(frozen_target / "config.resolved.json")
    frozen_context_dependency = _load_json(frozen_target / "context_ref.json")
    frozen_context_target = _resolve_embedded_context_target(
        frozen_target, frozen_context_dependency
    )
    test_context_locator, test_context_target = _resolve_locator(
        test_context_ref, kinds={"test-context"}
    )
    from data.build_context_manifest import validate_context_ref, validate_context_target

    frozen_context_report = validate_context_target(
        frozen_context_target,
        workspace_root=root,
    )
    if (
        frozen_context_report.get("split") != "dev"
        or frozen_context_report.get("scientific_eligible") is not True
    ):
        raise ControlManifestError(
            "seal-test requires a dev control over a scientific dev context"
        )

    test_context_report = validate_context_ref(
        test_context_ref, workspace_root=root
    )
    if test_context_report.get("split") != "test" or test_context_report.get(
        "scientific_eligible"
    ) is not True:
        raise ControlManifestError("sealed controls require a scientific test context")
    context_policy = _load_json(test_context_target / "frozen_policy_ref.json")
    sealed_dev_context = context_policy.get("frozen_dev_context_dependency")
    # Context sealing stores paths workspace-relative, whereas controls store
    # context paths artifact-root-relative.  The immutable identity fields must
    # match exactly; each validator independently resolves its own portable path.
    if not isinstance(sealed_dev_context, Mapping) or any(
        sealed_dev_context.get(key) != frozen_context_dependency.get(key)
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
        )
    ):
        raise ControlManifestError(
            "test context frozen-dev lineage differs from dev control context"
        )
    test_context_dependency = _portable_dependency(test_context_locator)
    lineage = {
        "schema_version": FROZEN_CONTROL_POLICY_SCHEMA,
        "frozen_dev_control_dependency": _workspace_dependency(
            frozen_locator, frozen_target, root
        ),
        "frozen_dev_context_dependency": dict(frozen_context_dependency),
        "test_context_dependency": test_context_dependency,
        "resolved_control_config_sha256": canonical_sha256(frozen_config),
        "control_builder_code_sha256": frozen_inputs[
            "control_builder_code_sha256"
        ],
        "candidate_tier_policy_sha256": canonical_sha256(
            {
                "quantile_policy": frozen_config["quantile_policy"],
                "candidate_expansion_tiers_percent": frozen_config[
                    "candidate_expansion_tiers_percent"
                ],
                "max_block_token_delta_fraction": frozen_config[
                    "max_block_token_delta_fraction"
                ],
                "lexical_overlap_policy": frozen_config[
                    "lexical_overlap_policy"
                ],
            }
        ),
    }
    return build_control_artifact(
        config=frozen_config,
        context_ref=test_context_ref,
        write_ref=write_ref,
        split="test",
        tokenizer_revision=None,
        target_root=target_root,
        allow_unavailable=True,
        sealed_lineage=lineage,
        workspace_root=root,
    )


def validate_control_target(
    target_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    context_target: str | Path | None = None,
    require_directory_name: bool = True,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Replay controls under their frozen tokenizer source identity."""

    target = Path(target_dir)
    required = {
        "config.resolved.json",
        "context_ref.json",
        "control_manifest.meta.json",
        "payload_manifest.json",
    }
    if not target.is_dir() or not required.issubset(
        {path.name for path in target.iterdir() if path.is_file()}
    ):
        raise ControlManifestError("control target lacks required payload files")
    _verify_payload_manifest(target)
    config = _load_json(target / "config.resolved.json")
    resolved = resolve_control_config(config)
    dependency = _load_json(target / "context_ref.json")
    if not isinstance(dependency, Mapping):
        raise ControlManifestError("control target context ref must be an object")
    _validate_dependency(dependency, kinds={"context", "test-context"})
    upstream = (
        _resolve_embedded_context_target(target, dependency)
        if context_target is None
        else Path(context_target)
    )
    _verify_context_dependency(upstream, dependency)
    meta = _load_json(target / "control_manifest.meta.json")
    split = meta.get("split") if isinstance(meta, Mapping) else None
    if split not in {"train", "dev", "test"}:
        raise ControlManifestError("invalid control split")
    snapshot = _context_snapshot(
        upstream, split=split, expected_id=dependency["artifact_id"]
    )
    scientific = snapshot["meta"].get("scientific_eligible") is True or split == "test"
    revision = _tokenizer_revision(resolved, snapshot["meta"], None)
    if scientific and tokenizer is not None:
        raise ControlManifestError(
            "scientific control validation forbids caller-injected tokenizer"
        )
    stored_source_identity = (
        meta.get("tokenizer_source_identity")
        if isinstance(meta, Mapping)
        else None
    )
    stored_constructor_policy = (
        stored_source_identity.get("constructor_policy")
        if isinstance(stored_source_identity, Mapping)
        else None
    )
    stored_as_engineering_injection = (
        isinstance(stored_source_identity, Mapping)
        and stored_source_identity.get("inventory") is None
        and isinstance(stored_constructor_policy, Mapping)
        and stored_constructor_policy.get("backend")
        == "caller-injected-engineering-test-only/v1"
    )
    # A caller-provided engineering tokenizer may only replay an artifact that
    # was itself built under the explicit test-only injection identity.  When
    # the artifact freezes a real tokenizer tree, ignore the compatibility
    # argument and reconstruct under that frozen full-tree lease.  Otherwise a
    # downstream registry tokenizer would spuriously change the replay
    # identity (or, worse, influence validation of a content-addressed target).
    if tokenizer is not None and stored_as_engineering_injection:
        return _validate_control_target_with_tokenizer(
            target,
            tokenizer=tokenizer,
            tokenizer_source_identity=_injected_engineering_tokenizer_identity(
                revision
            ),
            context_target=upstream,
            require_directory_name=require_directory_name,
            workspace_root=workspace_root,
        )
    source_identity, contract = _frozen_tokenizer_source_identity(
        resolved,
        snapshot["meta"],
        revision=revision,
        workspace_root=workspace_root,
        scientific=scientific,
    )
    try:
        with verified_model_source_lease(
            contract, source_names=("tokenizer",)
        ) as paths:
            frozen_tokenizer = _construct_control_tokenizer(paths.tokenizer_path)
            return _validate_control_target_with_tokenizer(
                target,
                tokenizer=frozen_tokenizer,
                tokenizer_source_identity=source_identity,
                context_target=upstream,
                require_directory_name=require_directory_name,
                workspace_root=workspace_root,
            )
    except ModelRegistryError as exc:
        raise ControlManifestError(f"control tokenizer source lease failed: {exc}") from exc


def _validate_control_target_with_tokenizer(
    target_dir: str | Path,
    *,
    tokenizer: Any,
    tokenizer_source_identity: Mapping[str, Any],
    context_target: str | Path | None = None,
    require_directory_name: bool = True,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Private exact replay; caller owns the tokenizer source lease."""

    target = Path(target_dir)
    required_top = {
        "config.resolved.json",
        "context_ref.json",
        "provenance.json",
        "control_manifest.meta.json",
        "payload_manifest.json",
    }
    if not target.is_dir() or not required_top.issubset(
        {path.name for path in target.iterdir() if path.is_file()}
    ):
        raise ControlManifestError("control target lacks required payload files")
    _verify_payload_manifest(target)
    config = _load_json(target / "config.resolved.json")
    resolved = resolve_control_config(config)
    dependency = _load_json(target / "context_ref.json")
    if not isinstance(dependency, Mapping):
        raise ControlManifestError("control target context ref must be an object")
    _validate_dependency(dependency, kinds={"context", "test-context"})
    if context_target is None:
        upstream = _resolve_embedded_context_target(target, dependency)
    else:
        upstream = Path(context_target)
    _verify_context_dependency(upstream, dependency)
    meta = _load_json(target / "control_manifest.meta.json")
    provenance = _load_json(target / "provenance.json")
    if not isinstance(meta, Mapping) or meta.get("schema_version") != CONTROL_META_SCHEMA:
        raise ControlManifestError("wrong control meta schema")
    if not isinstance(provenance, Mapping) or provenance.get("schema_version") != CONTROL_PROVENANCE_SCHEMA:
        raise ControlManifestError("wrong control provenance schema")
    build_id = meta.get("control_build_id")
    if not isinstance(build_id, str) or not re.fullmatch(r"ctl-[0-9a-f]{64}", build_id):
        raise ControlManifestError("invalid control build ID")
    if require_directory_name and target.name != build_id:
        raise ControlManifestError("control target directory name differs from build ID")
    if provenance.get("control_build_id") != build_id:
        raise ControlManifestError("control provenance build ID mismatch")
    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping) or "ctl-" + canonical_sha256(id_inputs) != build_id:
        raise ControlManifestError("control build ID cannot be recomputed")
    if id_inputs.get("resolved_control_config") != resolved:
        raise ControlManifestError("resolved control config disagrees with ID inputs")
    if id_inputs.get("context_dependency") != dependency:
        raise ControlManifestError("context dependency disagrees with ID inputs")
    split = meta.get("split")
    if split not in {"train", "dev", "test"} or id_inputs.get("split") != split:
        raise ControlManifestError("invalid control split")
    records_path = target / f"control_manifest.{split}.jsonl"
    if not records_path.is_file():
        raise ControlManifestError("control target lacks its split record file")
    snapshot = _context_snapshot(upstream, split=split, expected_id=dependency["artifact_id"])
    revision = _tokenizer_revision(resolved, snapshot["meta"], None)
    if id_inputs.get("tokenizer_revision") != revision:
        raise ControlManifestError("tokenizer revision disagrees with ID inputs")
    scientific = snapshot["meta"].get("scientific_eligible") is True or split == "test"
    expected_source_identity = dict(tokenizer_source_identity)
    if (
        expected_source_identity.get("schema_version")
        != TOKENIZER_SOURCE_IDENTITY_SCHEMA
        or expected_source_identity.get("scientific_eligible") is not scientific
        or expected_source_identity.get("declared_revision") != revision
        or id_inputs.get("tokenizer_source_identity") != expected_source_identity
        or meta.get("tokenizer_source_identity") != expected_source_identity
        or provenance.get("tokenizer_source_identity") != expected_source_identity
    ):
        raise ControlManifestError("control tokenizer source identity mismatch")
    if scientific and (
        not isinstance(expected_source_identity.get("inventory"), Mapping)
        or expected_source_identity.get("inventory", {}).get("inventory_policy")
        != "all-regular-files/v1"
        or expected_source_identity.get("constructor_policy")
        != TOKENIZER_CONSTRUCTOR_POLICY
    ):
        raise ControlManifestError("scientific control tokenizer source is not fully frozen")
    if id_inputs.get("train_pool_sha256") != snapshot["train_pool_sha256"]:
        raise ControlManifestError("train pool hash disagrees with ID inputs")
    if provenance.get("train_pool_sha256") != snapshot["train_pool_sha256"]:
        raise ControlManifestError("train pool hash disagrees with provenance")
    if provenance.get("context_dependency") != dependency:
        raise ControlManifestError("provenance context dependency mismatch")
    if provenance.get("context_records_sha256") != sha256_file(snapshot["records_path"]):
        raise ControlManifestError("provenance context records hash mismatch")
    if provenance.get("lexicon_catalog_sha256") != canonical_sha256(snapshot["lexicon_catalog"]):
        raise ControlManifestError("provenance lexicon catalog hash mismatch")
    if provenance.get("demo_catalog_sha256") != canonical_sha256(snapshot["demo_catalog"]):
        raise ControlManifestError("provenance demo catalog hash mismatch")
    if provenance.get("tokenizer_revision") != revision:
        raise ControlManifestError("provenance tokenizer revision mismatch")
    if provenance.get("builder_code_sha256") != id_inputs.get("control_builder_code_sha256"):
        raise ControlManifestError("provenance builder code hash mismatch")
    if provenance.get("saw_condition_predictions") is not False:
        raise ControlManifestError("control provenance does not attest prediction blindness")
    if provenance.get("saw_post_outcome_model_scores") is not False:
        raise ControlManifestError("control provenance does not attest score blindness")
    records = _load_jsonl(records_path)
    if len(records) != len(snapshot["records"]) or meta.get("record_count") != len(records):
        raise ControlManifestError("control/context record counts disagree")
    if meta.get("ordered_query_ids_sha256") != canonical_sha256(snapshot["query_ids"]):
        raise ControlManifestError("control ordered query frame mismatch")
    if meta.get("records_sha256") != sha256_file(records_path):
        raise ControlManifestError("control records file hash mismatch")
    shared_token_costs = {
        "lexicon": _prepare_token_costs(
            _catalog_map(snapshot["lexicon_catalog"], "lexicon"), tokenizer
        ),
        "demo": _prepare_token_costs(
            _catalog_map(snapshot["demo_catalog"], "demo"), tokenizer
        ),
    }
    for record, context_record in zip(records, snapshot["records"], strict=True):
        validate_control_record(
            record,
            context_record,
            lexicon_catalog=snapshot["lexicon_catalog"],
            demo_catalog=snapshot["demo_catalog"],
            tokenizer=tokenizer,
            config=resolved,
            allow_unavailable=not (split == "dev" and resolved["dev_nonempty_required"] is True),
            token_costs=shared_token_costs,
        )
    expected_status_counts = {
        condition: dict(sorted(Counter(record[condition]["status"] for record in records).items()))
        for condition in CONTROL_CONDITIONS
    }
    if meta.get("status_counts") != expected_status_counts:
        raise ControlManifestError("control status summary mismatch")
    expected_availability = {
        condition: [
            record["query_id"]
            for record in records
            if record[condition]["status"] != "unavailable"
        ]
        for condition in CONTROL_CONDITIONS
    }
    expected_unavailable = {
        condition: [
            record["query_id"]
            for record in records
            if record[condition]["status"] == "unavailable"
        ]
        for condition in CONTROL_CONDITIONS
    }
    expected_nondegenerate = {
        "L_nondegenerate": [record["query_id"] for record in records if record["PL"]["status"] == "ok"],
        "D_nondegenerate": [record["query_id"] for record in records if record["PD"]["status"] == "ok"],
        "LD_nondegenerate": [
            record["query_id"]
            for record in records
            if record["PL"]["status"] == "ok" and record["PD"]["status"] == "ok"
        ],
    }
    if meta.get("availability_mask") != expected_availability:
        raise ControlManifestError("control availability mask mismatch")
    if meta.get("unavailable_query_ids") != expected_unavailable:
        raise ControlManifestError("control unavailable-query mask mismatch")
    if meta.get("nondegenerate_masks") != expected_nondegenerate:
        raise ControlManifestError("control nondegenerate masks mismatch")
    expected_runners = {
        condition: _runner_items(
            records,
            snapshot["records"],
            lexicon_catalog=snapshot["lexicon_catalog"],
            demo_catalog=snapshot["demo_catalog"],
            tokenizer=tokenizer,
            condition=condition,
        )
        for condition in CONTROL_CONDITIONS
    }
    for condition, expected in expected_runners.items():
        path = target / "conditions" / "runner" / condition / f"{split}.json"
        if not path.is_file() or _load_json(path) != expected:
            raise ControlManifestError(f"{condition} runner adapter cannot be reproduced")
    expected_files = {
        "config.resolved.json",
        "context_ref.json",
        "provenance.json",
        "control_manifest.meta.json",
        f"control_manifest.{split}.jsonl",
        f"conditions/runner/PL/{split}.json",
        f"conditions/runner/PD/{split}.json",
        "payload_manifest.json",
    }
    if split == "test":
        expected_files.add("frozen_policy_ref.json")
    actual_files = {
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ControlManifestError(
            f"control target file set mismatch: {sorted(actual_files ^ expected_files)}"
        )
    if split == "test":
        if meta.get("artifact_kind") != TEST_CONTROL_ARTIFACT_KIND:
            raise ControlManifestError("test control lacks test-control artifact kind")
        frozen_policy = _load_json(target / "frozen_policy_ref.json")
        if (
            not isinstance(frozen_policy, Mapping)
            or frozen_policy.get("schema_version") != FROZEN_CONTROL_POLICY_SCHEMA
            or id_inputs.get("frozen_policy_ref") != frozen_policy
            or provenance.get("frozen_policy_ref") != frozen_policy
            or frozen_policy.get("test_context_dependency") != dependency
        ):
            raise ControlManifestError("sealed control frozen-policy chain is invalid")
        frozen_target = _resolve_workspace_dependency(
            frozen_policy["frozen_dev_control_dependency"],
            workspace_root,
            kinds={CONTROL_ARTIFACT_KIND},
        )
        frozen_report = validate_control_target(
            frozen_target, workspace_root=workspace_root
        )
        if frozen_report.get("split") != "dev":
            raise ControlManifestError("sealed control does not bind a dev control")
        frozen_config = _load_json(frozen_target / "config.resolved.json")
        frozen_meta = _load_json(frozen_target / "control_manifest.meta.json")
        frozen_context = _load_json(frozen_target / "context_ref.json")
        frozen_context_target = _resolve_embedded_context_target(
            frozen_target, frozen_context
        )
        from data.build_context_manifest import validate_context_target

        frozen_context_report = validate_context_target(
            frozen_context_target,
            workspace_root=workspace_root,
        )
        if (
            frozen_context_report.get("split") != "dev"
            or frozen_context_report.get("scientific_eligible") is not True
        ):
            raise ControlManifestError(
                "sealed control does not bind a scientific dev context"
            )
        test_context_report = validate_context_target(
            upstream,
            workspace_root=workspace_root,
        )
        if (
            test_context_report.get("split") != "test"
            or test_context_report.get("scientific_eligible") is not True
        ):
            raise ControlManifestError(
                "sealed control does not bind a scientific test context"
            )
        context_policy = _load_json(upstream / "frozen_policy_ref.json")
        expected_policy = {
            "schema_version": FROZEN_CONTROL_POLICY_SCHEMA,
            "frozen_dev_control_dependency": frozen_policy[
                "frozen_dev_control_dependency"
            ],
            "frozen_dev_context_dependency": frozen_context,
            "test_context_dependency": dependency,
            "resolved_control_config_sha256": canonical_sha256(frozen_config),
            "control_builder_code_sha256": sha256_file(__file__),
            "candidate_tier_policy_sha256": canonical_sha256(
                {
                    "quantile_policy": frozen_config["quantile_policy"],
                    "candidate_expansion_tiers_percent": frozen_config[
                        "candidate_expansion_tiers_percent"
                    ],
                    "max_block_token_delta_fraction": frozen_config[
                        "max_block_token_delta_fraction"
                    ],
                    "lexical_overlap_policy": frozen_config[
                        "lexical_overlap_policy"
                    ],
                }
            ),
        }
        if (
            resolved != frozen_config
            or frozen_policy != expected_policy
            or any(
                context_policy.get("frozen_dev_context_dependency", {}).get(key)
                != frozen_context.get(key)
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "artifact_id",
                    "payload_manifest_sha256",
                )
            )
            or frozen_meta.get("id_inputs", {}).get(
                "control_builder_code_sha256"
            )
            != sha256_file(__file__)
        ):
            raise ControlManifestError("sealed control dev config/code/context policy changed")
        if actual_files != expected_files:
            raise ControlManifestError("sealed control file set is not canonical")
    elif "frozen_policy_ref" in id_inputs or (target / "frozen_policy_ref.json").exists():
        raise ControlManifestError("non-test control cannot claim sealed lineage")
    return {
        "schema_version": "stage1-control-validation-report/v1",
        "valid": True,
        "control_build_id": build_id,
        "context_build_id": dependency["artifact_id"],
        "split": split,
        "record_count": len(records),
        "status_counts": expected_status_counts,
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_control_ref(
    control_ref: str | Path,
    *,
    tokenizer: Any | None = None,
    context_ref: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    locator, target = _resolve_locator(
        control_ref, kinds={CONTROL_ARTIFACT_KIND, TEST_CONTROL_ARTIFACT_KIND}
    )
    context_target: Path | None = None
    if context_ref is not None:
        context_locator, context_target = _resolve_locator(
            context_ref, kinds={"context", "test-context"}
        )
        dependency = _load_json(target / "context_ref.json")
        if dependency.get("artifact_id") != context_locator["artifact_id"] or dependency.get(
            "payload_manifest_sha256"
        ) != context_locator["payload_manifest_sha256"]:
            raise ControlManifestError("supplied context ref does not match control dependency")
    report = validate_control_target(
        target,
        tokenizer=tokenizer,
        context_target=context_target,
        workspace_root=workspace_root,
    )
    if report["control_build_id"] != locator["artifact_id"]:
        raise ControlManifestError("control locator ID mismatch")
    if report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise ControlManifestError("control locator payload hash mismatch")
    return report


__all__ = [
    "CONTROL_ARTIFACT_SCHEMA",
    "CONTROL_RECORD_SCHEMA",
    "CONTROL_META_SCHEMA",
    "CONTROL_POLICY",
    "FROZEN_CONTROL_POLICY_SCHEMA",
    "LEXICAL_POLICY",
    "QUANTILE_POLICY",
    "ControlManifestError",
    "ControlUnavailableError",
    "build_control_artifact",
    "build_control_record",
    "seal_test_control_artifact",
    "canonical_sha256",
    "demo_query_overlap",
    "lexical_grams",
    "lexical_normalize",
    "lexicon_query_overlap",
    "render_control_condition_item",
    "render_control_item",
    "resolve_control_config",
    "round_written_similarity",
    "validate_control_record",
    "validate_control_ref",
    "validate_control_target",
]
