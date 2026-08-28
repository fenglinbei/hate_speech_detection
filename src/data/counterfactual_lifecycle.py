"""Immutable two-stage lifecycle for Stage-1 counterfactual cohorts.

The lifecycle is deliberately separate from model generation and margin
scoring.  A proposal freezes model-independent candidates and the exact review
rubric; an immutable dual-model blind-review artifact freezes automatic
consensus and the human queue; a signed completion receives its own review
identity; only then can a final counterfactual manifest be published.

Formal builds consume the frozen context artifact layout specified by the
Stage-1 protocol.  Small synthetic fixtures are supported through an explicit
engineering-only entry point whose artifacts carry
``scientific_eligible=false``.
"""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.context_manifest import text_sha256, validate_context_record
from data.counterfactual_manifest import (
    CANDIDATE_POLICY_VERSION,
    DECLARATION_KEYS,
    ENGINEERING_CONTEXT_SCHEMA,
    FAMILY_ORDER,
    FIELDS,
    FINAL_ARTIFACT_KIND,
    FINAL_META_SCHEMA,
    FOIL_POLICY_SCHEMA,
    PROPOSAL_ARTIFACT_KIND,
    PROPOSAL_META_SCHEMA,
    PROPOSAL_ROW_KEYS,
    PROPOSAL_ROW_SCHEMA,
    RESOLVED_CONFIG_SCHEMA,
    REVIEW_ARTIFACT_KIND,
    REVIEW_FIELDS,
    REVIEW_META_SCHEMA,
    REVIEW_ROW_KEYS,
    REVIEW_ROW_SCHEMA,
    REVIEWER_DECLARATION_SCHEMA,
    RUBRIC_META_SCHEMA,
    SCHEMA_ROOT,
    SELECTION_POLICY,
    SHA256_RE,
    CounterfactualError,
    build_proposal_rows,
    candidate_id,
    canonical_sha256,
    canonical_value,
    finalize_rows,
    group_combination_frequencies,
    review_template,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    ensure_exact_file_set,
    finalize_target_atomic,
    load_json,
    load_jsonl,
    new_staging_directory,
    portable_dependency,
    resolve_dependency_target,
    resolve_locator_ref,
    sha256_file,
    validate_dependency_ref,
    validate_json_schema,
    validate_payload_manifest,
    write_bytes_atomic,
    write_canonical_json,
    write_canonical_jsonl,
    write_locator_ref,
)
from metrics.stage1_margin import replace_one_field
from utils.quadruple import canonicalize_quadruples, serialize_quadruples


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CF_POLICY_SCHEMA = "stage1-frozen-cf-policy-ref/v1"
CF_BLIND_REVIEW_ARTIFACT_KIND = "cf-blind-review"
CF_REVIEW_ID_INPUT_SCHEMA = "stage1-cf-review-id-input/v1"
CF_REVIEW_PROVENANCE_SCHEMA = "stage1-cf-review-provenance/v1"
CF_PANEL_REVIEWER_ID = "dual-blind-panel-v1"
POST_OUTCOME_RE = re.compile(
    r"(?:prediction|condition_output|model_output|model_score|margin|logprob|evaluation)",
    re.IGNORECASE,
)


class CounterfactualLifecycleError(CounterfactualError):
    """Raised when an immutable counterfactual artifact is invalid."""


def _translate_artifact_error(exc: Exception) -> CounterfactualLifecycleError:
    return CounterfactualLifecycleError(str(exc))


def _read_object(source: str | Path | Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    try:
        value = load_json(source)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    if not isinstance(value, dict):
        raise CounterfactualLifecycleError(f"{label} must be a JSON object")
    return value


def _reject_post_outcome_fields(value: Any, *, where: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if POST_OUTCOME_RE.search(key_text):
                raise CounterfactualLifecycleError(
                    f"{where} contains forbidden post-outcome field at {path}.{key_text}"
                )
            _reject_post_outcome_fields(child, where=where, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_post_outcome_fields(child, where=where, path=f"{path}[{index}]")


def _validate_foil_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(policy))
    _reject_post_outcome_fields(frozen, where="counterfactual foil policy")
    if frozen.get("schema_version") != FOIL_POLICY_SCHEMA:
        raise CounterfactualLifecycleError("unsupported counterfactual foil policy schema")
    if frozen.get("candidate_policy_version") != CANDIDATE_POLICY_VERSION:
        raise CounterfactualLifecycleError("counterfactual candidate policy version mismatch")
    if frozen.get("selection_policy") != SELECTION_POLICY:
        raise CounterfactualLifecycleError("counterfactual selection policy mismatch")
    sampling = frozen.get("sampling")
    if not isinstance(sampling, Mapping):
        raise CounterfactualLifecycleError("foil policy lacks sampling settings")
    if sampling.get("fields") != list(REVIEW_FIELDS):
        raise CounterfactualLifecycleError("review sampling fields must be target/argument")
    quota = sampling.get("field_quota")
    if not isinstance(quota, Mapping) or set(quota) != set(REVIEW_FIELDS):
        raise CounterfactualLifecycleError("sampling field_quota must cover target/argument")
    if any(not isinstance(quota[field], int) or isinstance(quota[field], bool) or quota[field] < 0 for field in REVIEW_FIELDS):
        raise CounterfactualLifecycleError("sampling field quotas must be non-negative integers")
    if sampling.get("replacement") is not False:
        raise CounterfactualLifecycleError("counterfactual cohort sampling cannot use replacement")
    if not isinstance(sampling.get("stable_hash_seed"), int) or isinstance(
        sampling.get("stable_hash_seed"), bool
    ):
        raise CounterfactualLifecycleError("sampling stable_hash_seed must be an integer")
    if frozen.get("family_order") != list(FAMILY_ORDER):
        raise CounterfactualLifecycleError("foil family order differs from frozen policy")
    review = frozen.get("review")
    if not isinstance(review, Mapping):
        raise CounterfactualLifecycleError("foil policy lacks review contract")
    expected = {
        "decision_codes": {"pass", "reject", "not_required"},
        "pass_reason_codes": {"valid-local-foil"},
        "reject_reason_codes": {
            "equivalent-to-gold",
            "invalid-boundary-fragment",
            "unsupported-by-query",
            "changes-more-than-one-field",
            "malformed-candidate",
        },
        "automatic_reason_codes": {"deterministic-label-foil"},
    }
    for key, values in expected.items():
        actual = review.get(key)
        if not isinstance(actual, list) or set(actual) != values or len(actual) != len(values):
            raise CounterfactualLifecycleError(f"foil review {key} differs from frozen contract")
    if review.get("required_fields") != list(REVIEW_FIELDS):
        raise CounterfactualLifecycleError("review-required fields differ from frozen contract")
    if set(review.get("automatic_fields", [])) != {"targeted_group", "hateful"}:
        raise CounterfactualLifecycleError("automatic fields differ from frozen contract")
    group = frozen.get("group")
    hateful = frozen.get("hateful")
    if not isinstance(group, Mapping) or group.get("construction_coverage_required") != 1.0:
        raise CounterfactualLifecycleError("targeted_group construction coverage must be 1.0")
    if not isinstance(hateful, Mapping) or hateful.get("construction_coverage_required") != 1.0:
        raise CounterfactualLifecycleError("hateful construction coverage must be 1.0")
    reference = frozen.get("reference_hardest")
    if not isinstance(reference, Mapping) or reference.get("enabled_for_primary") is not False:
        raise CounterfactualLifecycleError("reference-hardest cannot choose primary foils")
    return frozen


def _artifact_root(
    source_config: Mapping[str, Any], *, target_root: str | Path | None
) -> Path:
    if target_root is not None:
        return Path(target_root).resolve()
    raw = source_config.get("artifact_root", "exps/causal_context/stage1_p0")
    if not isinstance(raw, str) or not raw:
        raise CounterfactualLifecycleError("config artifact_root must be a non-empty path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise CounterfactualLifecycleError("config artifact_root must be repository-relative")
    return (REPOSITORY_ROOT / path).resolve()


def _resolved_config(policy: Mapping[str, Any], *, split: str, scientific_eligible: bool) -> dict[str, Any]:
    return {
        "schema_version": RESOLVED_CONFIG_SCHEMA,
        "split": split,
        "scientific_eligible": scientific_eligible,
        "foil_policy": copy.deepcopy(dict(policy)),
    }


def canonical_rubric_bytes(source: str | Path | bytes) -> bytes:
    """Canonicalize rubric bytes by CRLF-to-LF only, without trimming."""

    try:
        raw = source if isinstance(source, bytes) else Path(source).read_bytes()
        raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CounterfactualLifecycleError(f"cannot read UTF-8 CF rubric: {exc}") from exc
    return raw.replace(b"\r\n", b"\n")


def build_rubric_meta(rubric_bytes: bytes, policy: Mapping[str, Any]) -> dict[str, Any]:
    review = policy["review"]
    meta = {
        "schema_version": RUBRIC_META_SCHEMA,
        "rubric_version": "stage1-cf-review-rubric/v1",
        "policy_version": policy["candidate_policy_version"],
        "selection_policy": policy["selection_policy"],
        "rubric_body_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
        "proposal_row_schema_version": PROPOSAL_ROW_SCHEMA,
        "review_row_schema_version": REVIEW_ROW_SCHEMA,
        "reviewer_declaration_schema_version": REVIEWER_DECLARATION_SCHEMA,
        "allowed_decision_codes": list(review["decision_codes"]),
        "pass_reason_codes": list(review["pass_reason_codes"]),
        "reject_reason_codes": list(review["reject_reason_codes"]),
        "automatic_reason_codes": list(review["automatic_reason_codes"]),
        "field_contract": {
            "target": {"review_required": True, "allowed_sources": ["same-query"]},
            "argument": {"review_required": True, "allowed_sources": ["same-query"]},
            "targeted_group": {
                "review_required": False,
                "construction_coverage_required": 1.0,
            },
            "hateful": {
                "review_required": False,
                "construction_coverage_required": 1.0,
            },
        },
    }
    try:
        validate_json_schema(meta, SCHEMA_ROOT / "stage1_cf_rubric_meta_v1.schema.json")
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    return meta


def _quad_gold_hash(gold: Any) -> str:
    return hashlib.sha256(serialize_quadruples(canonicalize_quadruples(gold)).encode("utf-8")).hexdigest()


def _record_query(record: Mapping[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    query = record.get("query", record)
    if not isinstance(query, Mapping):
        raise CounterfactualLifecycleError("query record must be an object")
    query_id = str(query.get("id", ""))
    content = query.get("content")
    gold = query.get("gold", query.get("quadruples"))
    if not re.fullmatch(r"[1-9][0-9]*", query_id):
        raise CounterfactualLifecycleError(f"CF query ID is not canonical decimal: {query_id!r}")
    if not isinstance(content, str) or not content:
        raise CounterfactualLifecycleError(f"CF query {query_id} lacks content")
    try:
        quads = canonicalize_quadruples(gold)
    except Exception as exc:
        raise CounterfactualLifecycleError(f"CF query {query_id} has invalid gold: {exc}") from exc
    mappings = [
        {
            "target": quad.target,
            "argument": quad.argument,
            "targeted_group": list(quad.targeted_group),
            "hateful": quad.hateful,
        }
        for quad in quads
    ]
    return query_id, content, mappings


def _sorted_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = [copy.deepcopy(dict(record)) for record in records]
    identities = [_record_query(record)[0] for record in materialized]
    if len(identities) != len(set(identities)):
        raise CounterfactualLifecycleError("CF input contains duplicate query IDs")
    return [record for _, record in sorted(zip((int(value) for value in identities), materialized), key=lambda pair: pair[0])]


def _query_pool_map(rows: Sequence[Mapping[str, Any]], *, split: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        query_id, content, gold = _record_query(raw)
        if query_id in result:
            raise CounterfactualLifecycleError(f"duplicate {split} query-pool ID: {query_id}")
        result[query_id] = {"id": query_id, "content": content, "quadruples": gold}
    return result


def _formal_context_snapshot(
    context_ref: str | Path,
    *,
    split: str,
    workspace_root: Path,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(context_ref, ("context", "test-context"))
        dependency = portable_dependency(locator, target, workspace_root)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    if split == "test" and locator["artifact_kind"] != "test-context":
        raise CounterfactualLifecycleError("sealed test CF requires a test-context artifact")
    if split != "test" and locator["artifact_kind"] != "context":
        raise CounterfactualLifecycleError("train/dev CF requires a context artifact")
    meta_path = target / f"context_manifest.{split}.meta.json"
    records_path = target / f"context_manifest.{split}.jsonl"
    query_path = target / "catalogs" / f"query_pool.{split}.jsonl"
    train_path = target / "catalogs" / "query_pool.train.jsonl"
    required = (meta_path, records_path, query_path, train_path)
    if any(not path.is_file() or path.is_symlink() for path in required):
        missing = [path.relative_to(target).as_posix() for path in required if not path.is_file()]
        raise CounterfactualLifecycleError(
            f"formal context target lacks required frozen layout files: {missing}"
        )
    try:
        meta = load_json(meta_path)
        context_rows = load_jsonl(records_path)
        query_rows = load_jsonl(query_path)
        train_rows = load_jsonl(train_path)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    if not isinstance(meta, Mapping) or meta.get("schema_version") != "stage1-context-manifest/v1":
        raise CounterfactualLifecycleError("formal context meta has wrong schema")
    if meta.get("context_build_id") != locator["artifact_id"]:
        raise CounterfactualLifecycleError("formal context meta/locator ID mismatch")
    if meta.get("scientific_eligible") is not True:
        raise CounterfactualLifecycleError("formal CF cannot consume an engineering-only context")
    query_pool = _query_pool_map(query_rows, split=split)
    if len(context_rows) != len(query_pool):
        raise CounterfactualLifecycleError("context manifest/query pool row counts differ")
    completed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in context_rows:
        try:
            validate_context_record(raw)
        except Exception as exc:
            raise CounterfactualLifecycleError(f"invalid frozen context record: {exc}") from exc
        row = copy.deepcopy(dict(raw))
        query = row.get("query")
        if not isinstance(query, Mapping):
            raise CounterfactualLifecycleError("context record lacks query object")
        query_id = str(query.get("id", ""))
        if query_id in seen or query_id not in query_pool:
            raise CounterfactualLifecycleError("context/query-pool ID frame is not exact")
        seen.add(query_id)
        frozen_query = query_pool[query_id]
        if query.get("content") not in {None, frozen_query["content"]}:
            raise CounterfactualLifecycleError("context/query-pool content mismatch")
        if query.get("gold") is not None and canonicalize_quadruples(query["gold"]) != canonicalize_quadruples(
            frozen_query["quadruples"]
        ):
            raise CounterfactualLifecycleError("context/query-pool gold mismatch")
        content_hash = text_sha256(frozen_query["content"])
        gold_hash = _quad_gold_hash(frozen_query["quadruples"])
        if query.get("content_sha256") not in {None, content_hash}:
            raise CounterfactualLifecycleError("context query content hash mismatch")
        if query.get("gold_sha256") not in {None, gold_hash}:
            raise CounterfactualLifecycleError("context query gold hash mismatch")
        row["query"] = {
            **dict(query),
            "id": query_id,
            "content": frozen_query["content"],
            "gold": frozen_query["quadruples"],
        }
        completed.append(row)
    if seen != set(query_pool):
        raise CounterfactualLifecycleError("context/query-pool candidate set is not exact")
    train = list(_query_pool_map(train_rows, split="train").values())
    group_combination_frequencies(train)
    return {
        "scientific_eligible": True,
        "context_dependency": dependency,
        "context_target": target,
        "context_records": _sorted_records(completed),
        "train_records": _sorted_records(train),
        "context_records_sha256": sha256_file(records_path),
        "train_records_sha256": sha256_file(train_path),
    }


def _engineering_context_snapshot(
    context_records: Sequence[Mapping[str, Any]],
    train_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _reject_post_outcome_fields(context_records, where="engineering context fixture")
    _reject_post_outcome_fields(train_records, where="engineering train fixture")
    contexts = _sorted_records(context_records)
    train = _sorted_records(train_records)
    if not contexts or not train:
        raise CounterfactualLifecycleError("engineering CF fixture cannot be empty")
    group_combination_frequencies(train)
    context_bytes = canonical_jsonl_bytes(contexts, key="query_id") if all(
        "query_id" in row for row in contexts
    ) else b"".join(canonical_json_bytes(row) + b"\n" for row in contexts)
    train_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in train)
    descriptor = {
        "schema_version": ENGINEERING_CONTEXT_SCHEMA,
        "scientific_eligible": False,
        "context_records_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "train_records_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "context_record_count": len(contexts),
        "train_record_count": len(train),
    }
    descriptor["fixture_id"] = "engctx-" + canonical_sha256(descriptor)
    return {
        "scientific_eligible": False,
        "context_dependency": descriptor,
        "context_target": None,
        "context_records": contexts,
        "train_records": train,
        "context_records_sha256": descriptor["context_records_sha256"],
        "train_records_sha256": descriptor["train_records_sha256"],
    }


def _sampling_frame(snapshot: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    query_units = []
    for record in snapshot["context_records"]:
        query_id, content, gold = _record_query(record)
        record_hash = record.get("record_sha256")
        if not isinstance(record_hash, str) or not SHA256_RE.fullmatch(record_hash):
            record_hash = canonical_sha256(record)
        query_units.append(
            {
                "query_id": query_id,
                "context_record_sha256": record_hash,
                "content_sha256": text_sha256(content),
                "gold_sha256": _quad_gold_hash(gold),
                "gold_tuple_count": len(gold),
            }
        )
    frequencies = group_combination_frequencies(snapshot["train_records"])
    group_catalog = [
        {"targeted_group": list(combo), "train_frequency": count}
        for combo, count in sorted(frequencies.items(), key=lambda item: (item[0], item[1]))
    ]
    return {
        "schema_version": "stage1-cf-sampling-frame/v1",
        "split": split,
        "query_units": query_units,
        "train_group_catalog": group_catalog,
        "context_records_sha256": snapshot["context_records_sha256"],
        "train_records_sha256": snapshot["train_records_sha256"],
    }


def _code_sha256() -> str:
    pure_module = Path(__file__).with_name("counterfactual_manifest.py")
    return canonical_sha256(
        {"lifecycle_sha256": sha256_file(__file__), "pure_module_sha256": sha256_file(pure_module)}
    )


def _proposal_cohort(
    *,
    proposal_id: str,
    id_inputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    scientific_eligible: bool,
) -> dict[str, Any]:
    review_rows = [row for row in rows if row["review_required"]]
    automatic_rows = [row for row in rows if not row["review_required"]]
    return {
        "schema_version": PROPOSAL_META_SCHEMA,
        "cf_proposal_id": proposal_id,
        "scientific_eligible": scientific_eligible,
        "id_inputs": copy.deepcopy(dict(id_inputs)),
        "summary": copy.deepcopy(dict(summary)),
        "candidate_count": len(rows),
        "candidate_ids_sha256": canonical_sha256([row["candidate_id"] for row in rows]),
        "review_candidate_ids": [row["candidate_id"] for row in review_rows],
        "automatic_candidate_ids": [row["candidate_id"] for row in automatic_rows],
        "review_family_counts": dict(sorted(Counter(row["family"] for row in review_rows).items())),
    }


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]], *, proposal_id: str
) -> None:
    if not rows:
        raise CounterfactualLifecycleError("counterfactual proposal has no candidates")
    identities: list[str] = []
    automatic_units: Counter[tuple[str, int, str]] = Counter()
    for raw in rows:
        row = dict(raw)
        if set(row) != PROPOSAL_ROW_KEYS:
            raise CounterfactualLifecycleError("proposal candidate has non-canonical fields")
        try:
            validate_json_schema(row, SCHEMA_ROOT / "stage1_cf_proposal_v1.schema.json")
        except TrainingArtifactError as exc:
            raise _translate_artifact_error(exc) from exc
        if row["cf_proposal_id"] != proposal_id:
            raise CounterfactualLifecycleError("candidate proposal ID mismatch")
        expected_id = candidate_id(
            query_id=row["query_id"],
            gold_sha256=row["gold_sha256"],
            tuple_index=row["tuple_index"],
            field=row["field"],
            candidate_value=row["candidate_value"],
            family=row["family"],
            source=row["source"],
            source_id=row["source_id"],
        )
        if row["candidate_id"] != expected_id:
            raise CounterfactualLifecycleError("candidate ID cannot be recomputed")
        if row["candidate_value_sha256"] != canonical_sha256(
            canonical_value(row["field"], row["candidate_value"])
        ):
            raise CounterfactualLifecycleError("candidate value hash mismatch")
        if row["gold_sha256"] != _quad_gold_hash(row["gold_quadruples"]):
            raise CounterfactualLifecycleError("candidate gold hash mismatch")
        quads = canonicalize_quadruples(row["gold_quadruples"])
        if not 0 <= row["tuple_index"] < len(quads):
            raise CounterfactualLifecycleError("candidate tuple index is outside gold")
        gold_value: Any = getattr(quads[row["tuple_index"]], row["field"])
        if row["field"] == "targeted_group":
            gold_value = list(gold_value)
        if row["gold_value"] != gold_value:
            raise CounterfactualLifecycleError("candidate gold value mismatch")
        try:
            replace_one_field(
                quads,
                tuple_index=row["tuple_index"],
                field=row["field"],
                candidate_value=row["candidate_value"],
            )
        except Exception as exc:
            raise CounterfactualLifecycleError(
                f"candidate does not change exactly one requested field: {exc}"
            ) from exc
        if row["review_required"] is not (row["field"] in REVIEW_FIELDS):
            raise CounterfactualLifecycleError("candidate review-required flag is field-inconsistent")
        if row["review_required"] and row["source"] != "same-query":
            raise CounterfactualLifecycleError("primary target/argument candidate is not query-local")
        if not row["review_required"]:
            automatic_units[(row["query_id"], row["tuple_index"], row["field"])] += 1
        identities.append(row["candidate_id"])
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise CounterfactualLifecycleError("candidate IDs must be unique and sorted")
    tuple_units = {
        (row["query_id"], row["tuple_index"])
        for row in rows
        if row["field"] in {"targeted_group", "hateful"}
    }
    expected = {(query_id, index, field) for query_id, index in tuple_units for field in ("targeted_group", "hateful")}
    if set(automatic_units) != expected or any(automatic_units[unit] != 1 for unit in expected):
        raise CounterfactualLifecycleError("group/hate automatic candidate coverage is not exactly 100%")


def _load_engineering_snapshot(target: Path) -> dict[str, Any]:
    try:
        contexts = load_jsonl(target / "engineering_context.jsonl")
        train = load_jsonl(target / "engineering_train.jsonl")
        descriptor = load_json(target / "context_ref.json")
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    expected = _engineering_context_snapshot(contexts, train)
    if descriptor != expected["context_dependency"]:
        raise CounterfactualLifecycleError("engineering context descriptor cannot be reproduced")
    return expected


def _proposal_file_set(
    *, split: str, engineering: bool, sealed: bool = False
) -> set[str]:
    files = {
        "config.resolved.json",
        "context_ref.json",
        "cohort.json",
        f"candidates.{split}.jsonl",
        "review_rubric.md",
        "review_rubric.meta.json",
        "review_template.jsonl",
        "provenance.json",
        "payload_manifest.json",
    }
    if engineering:
        files |= {"engineering_context.jsonl", "engineering_train.jsonl"}
    if sealed:
        files.add("frozen_policy_ref.json")
    return files


def _dependency_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
        )
    }


def _sealed_cf_lineage(
    *,
    frozen_cf_ref: str | Path,
    test_context_ref: str | Path,
    workspace_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    """Resolve the model-independent dev CF policy copied into sealed test."""

    validate_cf_ref(frozen_cf_ref, workspace_root=workspace_root)
    try:
        frozen_locator, frozen_target = resolve_locator_ref(
            frozen_cf_ref, FINAL_ARTIFACT_KIND
        )
        test_context_locator, test_context_target = resolve_locator_ref(
            test_context_ref, "test-context"
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    frozen_config = load_json(frozen_target / "config.resolved.json")
    frozen_meta = load_json(frozen_target / "cf_manifest.meta.json")
    frozen_provenance = load_json(frozen_target / "provenance.json")
    if (
        frozen_config.get("split") != "dev"
        or frozen_config.get("scientific_eligible") is not True
        or frozen_meta.get("split") != "dev"
    ):
        raise CounterfactualLifecycleError(
            "sealed test requires a scientific frozen dev CF"
        )
    if (
        frozen_provenance.get("selection_policy_sha256")
        != _final_selection_policy_hash()
        or frozen_provenance.get("finalizer_code_sha256") != _code_sha256()
    ):
        raise CounterfactualLifecycleError(
            "frozen dev CF selection/finalizer code differs from current seal code"
        )
    frozen_context = load_json(frozen_target / "context_ref.json")
    test_context_policy = load_json(test_context_target / "frozen_policy_ref.json")
    frozen_dev_context = test_context_policy.get("frozen_dev_context_dependency")
    if not isinstance(frozen_dev_context, Mapping) or _dependency_identity(
        frozen_dev_context
    ) != _dependency_identity(frozen_context):
        raise CounterfactualLifecycleError(
            "test context does not descend from frozen dev CF context"
        )
    try:
        proposal_dependency = validate_dependency_ref(
            load_json(frozen_target / "proposal_ref.json"),
            expected_kind=PROPOSAL_ARTIFACT_KIND,
        )
        frozen_proposal = resolve_dependency_target(
            proposal_dependency, workspace_root
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    proposal_provenance = load_json(frozen_proposal / "provenance.json")
    rubric_bytes = (frozen_proposal / "review_rubric.md").read_bytes()
    rubric_meta = load_json(frozen_proposal / "review_rubric.meta.json")
    if (
        proposal_provenance.get("proposal_code_sha256") != _code_sha256()
        or rubric_bytes != canonical_rubric_bytes(rubric_bytes)
        or rubric_meta != build_rubric_meta(rubric_bytes, frozen_config["foil_policy"])
    ):
        raise CounterfactualLifecycleError(
            "frozen dev CF proposal code/rubric cannot be replayed"
        )
    test_context_dependency = portable_dependency(
        test_context_locator, test_context_target, workspace_root
    )
    lineage = {
        "schema_version": FROZEN_CF_POLICY_SCHEMA,
        "frozen_dev_cf_dependency": portable_dependency(
            frozen_locator, frozen_target, workspace_root
        ),
        "frozen_dev_context_dependency": dict(frozen_context),
        "test_context_dependency": test_context_dependency,
        "foil_config_sha256": canonical_sha256(frozen_config["foil_policy"]),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "proposal_code_sha256": _code_sha256(),
        "selection_policy_sha256": _final_selection_policy_hash(),
        "finalizer_code_sha256": _code_sha256(),
    }
    return lineage, dict(frozen_config["foil_policy"]), rubric_bytes


def _validate_embedded_sealed_cf_lineage(
    lineage: Mapping[str, Any],
    *,
    test_context_dependency: Mapping[str, Any],
    test_context_target: Path,
    workspace_root: Path,
) -> tuple[dict[str, Any], bytes]:
    expected_keys = {
        "schema_version",
        "frozen_dev_cf_dependency",
        "frozen_dev_context_dependency",
        "test_context_dependency",
        "foil_config_sha256",
        "review_rubric_sha256",
        "review_rubric_meta_sha256",
        "candidate_policy_version",
        "proposal_code_sha256",
        "selection_policy_sha256",
        "finalizer_code_sha256",
    }
    if set(lineage) != expected_keys or lineage.get(
        "schema_version"
    ) != FROZEN_CF_POLICY_SCHEMA:
        raise CounterfactualLifecycleError("sealed CF frozen-policy ref is non-canonical")
    try:
        frozen_dependency = validate_dependency_ref(
            lineage["frozen_dev_cf_dependency"], expected_kind=FINAL_ARTIFACT_KIND
        )
        frozen_target = resolve_dependency_target(frozen_dependency, workspace_root)
    except (TrainingArtifactError, KeyError) as exc:
        raise _translate_artifact_error(exc) from exc
    validate_cf_target(frozen_target, workspace_root=workspace_root)
    frozen_config = load_json(frozen_target / "config.resolved.json")
    frozen_meta = load_json(frozen_target / "cf_manifest.meta.json")
    frozen_provenance = load_json(frozen_target / "provenance.json")
    if (
        frozen_config.get("split") != "dev"
        or frozen_config.get("scientific_eligible") is not True
        or frozen_meta.get("split") != "dev"
    ):
        raise CounterfactualLifecycleError("sealed CF lineage does not bind scientific dev")
    frozen_context = load_json(frozen_target / "context_ref.json")
    test_context_policy = load_json(test_context_target / "frozen_policy_ref.json")
    if (
        _dependency_identity(test_context_dependency)
        != _dependency_identity(lineage["test_context_dependency"])
        or _dependency_identity(frozen_context)
        != _dependency_identity(lineage["frozen_dev_context_dependency"])
        or _dependency_identity(
            test_context_policy.get("frozen_dev_context_dependency", {})
        )
        != _dependency_identity(frozen_context)
    ):
        raise CounterfactualLifecycleError("sealed CF context lineage is inconsistent")
    try:
        frozen_proposal_dependency = validate_dependency_ref(
            load_json(frozen_target / "proposal_ref.json"),
            expected_kind=PROPOSAL_ARTIFACT_KIND,
        )
        frozen_proposal = resolve_dependency_target(
            frozen_proposal_dependency, workspace_root
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    rubric_bytes = (frozen_proposal / "review_rubric.md").read_bytes()
    rubric_meta = load_json(frozen_proposal / "review_rubric.meta.json")
    proposal_provenance = load_json(frozen_proposal / "provenance.json")
    expected = {
        "schema_version": FROZEN_CF_POLICY_SCHEMA,
        "frozen_dev_cf_dependency": dict(frozen_dependency),
        "frozen_dev_context_dependency": dict(frozen_context),
        "test_context_dependency": dict(test_context_dependency),
        "foil_config_sha256": canonical_sha256(frozen_config["foil_policy"]),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "proposal_code_sha256": _code_sha256(),
        "selection_policy_sha256": _final_selection_policy_hash(),
        "finalizer_code_sha256": _code_sha256(),
    }
    if (
        dict(lineage) != expected
        or proposal_provenance.get("proposal_code_sha256") != _code_sha256()
        or frozen_provenance.get("finalizer_code_sha256") != _code_sha256()
        or frozen_provenance.get("selection_policy_sha256")
        != _final_selection_policy_hash()
        or rubric_meta
        != build_rubric_meta(rubric_bytes, frozen_config["foil_policy"])
    ):
        raise CounterfactualLifecycleError("sealed CF policy/code/rubric differs from frozen dev")
    return dict(frozen_config["foil_policy"]), rubric_bytes


def _load_proposal_snapshot(
    target: Path,
    *,
    workspace_root: Path,
    context_target_override: Path | None = None,
) -> dict[str, Any]:
    config = load_json(target / "config.resolved.json")
    if not isinstance(config, Mapping) or config.get("schema_version") != RESOLVED_CONFIG_SCHEMA:
        raise CounterfactualLifecycleError("proposal resolved config has wrong schema")
    split = config.get("split")
    if split not in {"dev", "test"}:
        raise CounterfactualLifecycleError("CF proposal split must be dev/test")
    scientific = config.get("scientific_eligible")
    if not isinstance(scientific, bool):
        raise CounterfactualLifecycleError("proposal scientific_eligible must be boolean")
    policy = _validate_foil_policy(config.get("foil_policy", {}))
    dependency = load_json(target / "context_ref.json")
    if scientific:
        try:
            validate_dependency_ref(dependency, expected_kind=("context", "test-context"))
            context_target = context_target_override or resolve_dependency_target(
                dependency, workspace_root
            )
        except TrainingArtifactError as exc:
            raise _translate_artifact_error(exc) from exc
        # Reconstruct a temporary locator-free snapshot from the strict target.
        kind = dependency["artifact_kind"]
        expected_kind = "test-context" if split == "test" else "context"
        if kind != expected_kind or context_target.name != dependency["artifact_id"]:
            raise CounterfactualLifecycleError("proposal context dependency kind/ID mismatch")
        if validate_payload_manifest(context_target) != dependency["payload_manifest_sha256"]:
            raise CounterfactualLifecycleError("proposal context dependency payload mismatch")
        meta_path = context_target / f"context_manifest.{split}.meta.json"
        records_path = context_target / f"context_manifest.{split}.jsonl"
        query_path = context_target / "catalogs" / f"query_pool.{split}.jsonl"
        train_path = context_target / "catalogs" / "query_pool.train.jsonl"
        if any(not path.is_file() for path in (meta_path, records_path, query_path, train_path)):
            raise CounterfactualLifecycleError("proposal context dependency lacks formal layout")
        meta = load_json(meta_path)
        if meta.get("schema_version") != "stage1-context-manifest/v1" or meta.get(
            "context_build_id"
        ) != dependency["artifact_id"]:
            raise CounterfactualLifecycleError("proposal context dependency meta mismatch")
        context_rows = load_jsonl(records_path)
        query_pool = _query_pool_map(load_jsonl(query_path), split=split)
        completed = []
        for raw in context_rows:
            validate_context_record(raw)
            query = raw.get("query", {})
            query_id = str(query.get("id", ""))
            if query_id not in query_pool:
                raise CounterfactualLifecycleError("proposal context/query frame mismatch")
            item = copy.deepcopy(dict(raw))
            item["query"] = {
                **dict(query),
                "id": query_id,
                "content": query_pool[query_id]["content"],
                "gold": query_pool[query_id]["quadruples"],
            }
            completed.append(item)
        if {str(row.get("query", {}).get("id", "")) for row in completed} != set(query_pool):
            raise CounterfactualLifecycleError("proposal context/query candidate set mismatch")
        train = list(_query_pool_map(load_jsonl(train_path), split="train").values())
        snapshot = {
            "scientific_eligible": True,
            "context_dependency": dependency,
            "context_target": context_target,
            "context_records": _sorted_records(completed),
            "train_records": _sorted_records(train),
            "context_records_sha256": sha256_file(records_path),
            "train_records_sha256": sha256_file(train_path),
        }
    else:
        snapshot = _load_engineering_snapshot(target)
        if dependency != snapshot["context_dependency"]:
            raise CounterfactualLifecycleError("engineering proposal context descriptor mismatch")
    return {"config": dict(config), "policy": policy, "split": split, "snapshot": snapshot}


def validate_proposal_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    context_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        validate_payload_manifest(target)
        loaded = _load_proposal_snapshot(
            target,
            workspace_root=Path(workspace_root).resolve(),
            context_target_override=Path(context_target) if context_target is not None else None,
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    config = loaded["config"]
    policy = loaded["policy"]
    split = loaded["split"]
    snapshot = loaded["snapshot"]
    engineering = not snapshot["scientific_eligible"]
    sealed = split == "test"
    if sealed and engineering:
        raise CounterfactualLifecycleError("sealed test CF proposal must be scientific")
    try:
        ensure_exact_file_set(
            target,
            _proposal_file_set(
                split=split, engineering=engineering, sealed=sealed
            ),
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    rubric_bytes = (target / "review_rubric.md").read_bytes()
    if rubric_bytes != canonical_rubric_bytes(rubric_bytes):
        raise CounterfactualLifecycleError("stored review rubric is not canonical CRLF-to-LF bytes")
    rubric_meta = load_json(target / "review_rubric.meta.json")
    expected_rubric_meta = build_rubric_meta(rubric_bytes, policy)
    if rubric_meta != expected_rubric_meta:
        raise CounterfactualLifecycleError("review rubric meta/hash cannot be reproduced")
    frame = _sampling_frame(snapshot, split=split)
    code_sha = _code_sha256()
    frozen_policy: dict[str, Any] | None = None
    if sealed:
        raw_frozen = load_json(target / "frozen_policy_ref.json")
        if not isinstance(raw_frozen, Mapping):
            raise CounterfactualLifecycleError("sealed proposal frozen policy must be an object")
        frozen_policy = dict(raw_frozen)
        expected_policy, expected_rubric = _validate_embedded_sealed_cf_lineage(
            frozen_policy,
            test_context_dependency=snapshot["context_dependency"],
            test_context_target=snapshot["context_target"],
            workspace_root=Path(workspace_root).resolve(),
        )
        if policy != expected_policy or rubric_bytes != expected_rubric:
            raise CounterfactualLifecycleError(
                "sealed proposal policy/rubric is not the frozen dev copy"
            )
    id_inputs = {
        "context_dependency": snapshot["context_dependency"],
        "foil_config_sha256": canonical_sha256(policy),
        "sampling_frame_sha256": canonical_sha256(frame),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "proposal_code_sha256": code_sha,
    }
    if frozen_policy is not None:
        id_inputs["frozen_policy_ref"] = frozen_policy
    proposal_id = "cfp-" + canonical_sha256(id_inputs)
    if require_directory_name and target.name != proposal_id:
        raise CounterfactualLifecycleError("proposal target directory name differs from proposal ID")
    quota = policy["sampling"]["field_quota"]
    rows, summary = build_proposal_rows(
        cf_proposal_id=proposal_id,
        context_records=snapshot["context_records"],
        train_records=snapshot["train_records"],
        field_quota={field: int(quota[field]) for field in REVIEW_FIELDS},
        stable_hash_seed=int(policy["sampling"]["stable_hash_seed"]),
    )
    _validate_candidate_rows(rows, proposal_id=proposal_id)
    stored_rows = load_jsonl(target / f"candidates.{split}.jsonl")
    if stored_rows != rows:
        raise CounterfactualLifecycleError("proposal candidates cannot be deterministically replayed")
    if (target / f"candidates.{split}.jsonl").read_bytes() != canonical_jsonl_bytes(
        rows, key="candidate_id"
    ):
        raise CounterfactualLifecycleError("proposal candidate JSONL is not canonical")
    expected_cohort = _proposal_cohort(
        proposal_id=proposal_id,
        id_inputs=id_inputs,
        rows=rows,
        summary=summary,
        scientific_eligible=snapshot["scientific_eligible"],
    )
    cohort = load_json(target / "cohort.json")
    if cohort != expected_cohort:
        raise CounterfactualLifecycleError("proposal cohort/meta cannot be reproduced")
    templates = review_template(rows)
    if load_jsonl(target / "review_template.jsonl") != templates:
        raise CounterfactualLifecycleError("proposal review template cannot be reproduced")
    if (target / "review_template.jsonl").read_bytes() != canonical_jsonl_bytes(
        templates, key="candidate_id"
    ):
        raise CounterfactualLifecycleError("proposal review template is not canonical")
    provenance = load_json(target / "provenance.json")
    expected_provenance = {
        "schema_version": "stage1-cf-proposal-provenance/v1",
        "cf_proposal_id": proposal_id,
        "scientific_eligible": snapshot["scientific_eligible"],
        "context_dependency": snapshot["context_dependency"],
        "sampling_frame_sha256": canonical_sha256(frame),
        "candidate_rows_sha256": hashlib.sha256(canonical_jsonl_bytes(rows, key="candidate_id")).hexdigest(),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "proposal_code_sha256": code_sha,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
    }
    if frozen_policy is not None:
        expected_provenance["frozen_policy_ref"] = frozen_policy
    if provenance != expected_provenance:
        raise CounterfactualLifecycleError("proposal provenance cannot be reproduced")
    if config != _resolved_config(
        policy, split=split, scientific_eligible=snapshot["scientific_eligible"]
    ):
        raise CounterfactualLifecycleError("proposal resolved config is non-canonical")
    return {
        "schema_version": "stage1-cf-proposal-validation/v1",
        "valid": True,
        "cf_proposal_id": proposal_id,
        "split": split,
        "scientific_eligible": snapshot["scientific_eligible"],
        "candidate_count": len(rows),
        "review_candidate_count": summary["review_candidate_count"],
        "gold_tuple_count": summary["gold_tuple_count"],
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def propose_counterfactual_artifact(
    *,
    write_ref: str | Path,
    config: str | Path | Mapping[str, Any] | None = None,
    foil_policy: str | Path | Mapping[str, Any] | None = None,
    review_rubric: str | Path | bytes | None = None,
    split: str = "dev",
    context_ref: str | Path | None = None,
    frozen_cf_ref: str | Path | None = None,
    engineering_context_records: Sequence[Mapping[str, Any]] | None = None,
    engineering_train_records: Sequence[Mapping[str, Any]] | None = None,
    target_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Publish an immutable model-blind proposal artifact."""

    if split not in {"dev", "test"}:
        raise CounterfactualLifecycleError("CF proposal split must be dev/test")
    # ``target_root`` controls only where the new immutable target is published.
    # Portable upstream dependencies are always resolved relative to the
    # repository workspace unless the caller explicitly supplies another root.
    workspace = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else REPOSITORY_ROOT
    )
    frozen_policy: dict[str, Any] | None = None
    if split == "test":
        if frozen_cf_ref is None or context_ref is None:
            raise CounterfactualLifecycleError(
                "sealed test propose-cf requires --frozen-cf-ref and test context"
            )
        if config is not None or foil_policy is not None or review_rubric is not None:
            raise CounterfactualLifecycleError(
                "sealed test CF policy/config/rubric must come only from frozen dev CF"
            )
        frozen_policy, policy, rubric_source = _sealed_cf_lineage(
            frozen_cf_ref=frozen_cf_ref,
            test_context_ref=context_ref,
            workspace_root=workspace,
        )
        source_config: dict[str, Any] = {}
    else:
        if frozen_cf_ref is not None:
            raise CounterfactualLifecycleError("dev propose-cf cannot use --frozen-cf-ref")
        if config is None or foil_policy is None or review_rubric is None:
            raise CounterfactualLifecycleError(
                "dev propose-cf requires config, foil policy, and review rubric"
            )
        source_config = _read_object(config, label="CF source config")
        policy = _validate_foil_policy(
            _read_object(foil_policy, label="CF foil policy")
        )
        rubric_source = review_rubric
    root = _artifact_root(source_config, target_root=target_root)
    using_formal = context_ref is not None
    using_fixture = engineering_context_records is not None or engineering_train_records is not None
    if split == "test" and using_fixture:
        raise CounterfactualLifecycleError("sealed test CF cannot use engineering fixtures")
    if using_formal == using_fixture:
        raise CounterfactualLifecycleError(
            "provide exactly one formal context-ref or an explicit engineering fixture pair"
        )
    if using_formal:
        snapshot = _formal_context_snapshot(
            context_ref, split=split, workspace_root=workspace  # type: ignore[arg-type]
        )
    else:
        if engineering_context_records is None or engineering_train_records is None:
            raise CounterfactualLifecycleError("engineering context and train fixtures are both required")
        snapshot = _engineering_context_snapshot(
            engineering_context_records, engineering_train_records
        )
    rubric_bytes = canonical_rubric_bytes(rubric_source)
    rubric_meta = build_rubric_meta(rubric_bytes, policy)
    frame = _sampling_frame(snapshot, split=split)
    code_sha = _code_sha256()
    id_inputs = {
        "context_dependency": snapshot["context_dependency"],
        "foil_config_sha256": canonical_sha256(policy),
        "sampling_frame_sha256": canonical_sha256(frame),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "candidate_policy_version": CANDIDATE_POLICY_VERSION,
        "proposal_code_sha256": code_sha,
    }
    if frozen_policy is not None:
        id_inputs["frozen_policy_ref"] = frozen_policy
    proposal_id = "cfp-" + canonical_sha256(id_inputs)
    quota = policy["sampling"]["field_quota"]
    rows, summary = build_proposal_rows(
        cf_proposal_id=proposal_id,
        context_records=snapshot["context_records"],
        train_records=snapshot["train_records"],
        field_quota={field: int(quota[field]) for field in REVIEW_FIELDS},
        stable_hash_seed=int(policy["sampling"]["stable_hash_seed"]),
    )
    _validate_candidate_rows(rows, proposal_id=proposal_id)
    cohort = _proposal_cohort(
        proposal_id=proposal_id,
        id_inputs=id_inputs,
        rows=rows,
        summary=summary,
        scientific_eligible=snapshot["scientific_eligible"],
    )
    resolved = _resolved_config(
        policy, split=split, scientific_eligible=snapshot["scientific_eligible"]
    )
    provenance = {
        "schema_version": "stage1-cf-proposal-provenance/v1",
        "cf_proposal_id": proposal_id,
        "scientific_eligible": snapshot["scientific_eligible"],
        "context_dependency": snapshot["context_dependency"],
        "sampling_frame_sha256": canonical_sha256(frame),
        "candidate_rows_sha256": hashlib.sha256(canonical_jsonl_bytes(rows, key="candidate_id")).hexdigest(),
        "review_rubric_sha256": rubric_meta["rubric_body_sha256"],
        "review_rubric_meta_sha256": canonical_sha256(rubric_meta),
        "proposal_code_sha256": code_sha,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
    }
    if frozen_policy is not None:
        provenance["frozen_policy_ref"] = frozen_policy
    dirname = "test_cf_proposals" if split == "test" else "cf_proposals"
    parent = root / dirname
    target = parent / proposal_id
    staging = new_staging_directory(parent, proposal_id)
    try:
        write_canonical_json(staging / "config.resolved.json", resolved)
        write_canonical_json(staging / "context_ref.json", snapshot["context_dependency"])
        write_canonical_json(staging / "cohort.json", cohort)
        write_canonical_jsonl(staging / f"candidates.{split}.jsonl", rows, key="candidate_id")
        write_bytes_atomic(staging / "review_rubric.md", rubric_bytes)
        write_canonical_json(staging / "review_rubric.meta.json", rubric_meta)
        write_canonical_jsonl(staging / "review_template.jsonl", review_template(rows), key="candidate_id")
        write_canonical_json(staging / "provenance.json", provenance)
        if frozen_policy is not None:
            write_canonical_json(staging / "frozen_policy_ref.json", frozen_policy)
        if not snapshot["scientific_eligible"]:
            context_rows = snapshot["context_records"]
            train_rows = snapshot["train_records"]
            write_bytes_atomic(
                staging / "engineering_context.jsonl",
                b"".join(canonical_json_bytes(row) + b"\n" for row in context_rows),
            )
            write_bytes_atomic(
                staging / "engineering_train.jsonl",
                b"".join(canonical_json_bytes(row) + b"\n" for row in train_rows),
            )
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_proposal_target(
                candidate,
                workspace_root=workspace,
                context_target=snapshot["context_target"],
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise _translate_artifact_error(exc) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        return write_locator_ref(
            write_ref,
            artifact_kind=PROPOSAL_ARTIFACT_KIND,
            artifact_id=proposal_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc


def validate_proposal_ref(
    proposal_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    context_ref: str | Path | None = None,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(proposal_ref, PROPOSAL_ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    context_target = None
    if context_ref is not None:
        try:
            context_locator, context_target = resolve_locator_ref(
                context_ref, ("context", "test-context")
            )
        except TrainingArtifactError as exc:
            raise _translate_artifact_error(exc) from exc
        dependency = load_json(target / "context_ref.json")
        if dependency.get("artifact_id") != context_locator["artifact_id"] or dependency.get(
            "payload_manifest_sha256"
        ) != context_locator["payload_manifest_sha256"]:
            raise CounterfactualLifecycleError("supplied context ref differs from proposal dependency")
    report = validate_proposal_target(
        target, workspace_root=workspace_root, context_target=context_target
    )
    if report["cf_proposal_id"] != locator["artifact_id"]:
        raise CounterfactualLifecycleError("proposal locator ID mismatch")
    if report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise CounterfactualLifecycleError("proposal locator payload mismatch")
    return report


def completed_review_rows_sha256(review_rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(review_rows, key="candidate_id")).hexdigest()


def _canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl_bytes(rows, key="candidate_id")).hexdigest()


def _inspect_cf_blind_review_target(
    *,
    blind_review_target: Path,
    blind_review_dependency: Mapping[str, Any],
    proposal_target: Path,
    proposal_dependency: Mapping[str, Any],
    workspace_root: Path,
) -> dict[str, Any]:
    """Deeply replay one blind-review target and freeze its review partitions."""

    try:
        from review.cf_blind_review import (
            BlindReviewError,
            validate_cf_blind_review_target,
        )

        dependency = validate_dependency_ref(
            blind_review_dependency,
            expected_kind=CF_BLIND_REVIEW_ARTIFACT_KIND,
        )
        resolved = blind_review_target.resolve()
        resolved.relative_to(workspace_root)
        if (
            resolved.name != dependency["artifact_id"]
            or validate_payload_manifest(resolved)
            != dependency["payload_manifest_sha256"]
        ):
            raise CounterfactualLifecycleError(
                "CF blind-review dependency does not match its immutable target"
            )
        embedded_proposal = validate_dependency_ref(
            load_json(resolved / "proposal_ref.json"),
            expected_kind=PROPOSAL_ARTIFACT_KIND,
        )
        if embedded_proposal != dict(proposal_dependency):
            raise CounterfactualLifecycleError(
                "CF blind review targets a different proposal dependency"
            )
        report = validate_cf_blind_review_target(
            resolved,
            workspace_root=workspace_root,
            proposal_target=proposal_target,
        )
        if (
            report.get("cf_blind_review_id") != dependency["artifact_id"]
            or report.get("payload_manifest_sha256")
            != dependency["payload_manifest_sha256"]
        ):
            raise CounterfactualLifecycleError(
                "CF blind-review report differs from its dependency"
            )
        auto_rows = load_jsonl(resolved / "auto_review.jsonl")
        queue_rows = load_jsonl(resolved / "human_queue.jsonl")
        templates = load_jsonl(resolved / "human_review_template.jsonl")
        if validate_payload_manifest(resolved) != dependency["payload_manifest_sha256"]:
            raise CounterfactualLifecycleError(
                "CF blind-review target changed while its frozen partitions were read"
            )
    except CounterfactualLifecycleError:
        raise
    except (BlindReviewError, TrainingArtifactError, OSError, KeyError, ValueError) as exc:
        raise CounterfactualLifecycleError(
            f"cannot replay CF blind-review dependency: {exc}"
        ) from exc

    queue_ids = [str(row.get("candidate_id")) for row in queue_rows]
    template_ids = [str(row.get("candidate_id")) for row in templates]
    auto_ids = [str(row.get("candidate_id")) for row in auto_rows]
    if (
        len(queue_ids) != len(set(queue_ids))
        or len(template_ids) != len(set(template_ids))
        or len(auto_ids) != len(set(auto_ids))
        or set(queue_ids) != set(template_ids)
        or set(queue_ids).intersection(auto_ids)
    ):
        raise CounterfactualLifecycleError(
            "CF blind-review automatic/human partitions are not exact and disjoint"
        )
    return {
        "target": resolved,
        "dependency": dict(dependency),
        "proposal_dependency": dict(embedded_proposal),
        "auto_rows": [dict(row) for row in auto_rows],
        "queue_rows": [dict(row) for row in queue_rows],
        "queue_candidate_ids": queue_ids,
        "auto_review_rows_sha256": _canonical_rows_sha256(auto_rows),
        "human_queue_rows_sha256": _canonical_rows_sha256(queue_rows),
    }


def _resolve_cf_blind_review_snapshot(
    *,
    blind_review_ref: str | Path,
    workspace_root: Path,
    proposal_target: Path | None = None,
    proposal_dependency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        locator, blind_target = resolve_locator_ref(
            blind_review_ref, CF_BLIND_REVIEW_ARTIFACT_KIND
        )
        blind_dependency = portable_dependency(locator, blind_target, workspace_root)
        embedded_proposal = validate_dependency_ref(
            load_json(blind_target / "proposal_ref.json"),
            expected_kind=PROPOSAL_ARTIFACT_KIND,
        )
        selected_proposal = (
            proposal_target.resolve()
            if proposal_target is not None
            else resolve_dependency_target(embedded_proposal, workspace_root)
        )
        selected_dependency = (
            dict(proposal_dependency)
            if proposal_dependency is not None
            else dict(embedded_proposal)
        )
    except (TrainingArtifactError, OSError, KeyError, ValueError) as exc:
        raise CounterfactualLifecycleError(
            f"cannot resolve CF blind-review ref: {exc}"
        ) from exc
    return _inspect_cf_blind_review_target(
        blind_review_target=blind_target,
        blind_review_dependency=blind_dependency,
        proposal_target=selected_proposal,
        proposal_dependency=selected_dependency,
        workspace_root=workspace_root,
    )


def _validate_blind_review_union(
    *,
    proposal_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    reviewer_id: str,
    policy: Mapping[str, Any],
    blind_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact frozen auto rows plus completed rows for every queued item."""

    if reviewer_id != CF_PANEL_REVIEWER_ID:
        raise CounterfactualLifecycleError(
            "CF review must use the frozen dual-blind panel reviewer identity"
        )
    _validate_completed_reviews(
        proposal_rows, review_rows, reviewer_id=reviewer_id, policy=policy
    )
    proposal_ids = {str(row["candidate_id"]) for row in proposal_rows}
    review_by_id = {str(row["candidate_id"]): dict(row) for row in review_rows}
    auto_by_id = {
        str(row["candidate_id"]): dict(row) for row in blind_snapshot["auto_rows"]
    }
    queue_ids = set(blind_snapshot["queue_candidate_ids"])
    if set(auto_by_id).intersection(queue_ids) or set(auto_by_id).union(queue_ids) != proposal_ids:
        raise CounterfactualLifecycleError(
            "CF blind-review partitions do not exactly cover the proposal"
        )
    for candidate_id, frozen in auto_by_id.items():
        if review_by_id.get(candidate_id) != frozen:
            raise CounterfactualLifecycleError(
                "completed CF review changed a frozen automatic consensus row"
            )
    human_rows = [review_by_id[candidate_id] for candidate_id in sorted(queue_ids)]
    if {row["candidate_id"] for row in human_rows} != queue_ids:
        raise CounterfactualLifecycleError(
            "completed CF human review differs from the exact frozen queue"
        )
    return {
        "auto_review_count": len(auto_by_id),
        "human_review_count": len(human_rows),
        "human_completed_rows_sha256": _canonical_rows_sha256(human_rows),
    }


def prepare_reviewer_declaration(
    *,
    blind_review_ref: str | Path,
    review_file: str | Path,
    reviewer_id: str,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Prepare an unsigned declaration only for an exact frozen blind partition."""

    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise CounterfactualLifecycleError("reviewer_id must be non-empty")
    root = Path(workspace_root).resolve()
    snapshot = _resolve_cf_blind_review_snapshot(
        blind_review_ref=blind_review_ref,
        workspace_root=root,
    )
    target = resolve_dependency_target(snapshot["proposal_dependency"], root)
    rows = load_jsonl(review_file)
    split = load_json(target / "config.resolved.json")["split"]
    proposals = load_jsonl(target / f"candidates.{split}.jsonl")
    policy = load_json(target / "config.resolved.json")["foil_policy"]
    _validate_blind_review_union(
        proposal_rows=proposals,
        review_rows=rows,
        reviewer_id=reviewer_id,
        policy=policy,
        blind_snapshot=snapshot,
    )
    rubric_meta = load_json(target / "review_rubric.meta.json")
    return {
        "schema_version": REVIEWER_DECLARATION_SCHEMA,
        "cf_proposal_id": snapshot["proposal_dependency"]["artifact_id"],
        "cf_blind_review_dependency": snapshot["dependency"],
        "human_queue_rows_sha256": snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": snapshot["auto_review_rows_sha256"],
        "reviewer_id": reviewer_id,
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_sha256(rubric_meta),
        "completed_review_rows_sha256": completed_review_rows_sha256(rows),
        "saw_condition_outputs": False,
        "saw_model_scores": False,
        "attestation_confirmed": False,
    }


def _validate_completed_reviews(
    proposal_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    reviewer_id: str,
    policy: Mapping[str, Any],
) -> None:
    _validate_foil_policy(policy)
    proposals = {str(row["candidate_id"]): dict(row) for row in proposal_rows}
    reviews: dict[str, dict[str, Any]] = {}
    for raw in review_rows:
        row = dict(raw)
        if set(row) != REVIEW_ROW_KEYS:
            raise CounterfactualLifecycleError("completed review row has non-canonical fields")
        try:
            validate_json_schema(row, SCHEMA_ROOT / "stage1_cf_review_v1.schema.json")
        except TrainingArtifactError as exc:
            raise _translate_artifact_error(exc) from exc
        identifier = row["candidate_id"]
        if identifier in reviews:
            raise CounterfactualLifecycleError("completed review contains duplicate candidate ID")
        if row["reviewer_id"] != reviewer_id:
            raise CounterfactualLifecycleError("review row reviewer_id differs from declaration")
        reviews[identifier] = row
    if set(reviews) != set(proposals):
        raise CounterfactualLifecycleError("review candidate set must exactly equal proposal")
    # Pure finalization executes all decision/reason and proposal-ID checks.
    finalize_rows(proposal_rows, review_rows, cf_build_id="cf-validation", review_id="review-validation")


def _review_id_inputs(
    *,
    proposal_dependency: Mapping[str, Any],
    blind_review_dependency: Mapping[str, Any],
    rows_sha256: str,
    human_queue_rows_sha256: str,
    auto_review_rows_sha256: str,
    human_completed_rows_sha256: str,
    declaration_sha256: str,
    publisher_code_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CF_REVIEW_ID_INPUT_SCHEMA,
        "proposal_dependency": copy.deepcopy(dict(proposal_dependency)),
        "cf_blind_review_dependency": copy.deepcopy(
            dict(blind_review_dependency)
        ),
        "review_rows_sha256": rows_sha256,
        "human_queue_rows_sha256": human_queue_rows_sha256,
        "auto_review_rows_sha256": auto_review_rows_sha256,
        "human_completed_rows_sha256": human_completed_rows_sha256,
        "declaration_sha256": declaration_sha256,
        "publisher_code_sha256": publisher_code_sha256,
    }


def _review_id(id_inputs: Mapping[str, Any]) -> str:
    return "review:v1:" + canonical_sha256(id_inputs)


def _publish_review_artifact(
    *,
    proposal_locator: Mapping[str, Any],
    proposal_target: Path,
    review_rows: Sequence[Mapping[str, Any]],
    declaration: Mapping[str, Any],
    blind_snapshot: Mapping[str, Any],
    artifact_root: Path,
    workspace_root: Path,
    write_review_ref: str | Path,
) -> tuple[dict[str, Any], Path]:
    split = load_json(proposal_target / "config.resolved.json")["split"]
    config = load_json(proposal_target / "config.resolved.json")
    policy = _validate_foil_policy(config["foil_policy"])
    proposals = load_jsonl(proposal_target / f"candidates.{split}.jsonl")
    proposal_dependency = portable_dependency(
        proposal_locator, proposal_target, workspace_root
    )
    if blind_snapshot.get("proposal_dependency") != proposal_dependency:
        raise CounterfactualLifecycleError(
            "CF blind review does not consume the selected proposal dependency"
        )
    if set(declaration) != DECLARATION_KEYS:
        raise CounterfactualLifecycleError("reviewer declaration has non-canonical fields")
    try:
        validate_json_schema(
            declaration, SCHEMA_ROOT / "stage1_cf_reviewer_declaration_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    reviewer_id = declaration.get("reviewer_id")
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise CounterfactualLifecycleError("reviewer declaration lacks reviewer_id")
    lineage_counts = _validate_blind_review_union(
        proposal_rows=proposals,
        review_rows=review_rows,
        reviewer_id=reviewer_id,
        policy=policy,
        blind_snapshot=blind_snapshot,
    )
    rubric_meta = load_json(proposal_target / "review_rubric.meta.json")
    rows_hash = completed_review_rows_sha256(review_rows)
    expected = {
        "schema_version": REVIEWER_DECLARATION_SCHEMA,
        "cf_proposal_id": proposal_locator["artifact_id"],
        "cf_blind_review_dependency": blind_snapshot["dependency"],
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "reviewer_id": reviewer_id,
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_sha256(rubric_meta),
        "completed_review_rows_sha256": rows_hash,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
        "attestation_confirmed": True,
    }
    if dict(declaration) != expected:
        mismatches = [key for key in expected if declaration.get(key) != expected[key]]
        raise CounterfactualLifecycleError(
            f"reviewer declaration is unsigned, non-blind, or hash-mismatched: {mismatches}"
        )
    declaration_hash = canonical_sha256(declaration)
    publisher_code_sha = _code_sha256()
    id_inputs = _review_id_inputs(
        proposal_dependency=proposal_dependency,
        blind_review_dependency=blind_snapshot["dependency"],
        rows_sha256=rows_hash,
        human_queue_rows_sha256=blind_snapshot["human_queue_rows_sha256"],
        auto_review_rows_sha256=blind_snapshot["auto_review_rows_sha256"],
        human_completed_rows_sha256=lineage_counts["human_completed_rows_sha256"],
        declaration_sha256=declaration_hash,
        publisher_code_sha256=publisher_code_sha,
    )
    review_id = _review_id(id_inputs)
    meta = {
        "schema_version": REVIEW_META_SCHEMA,
        "review_id": review_id,
        "cf_proposal_id": proposal_locator["artifact_id"],
        "cf_blind_review_id": blind_snapshot["dependency"]["artifact_id"],
        "scientific_eligible": config["scientific_eligible"],
        "reviewer_id": reviewer_id,
        "candidate_count": len(review_rows),
        "auto_review_count": lineage_counts["auto_review_count"],
        "human_review_count": lineage_counts["human_review_count"],
        "review_rows_sha256": rows_hash,
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "human_completed_rows_sha256": lineage_counts[
            "human_completed_rows_sha256"
        ],
        "reviewer_declaration_sha256": declaration_hash,
    }
    provenance = {
        "schema_version": CF_REVIEW_PROVENANCE_SCHEMA,
        "review_id": review_id,
        "id_inputs": id_inputs,
        "proposal_dependency": proposal_dependency,
        "cf_blind_review_dependency": blind_snapshot["dependency"],
        "review_rows_sha256": rows_hash,
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "human_completed_rows_sha256": lineage_counts[
            "human_completed_rows_sha256"
        ],
        "reviewer_declaration_sha256": declaration_hash,
        "publisher_code_sha256": publisher_code_sha,
    }
    frozen_policy = None
    if split == "test":
        frozen_policy = load_json(proposal_target / "frozen_policy_ref.json")
        meta["frozen_policy_ref"] = frozen_policy
    parent = artifact_root / "reviews"
    target = parent / review_id
    staging = new_staging_directory(parent, review_id)
    try:
        write_canonical_json(staging / "proposal_ref.json", proposal_dependency)
        write_canonical_json(
            staging / "cf_blind_review_ref.json", blind_snapshot["dependency"]
        )
        write_canonical_jsonl(staging / "foil_review.jsonl", review_rows, key="candidate_id")
        write_canonical_json(staging / "reviewer_declaration.json", declaration)
        write_canonical_json(staging / "review.meta.json", meta)
        write_canonical_json(staging / "review.provenance.json", provenance)
        if frozen_policy is not None:
            write_canonical_json(staging / "frozen_policy_ref.json", frozen_policy)
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_review_target(
                candidate,
                workspace_root=workspace_root,
                proposal_target=proposal_target,
                blind_review_target=blind_snapshot["target"],
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise _translate_artifact_error(exc) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    if (
        validate_payload_manifest(Path(blind_snapshot["target"]))
        != blind_snapshot["dependency"]["payload_manifest_sha256"]
    ):
        raise CounterfactualLifecycleError(
            "CF blind-review target changed while review artifact was published"
        )
    try:
        locator = write_locator_ref(
            write_review_ref,
            artifact_kind=REVIEW_ARTIFACT_KIND,
            artifact_id=review_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    return locator, target


def validate_review_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    proposal_target: str | Path | None = None,
    blind_review_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        payload_hash = validate_payload_manifest(target)
        expected_files = {
                "proposal_ref.json",
                "cf_blind_review_ref.json",
                "foil_review.jsonl",
                "reviewer_declaration.json",
                "review.meta.json",
                "review.provenance.json",
                "payload_manifest.json",
            }
        if (target / "frozen_policy_ref.json").is_file():
            expected_files.add("frozen_policy_ref.json")
        ensure_exact_file_set(target, expected_files)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    root = Path(workspace_root).resolve()
    dependency = load_json(target / "proposal_ref.json")
    blind_dependency = load_json(target / "cf_blind_review_ref.json")
    try:
        validate_dependency_ref(dependency, expected_kind=PROPOSAL_ARTIFACT_KIND)
        validate_dependency_ref(
            blind_dependency, expected_kind=CF_BLIND_REVIEW_ARTIFACT_KIND
        )
        proposal = Path(proposal_target) if proposal_target is not None else resolve_dependency_target(
            dependency, root
        )
        blind_target = (
            Path(blind_review_target)
            if blind_review_target is not None
            else resolve_dependency_target(blind_dependency, root)
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    if proposal.name != dependency["artifact_id"] or validate_payload_manifest(proposal) != dependency[
        "payload_manifest_sha256"
    ]:
        raise CounterfactualLifecycleError("review proposal dependency mismatch")
    validate_proposal_target(
        proposal, workspace_root=root, require_directory_name=True
    )
    blind_snapshot = _inspect_cf_blind_review_target(
        blind_review_target=blind_target,
        blind_review_dependency=blind_dependency,
        proposal_target=proposal,
        proposal_dependency=dependency,
        workspace_root=root,
    )
    config = load_json(proposal / "config.resolved.json")
    split = config["split"]
    proposals = load_jsonl(proposal / f"candidates.{split}.jsonl")
    rows = load_jsonl(target / "foil_review.jsonl")
    if (target / "foil_review.jsonl").read_bytes() != canonical_jsonl_bytes(
        rows, key="candidate_id"
    ):
        raise CounterfactualLifecycleError("review rows are not canonical JSONL")
    declaration = load_json(target / "reviewer_declaration.json")
    if set(declaration) != DECLARATION_KEYS:
        raise CounterfactualLifecycleError("reviewer declaration has non-canonical fields")
    try:
        validate_json_schema(
            declaration, SCHEMA_ROOT / "stage1_cf_reviewer_declaration_v1.schema.json"
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    reviewer_id = declaration.get("reviewer_id")
    lineage_counts = _validate_blind_review_union(
        proposal_rows=proposals,
        review_rows=rows,
        reviewer_id=reviewer_id,
        policy=config["foil_policy"],
        blind_snapshot=blind_snapshot,
    )
    rubric_meta = load_json(proposal / "review_rubric.meta.json")
    rows_hash = completed_review_rows_sha256(rows)
    expected_declaration = {
        "schema_version": REVIEWER_DECLARATION_SCHEMA,
        "cf_proposal_id": dependency["artifact_id"],
        "cf_blind_review_dependency": blind_dependency,
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "reviewer_id": reviewer_id,
        "rubric_body_sha256": rubric_meta["rubric_body_sha256"],
        "rubric_meta_sha256": canonical_sha256(rubric_meta),
        "completed_review_rows_sha256": rows_hash,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
        "attestation_confirmed": True,
    }
    if declaration != expected_declaration:
        raise CounterfactualLifecycleError("reviewer declaration blind/hash contract failed")
    declaration_hash = canonical_sha256(declaration)
    publisher_code_sha = _code_sha256()
    id_inputs = _review_id_inputs(
        proposal_dependency=dependency,
        blind_review_dependency=blind_dependency,
        rows_sha256=rows_hash,
        human_queue_rows_sha256=blind_snapshot["human_queue_rows_sha256"],
        auto_review_rows_sha256=blind_snapshot["auto_review_rows_sha256"],
        human_completed_rows_sha256=lineage_counts["human_completed_rows_sha256"],
        declaration_sha256=declaration_hash,
        publisher_code_sha256=publisher_code_sha,
    )
    review_id = _review_id(id_inputs)
    if require_directory_name and target.name != review_id:
        raise CounterfactualLifecycleError("review target directory name differs from review ID")
    meta = load_json(target / "review.meta.json")
    expected_meta = {
        "schema_version": REVIEW_META_SCHEMA,
        "review_id": review_id,
        "cf_proposal_id": dependency["artifact_id"],
        "cf_blind_review_id": blind_dependency["artifact_id"],
        "scientific_eligible": config["scientific_eligible"],
        "reviewer_id": reviewer_id,
        "candidate_count": len(rows),
        "auto_review_count": lineage_counts["auto_review_count"],
        "human_review_count": lineage_counts["human_review_count"],
        "review_rows_sha256": rows_hash,
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "human_completed_rows_sha256": lineage_counts[
            "human_completed_rows_sha256"
        ],
        "reviewer_declaration_sha256": declaration_hash,
    }
    if split == "test":
        frozen_policy = load_json(proposal / "frozen_policy_ref.json")
        if load_json(target / "frozen_policy_ref.json") != frozen_policy:
            raise CounterfactualLifecycleError(
                "sealed review frozen policy differs from proposal"
            )
        expected_meta["frozen_policy_ref"] = frozen_policy
    elif (target / "frozen_policy_ref.json").exists():
        raise CounterfactualLifecycleError("dev review cannot claim sealed lineage")
    if meta != expected_meta:
        raise CounterfactualLifecycleError("review meta cannot be reproduced")
    provenance = load_json(target / "review.provenance.json")
    expected_provenance = {
        "schema_version": CF_REVIEW_PROVENANCE_SCHEMA,
        "review_id": review_id,
        "id_inputs": id_inputs,
        "proposal_dependency": dependency,
        "cf_blind_review_dependency": blind_dependency,
        "review_rows_sha256": rows_hash,
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "auto_review_rows_sha256": blind_snapshot["auto_review_rows_sha256"],
        "human_completed_rows_sha256": lineage_counts[
            "human_completed_rows_sha256"
        ],
        "reviewer_declaration_sha256": declaration_hash,
        "publisher_code_sha256": publisher_code_sha,
    }
    if provenance != expected_provenance:
        raise CounterfactualLifecycleError("review provenance cannot be reproduced")
    return {
        "schema_version": "stage1-cf-review-validation/v1",
        "valid": True,
        "review_id": review_id,
        "cf_proposal_id": dependency["artifact_id"],
        "cf_blind_review_id": blind_dependency["artifact_id"],
        "candidate_count": len(rows),
        "auto_review_count": lineage_counts["auto_review_count"],
        "human_review_count": lineage_counts["human_review_count"],
        "human_queue_rows_sha256": blind_snapshot["human_queue_rows_sha256"],
        "scientific_eligible": config["scientific_eligible"],
        "payload_manifest_sha256": payload_hash,
    }


def _final_selection_policy_hash() -> str:
    return canonical_sha256(
        {
            "selection_policy": SELECTION_POLICY,
            "family_order": list(FAMILY_ORDER),
            "reference_model_sha256": None,
        }
    )


def _final_meta(
    *,
    cf_build_id: str,
    id_inputs: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    proposal_id: str,
    review_id: str,
    split: str,
    scientific_eligible: bool,
) -> dict[str, Any]:
    query_field_rows: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        query_field_rows[(str(row["query_id"]), str(row["field"]))].append(row)
    query_ids = sorted({str(row["query_id"]) for row in rows}, key=int)
    # A query is eligible only when every one of its gold tuples has a selected
    # foil for that field.  Treating "any constructed tuple" as complete would
    # silently overweight multi-tuple queries and break equal-within-query
    # aggregation in the teacher-forced margin analysis.
    complete_case = {
        field: [
            query_id
            for query_id in query_ids
            if query_field_rows[(query_id, field)]
            and all(
                row["construction_status"] == "ok"
                for row in query_field_rows[(query_id, field)]
            )
        ]
        for field in FIELDS
    }
    return {
        "schema_version": FINAL_META_SCHEMA,
        "cf_build_id": cf_build_id,
        "cf_proposal_id": proposal_id,
        "review_id": review_id,
        "split": split,
        "scientific_eligible": scientific_eligible,
        "id_inputs": copy.deepcopy(dict(id_inputs)),
        "record_count": len(rows),
        "records_sha256": hashlib.sha256(canonical_jsonl_bytes(rows, key="record_sha256")).hexdigest(),
        "field_construction": copy.deepcopy(summary["field_construction"]),
        "group_hate_coverage_gate": summary["group_hate_coverage_gate"],
        "complete_case_query_ids": complete_case,
    }


def finalize_counterfactual_artifact(
    *,
    proposal_ref: str | Path,
    blind_review_ref: str | Path,
    review_file: str | Path,
    reviewer_declaration: str | Path,
    write_review_ref: str | Path,
    write_ref: str | Path,
    target_root: str | Path | None = None,
    workspace_root: str | Path | None = None,
    sealed: bool = False,
) -> dict[str, Any]:
    """Bind a completed frozen blind review, then publish review/final CF refs."""

    # Keep dependency resolution independent from the output placement knob.
    workspace = (
        Path(workspace_root).resolve()
        if workspace_root is not None
        else REPOSITORY_ROOT
    )
    try:
        proposal_locator, proposal_target = resolve_locator_ref(
            proposal_ref, PROPOSAL_ARTIFACT_KIND
        )
        review_rows = load_jsonl(review_file)
        declaration = load_json(reviewer_declaration)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    validate_proposal_target(proposal_target, workspace_root=workspace)
    if not isinstance(declaration, Mapping):
        raise CounterfactualLifecycleError("reviewer declaration must be an object")
    source_config = load_json(proposal_target / "config.resolved.json")
    split = source_config["split"]
    if sealed is not (split == "test"):
        raise CounterfactualLifecycleError(
            "--sealed assertion differs from proposal split/frozen lineage"
        )
    frozen_policy = (
        load_json(proposal_target / "frozen_policy_ref.json")
        if split == "test"
        else None
    )
    proposal_dependency = portable_dependency(
        proposal_locator, proposal_target, workspace
    )
    blind_snapshot = _resolve_cf_blind_review_snapshot(
        blind_review_ref=blind_review_ref,
        workspace_root=workspace,
        proposal_target=proposal_target,
        proposal_dependency=proposal_dependency,
    )
    artifact_root = Path(target_root).resolve() if target_root is not None else proposal_target.parent.parent
    review_locator, review_target = _publish_review_artifact(
        proposal_locator=proposal_locator,
        proposal_target=proposal_target,
        review_rows=review_rows,
        declaration=declaration,
        blind_snapshot=blind_snapshot,
        artifact_root=artifact_root,
        workspace_root=workspace,
        write_review_ref=write_review_ref,
    )
    proposals = load_jsonl(proposal_target / f"candidates.{split}.jsonl")
    review_dependency = portable_dependency(review_locator, review_target, workspace)
    finalizer_code_sha = _code_sha256()
    id_inputs = {
        "cf_proposal_id": proposal_locator["artifact_id"],
        "review_id": review_locator["artifact_id"],
        "selection_policy_sha256": _final_selection_policy_hash(),
        "reference_model_sha256": None,
        "finalizer_code_sha256": finalizer_code_sha,
    }
    if frozen_policy is not None:
        id_inputs["frozen_policy_ref"] = frozen_policy
    cf_build_id = "cf-" + canonical_sha256(id_inputs)
    rows, summary = finalize_rows(
        proposals,
        review_rows,
        cf_build_id=cf_build_id,
        review_id=review_locator["artifact_id"],
    )
    if summary["group_hate_coverage_gate"] is not True:
        raise CounterfactualLifecycleError(
            "group/hate construction coverage gate must be exactly 100%"
        )
    meta = _final_meta(
        cf_build_id=cf_build_id,
        id_inputs=id_inputs,
        rows=rows,
        summary=summary,
        proposal_id=proposal_locator["artifact_id"],
        review_id=review_locator["artifact_id"],
        split=split,
        scientific_eligible=source_config["scientific_eligible"],
    )
    context_dependency = load_json(proposal_target / "context_ref.json")
    provenance = {
        "schema_version": "stage1-cf-provenance/v1",
        "cf_build_id": cf_build_id,
        "scientific_eligible": source_config["scientific_eligible"],
        "context_dependency": context_dependency,
        "proposal_dependency": proposal_dependency,
        "review_dependency": review_dependency,
        "selection_policy_sha256": _final_selection_policy_hash(),
        "finalizer_code_sha256": finalizer_code_sha,
        "reference_model_sha256": None,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
    }
    if frozen_policy is not None:
        provenance["frozen_policy_ref"] = frozen_policy
    dirname = "test_counterfactuals" if split == "test" else "counterfactuals"
    parent = artifact_root / dirname
    target = parent / cf_build_id
    staging = new_staging_directory(parent, cf_build_id)
    try:
        write_canonical_json(staging / "config.resolved.json", source_config)
        write_canonical_json(staging / "context_ref.json", context_dependency)
        write_canonical_json(staging / "proposal_ref.json", proposal_dependency)
        write_canonical_json(staging / "review_ref.json", review_dependency)
        write_canonical_json(staging / "provenance.json", provenance)
        write_canonical_json(staging / "cf_manifest.meta.json", meta)
        if frozen_policy is not None:
            write_canonical_json(staging / "frozen_policy_ref.json", frozen_policy)
        write_canonical_jsonl(staging / f"cf_manifest.{split}.jsonl", rows, key="record_sha256")
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda candidate: validate_cf_target(
                candidate,
                workspace_root=workspace,
                proposal_target=proposal_target,
                review_target=review_target,
                require_directory_name=False,
            ),
        )
    except (TrainingArtifactError, OSError) as exc:
        if staging.exists():
            shutil.rmtree(staging)
        raise _translate_artifact_error(exc) from exc
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    try:
        return write_locator_ref(
            write_ref,
            artifact_kind=FINAL_ARTIFACT_KIND,
            artifact_id=cf_build_id,
            target=target,
            payload_manifest_sha256=payload_hash,
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc


def validate_cf_target(
    target_dir: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    proposal_target: str | Path | None = None,
    review_target: str | Path | None = None,
    require_directory_name: bool = True,
) -> dict[str, Any]:
    target = Path(target_dir)
    try:
        validate_payload_manifest(target)
        split_for_files = load_json(target / "config.resolved.json")["split"]
        expected_files = {
            "config.resolved.json",
            "context_ref.json",
            "proposal_ref.json",
            "review_ref.json",
            "provenance.json",
            "cf_manifest.meta.json",
            f"cf_manifest.{split_for_files}.jsonl",
            "payload_manifest.json",
        }
        if split_for_files == "test":
            expected_files.add("frozen_policy_ref.json")
        ensure_exact_file_set(
            target,
            expected_files,
        )
    except (TrainingArtifactError, KeyError) as exc:
        raise _translate_artifact_error(exc) from exc
    config = load_json(target / "config.resolved.json")
    if not isinstance(config, Mapping) or config.get("schema_version") != RESOLVED_CONFIG_SCHEMA:
        raise CounterfactualLifecycleError("final CF resolved config has wrong schema")
    split = config.get("split")
    if split not in {"dev", "test"}:
        raise CounterfactualLifecycleError("final CF split is invalid")
    proposal_dependency = load_json(target / "proposal_ref.json")
    review_dependency = load_json(target / "review_ref.json")
    try:
        validate_dependency_ref(proposal_dependency, expected_kind=PROPOSAL_ARTIFACT_KIND)
        validate_dependency_ref(review_dependency, expected_kind=REVIEW_ARTIFACT_KIND)
        proposal = Path(proposal_target) if proposal_target is not None else resolve_dependency_target(
            proposal_dependency, workspace_root
        )
        review = Path(review_target) if review_target is not None else resolve_dependency_target(
            review_dependency, workspace_root
        )
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    if proposal.name != proposal_dependency["artifact_id"] or validate_payload_manifest(
        proposal
    ) != proposal_dependency["payload_manifest_sha256"]:
        raise CounterfactualLifecycleError("final proposal dependency mismatch")
    if review.name != review_dependency["artifact_id"] or validate_payload_manifest(
        review
    ) != review_dependency["payload_manifest_sha256"]:
        raise CounterfactualLifecycleError("final review dependency mismatch")
    validate_proposal_target(proposal, workspace_root=workspace_root)
    review_report = validate_review_target(
        review, workspace_root=workspace_root, proposal_target=proposal
    )
    if load_json(target / "context_ref.json") != load_json(proposal / "context_ref.json"):
        raise CounterfactualLifecycleError("final/proposal context dependency mismatch")
    if config != load_json(proposal / "config.resolved.json"):
        raise CounterfactualLifecycleError("final/proposal resolved config mismatch")
    proposals = load_jsonl(proposal / f"candidates.{split}.jsonl")
    reviews = load_jsonl(review / "foil_review.jsonl")
    finalizer_code_sha = _code_sha256()
    id_inputs = {
        "cf_proposal_id": proposal_dependency["artifact_id"],
        "review_id": review_dependency["artifact_id"],
        "selection_policy_sha256": _final_selection_policy_hash(),
        "reference_model_sha256": None,
        "finalizer_code_sha256": finalizer_code_sha,
    }
    frozen_policy = None
    if split == "test":
        frozen_policy = load_json(proposal / "frozen_policy_ref.json")
        if (
            load_json(review / "frozen_policy_ref.json") != frozen_policy
            or load_json(target / "frozen_policy_ref.json") != frozen_policy
        ):
            raise CounterfactualLifecycleError(
                "sealed final CF policy differs across proposal/review/final"
            )
        id_inputs["frozen_policy_ref"] = frozen_policy
    cf_build_id = "cf-" + canonical_sha256(id_inputs)
    if require_directory_name and target.name != cf_build_id:
        raise CounterfactualLifecycleError("final target directory name differs from CF build ID")
    expected_rows, summary = finalize_rows(
        proposals,
        reviews,
        cf_build_id=cf_build_id,
        review_id=review_dependency["artifact_id"],
    )
    if summary["group_hate_coverage_gate"] is not True:
        raise CounterfactualLifecycleError("group/hate construction coverage is below 100%")
    rows = load_jsonl(target / f"cf_manifest.{split}.jsonl")
    if rows != sorted(expected_rows, key=lambda row: row["record_sha256"]):
        # ``write_canonical_jsonl`` sorts on record_sha256; compare that wire order.
        raise CounterfactualLifecycleError("final CF rows cannot be deterministically replayed")
    if (target / f"cf_manifest.{split}.jsonl").read_bytes() != canonical_jsonl_bytes(
        expected_rows, key="record_sha256"
    ):
        raise CounterfactualLifecycleError("final CF JSONL is not canonical")
    for row in rows:
        try:
            validate_json_schema(row, SCHEMA_ROOT / "stage1_cf_manifest_v1.schema.json")
        except TrainingArtifactError as exc:
            raise _translate_artifact_error(exc) from exc
        if row["record_sha256"] != canonical_sha256(
            {key: value for key, value in row.items() if key != "record_sha256"}
        ):
            raise CounterfactualLifecycleError("final CF record hash mismatch")
    expected_meta = _final_meta(
        cf_build_id=cf_build_id,
        id_inputs=id_inputs,
        rows=expected_rows,
        summary=summary,
        proposal_id=proposal_dependency["artifact_id"],
        review_id=review_dependency["artifact_id"],
        split=split,
        scientific_eligible=config["scientific_eligible"],
    )
    if load_json(target / "cf_manifest.meta.json") != expected_meta:
        raise CounterfactualLifecycleError("final CF meta cannot be reproduced")
    expected_provenance = {
        "schema_version": "stage1-cf-provenance/v1",
        "cf_build_id": cf_build_id,
        "scientific_eligible": config["scientific_eligible"],
        "context_dependency": load_json(proposal / "context_ref.json"),
        "proposal_dependency": proposal_dependency,
        "review_dependency": review_dependency,
        "selection_policy_sha256": _final_selection_policy_hash(),
        "finalizer_code_sha256": finalizer_code_sha,
        "reference_model_sha256": None,
        "saw_condition_outputs": False,
        "saw_model_scores": False,
    }
    if frozen_policy is not None:
        expected_provenance["frozen_policy_ref"] = frozen_policy
    if load_json(target / "provenance.json") != expected_provenance:
        raise CounterfactualLifecycleError("final CF provenance cannot be reproduced")
    return {
        "schema_version": "stage1-cf-validation-report/v1",
        "valid": True,
        "cf_build_id": cf_build_id,
        "cf_proposal_id": proposal_dependency["artifact_id"],
        "review_id": review_dependency["artifact_id"],
        "cf_blind_review_id": review_report["cf_blind_review_id"],
        "split": split,
        "scientific_eligible": config["scientific_eligible"],
        "record_count": len(rows),
        "field_construction": summary["field_construction"],
        "group_hate_coverage_gate": True,
        "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
    }


def validate_cf_ref(
    cf_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    try:
        locator, target = resolve_locator_ref(cf_ref, FINAL_ARTIFACT_KIND)
    except TrainingArtifactError as exc:
        raise _translate_artifact_error(exc) from exc
    report = validate_cf_target(target, workspace_root=workspace_root)
    if report["cf_build_id"] != locator["artifact_id"]:
        raise CounterfactualLifecycleError("CF locator ID mismatch")
    if report["payload_manifest_sha256"] != locator["payload_manifest_sha256"]:
        raise CounterfactualLifecycleError("CF locator payload mismatch")
    return report


__all__ = [
    "CounterfactualLifecycleError",
    "build_rubric_meta",
    "canonical_rubric_bytes",
    "completed_review_rows_sha256",
    "finalize_counterfactual_artifact",
    "prepare_reviewer_declaration",
    "propose_counterfactual_artifact",
    "validate_cf_ref",
    "validate_cf_target",
    "validate_proposal_ref",
    "validate_proposal_target",
    "validate_review_target",
]
