"""Build a prepared Stage-1 context bundle from train-only cosine scores."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from data import context_selector as context_selector_module
from data.context_manifest import text_sha256
from data.context_selector import round_similarity_half_even, select_demos, select_lexicons
from prompt import STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1
from rag.types import (
    RetrievalHit,
    content_sha256,
    stable_demo_id,
    stable_term_evidence_id,
)
from utils.quadruple import canonicalize_quadruples, serialize_quadruples


BUNDLE_SCHEMA = "stage1-prepared-context-bundle/v1"
RETRIEVAL_BUNDLE_VERSION = "stage1-fit-only-cosine-bundle/v1"
LEGACY_RETRIEVAL_BUNDLE_VERSION = "stage1-train-only-cosine-bundle/v1"
LEXICON_EVIDENCE_RENDER_POLICY = "category-free-terminology-evidence/v1"
LEXICON_TASK_LABEL_VISIBILITY = "absent"
LEXICON_EVIDENCE_KIND = "terminology"
TERMINOLOGY_TASK_FIELD_KEYS = frozenset(
    {
        "annotation_count",
        "categories",
        "category",
        "category_counts",
        "category_purity",
        "hate_count",
        "hate_precision",
        "hateful",
        "label",
        "labels",
        "log_odds",
        "non_hate_count",
        "nonhate_penalty",
        "primary_category",
        "targeted_group",
    }
)
DEPENDENCY_SCHEMA = "stage1-dependency-ref/v1"
SCORE_EVIDENCE_SCHEMA = "stage1-retrieval-score-evidence/v1"
SCORE_EVIDENCE_ENCODING = "base64-little-endian-float64-c-order/v1"
# This is deliberately above the production Stage-1 train matrix while still
# bounding hostile JSON before base64 decoding allocates the binary payload.
MAX_SCORE_MATRIX_ELEMENTS = 100_000_000


class RetrievalBundleError(ValueError):
    pass


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def retrieval_code_sha256s() -> dict[str, str]:
    """Return path-independent hashes of the exact scorer-to-selector code."""

    selector_path = getattr(context_selector_module, "__file__", None)
    if not isinstance(selector_path, str):  # pragma: no cover - import invariant
        raise RetrievalBundleError("cannot resolve context-selector source")
    hashes: dict[str, str] = {}
    for key, path in (
        ("retrieval_builder_code_sha256", __file__),
        ("selector_code_sha256", selector_path),
    ):
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:  # pragma: no cover - installed source invariant
            raise RetrievalBundleError(f"cannot hash retrieval source {path}") from exc
        hashes[key] = digest.hexdigest()
    return hashes


def _ordered_text_frame_sha256(
    identifiers: Sequence[str], texts: Sequence[str]
) -> str:
    if len(identifiers) != len(texts):
        raise RetrievalBundleError("retrieval text frame length mismatch")
    return _canonical_sha(
        [
            [str(identifier), content_sha256(str(text))]
            for identifier, text in zip(identifiers, texts, strict=True)
        ]
    )


def _score_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f8").tobytes(order="C")


def _score_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def build_score_evidence(
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    demo_ids: Sequence[str],
    demo_texts: Sequence[str],
    lexicon_ids: Sequence[str],
    lexicon_texts: Sequence[str],
    demo_scores: np.ndarray,
    lexicon_scores: np.ndarray,
) -> dict[str, Any]:
    """Freeze complete portable score bytes and their ordered source frames."""

    query_frame = [str(value) for value in query_ids]
    demo_frame = [str(value) for value in demo_ids]
    lexicon_frame = [str(value) for value in lexicon_ids]
    demo_values = _validate_scores(
        demo_scores,
        query_count=len(query_frame),
        corpus_count=len(demo_frame),
        name="demo",
    )
    lexicon_values = _validate_scores(
        lexicon_scores,
        query_count=len(query_frame),
        corpus_count=len(lexicon_frame),
        name="lexicon",
    )
    demo_payload = _score_bytes(demo_values)
    lexicon_payload = _score_bytes(lexicon_values)
    return {
        "schema_version": SCORE_EVIDENCE_SCHEMA,
        "encoding": SCORE_EVIDENCE_ENCODING,
        "query_ids": query_frame,
        "demo_ids": demo_frame,
        "lexicon_ids": lexicon_frame,
        "query_texts_sha256": _ordered_text_frame_sha256(
            query_frame, query_texts
        ),
        "demo_texts_sha256": _ordered_text_frame_sha256(
            demo_frame, demo_texts
        ),
        "lexicon_texts_sha256": _ordered_text_frame_sha256(
            lexicon_frame, lexicon_texts
        ),
        "demo_shape": [len(query_frame), len(demo_frame)],
        "lexicon_shape": [len(query_frame), len(lexicon_frame)],
        "demo_scores_base64": base64.b64encode(demo_payload).decode("ascii"),
        "lexicon_scores_base64": base64.b64encode(lexicon_payload).decode(
            "ascii"
        ),
    }


def _decode_score_matrix(
    encoded: Any,
    *,
    shape: Any,
    expected_shape: tuple[int, int],
    label: str,
) -> np.ndarray:
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or tuple(shape) != expected_shape
    ):
        raise RetrievalBundleError(f"{label} score evidence shape is invalid")
    elements = expected_shape[0] * expected_shape[1]
    if elements < 0 or elements > MAX_SCORE_MATRIX_ELEMENTS:
        raise RetrievalBundleError(f"{label} score evidence exceeds size bound")
    expected_bytes = elements * np.dtype("<f8").itemsize
    expected_base64_length = ((expected_bytes + 2) // 3) * 4
    if (
        not isinstance(encoded, str)
        or len(encoded) != expected_base64_length
        or not encoded.isascii()
    ):
        raise RetrievalBundleError(
            f"{label} score evidence encoded length is invalid"
        )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RetrievalBundleError(
            f"{label} score evidence is not canonical base64"
        ) from exc
    if len(payload) != expected_bytes:
        raise RetrievalBundleError(f"{label} score evidence byte length is invalid")
    values = np.frombuffer(payload, dtype="<f8").reshape(expected_shape)
    if not np.isfinite(values).all():
        raise RetrievalBundleError(f"{label} score evidence contains non-finite values")
    # A unique base64 spelling is required; this also rejects odd padding forms.
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise RetrievalBundleError(f"{label} score evidence is not canonical base64")
    return values


def decode_score_evidence(
    evidence: Mapping[str, Any],
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    demo_ids: Sequence[str],
    demo_texts: Sequence[str],
    lexicon_ids: Sequence[str],
    lexicon_texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a frozen score frame and decode its two bounded matrices."""

    expected_fields = {
        "schema_version",
        "encoding",
        "query_ids",
        "demo_ids",
        "lexicon_ids",
        "query_texts_sha256",
        "demo_texts_sha256",
        "lexicon_texts_sha256",
        "demo_shape",
        "lexicon_shape",
        "demo_scores_base64",
        "lexicon_scores_base64",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_fields
        or evidence.get("schema_version") != SCORE_EVIDENCE_SCHEMA
        or evidence.get("encoding") != SCORE_EVIDENCE_ENCODING
    ):
        raise RetrievalBundleError("retrieval score evidence is not canonical")
    expected_query_ids = [str(value) for value in query_ids]
    expected_demo_ids = [str(value) for value in demo_ids]
    expected_lexicon_ids = [str(value) for value in lexicon_ids]
    if (
        evidence.get("query_ids") != expected_query_ids
        or evidence.get("demo_ids") != expected_demo_ids
        or evidence.get("lexicon_ids") != expected_lexicon_ids
    ):
        raise RetrievalBundleError("retrieval score evidence ID frame mismatch")
    expected_text_hashes = {
        "query_texts_sha256": _ordered_text_frame_sha256(
            expected_query_ids, query_texts
        ),
        "demo_texts_sha256": _ordered_text_frame_sha256(
            expected_demo_ids, demo_texts
        ),
        "lexicon_texts_sha256": _ordered_text_frame_sha256(
            expected_lexicon_ids, lexicon_texts
        ),
    }
    if any(evidence.get(key) != value for key, value in expected_text_hashes.items()):
        raise RetrievalBundleError("retrieval score evidence text frame mismatch")
    query_count = len(expected_query_ids)
    demo_values = _decode_score_matrix(
        evidence.get("demo_scores_base64"),
        shape=evidence.get("demo_shape"),
        expected_shape=(query_count, len(expected_demo_ids)),
        label="demo",
    )
    lexicon_values = _decode_score_matrix(
        evidence.get("lexicon_scores_base64"),
        shape=evidence.get("lexicon_shape"),
        expected_shape=(query_count, len(expected_lexicon_ids)),
        label="lexicon",
    )
    return demo_values, lexicon_values


def _portable_dependency(
    value: Mapping[str, Any], *, artifact_kind: str
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
        "logical_repo_path",
    }
    if (
        set(value) != expected
        or value.get("schema_version") != DEPENDENCY_SCHEMA
        or value.get("artifact_kind") != artifact_kind
        or not isinstance(value.get("artifact_id"), str)
        or not isinstance(value.get("payload_manifest_sha256"), str)
        or len(str(value.get("payload_manifest_sha256"))) != 64
        or not isinstance(value.get("logical_repo_path"), str)
        or not value.get("logical_repo_path")
        or str(value.get("logical_repo_path")).startswith("/")
        or ".." in str(value.get("logical_repo_path")).split("/")
    ):
        raise RetrievalBundleError(f"invalid {artifact_kind} dependency")
    return dict(value)


def _partition_dependency(value: Mapping[str, Any]) -> dict[str, Any]:
    dependency = _portable_dependency(value, artifact_kind="train-partition")
    if not str(dependency["artifact_id"]).startswith("tpart-"):
        raise RetrievalBundleError("invalid train-partition dependency")
    return dependency


def _assert_locator_identity(
    locator: Mapping[str, Any], dependency: Mapping[str, Any], *, label: str
) -> None:
    compact = {
        "schema_version": DEPENDENCY_SCHEMA,
        "artifact_kind": locator.get("artifact_kind"),
        "artifact_id": locator.get("artifact_id"),
        "payload_manifest_sha256": locator.get("payload_manifest_sha256"),
    }
    expected = {
        key: dependency[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
        )
    }
    if compact != expected:
        raise RetrievalBundleError(f"{label} locator disagrees with dependency")


def _partitioned_demo_records(
    *,
    train_records: Sequence[Mapping[str, Any]],
    fit_demo_records: Sequence[Mapping[str, Any]] | None,
    calibration_ids: Sequence[str] | None,
    train_partition_dependency: Mapping[str, Any] | None,
) -> tuple[Sequence[Mapping[str, Any]], dict[str, Any] | None, tuple[str, ...]]:
    """Validate the pre-embedding fit filter without ever deriving it from scores."""

    if train_partition_dependency is None:
        if fit_demo_records is not None or calibration_ids is not None:
            raise RetrievalBundleError(
                "fit demo records/calibration IDs require a train-partition dependency"
            )
        return train_records, None, ()
    dependency = _partition_dependency(train_partition_dependency)
    if fit_demo_records is None or calibration_ids is None:
        raise RetrievalBundleError(
            "partitioned retrieval requires explicit fit records and calibration IDs"
        )
    full_by_id: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(train_records):
        identifier, _, _ = _record(record)
        if identifier in full_by_id:
            raise RetrievalBundleError(f"duplicate full train ID {identifier!r}")
        full_by_id[identifier] = record
    fit_ids: list[str] = []
    for record in fit_demo_records:
        identifier, _, _ = _record(record)
        if identifier in fit_ids:
            raise RetrievalBundleError(f"duplicate fit demo ID {identifier!r}")
        if identifier not in full_by_id or dict(record) != dict(full_by_id[identifier]):
            raise RetrievalBundleError("fit demo record differs from the full train frame")
        fit_ids.append(identifier)
    calibration = tuple(str(value) for value in calibration_ids)
    if len(calibration) != len(set(calibration)):
        raise RetrievalBundleError("calibration IDs are not unique")
    if set(fit_ids).intersection(calibration):
        raise RetrievalBundleError("fit and calibration IDs overlap")
    if set(fit_ids).union(calibration) != set(full_by_id):
        raise RetrievalBundleError("fit/calibration IDs do not exactly cover full train")
    if not fit_ids or not calibration:
        raise RetrievalBundleError("fit and calibration partitions must both be non-empty")
    return fit_demo_records, dependency, calibration


def _record(record: Mapping[str, Any]) -> tuple[str, str, list[Any]]:
    query_id = str(record.get("id", ""))
    content = record.get("content")
    quads = record.get("quadruples", record.get("gold"))
    if not query_id or not isinstance(content, str) or not content:
        raise RetrievalBundleError("record requires id and non-empty content")
    try:
        canonical = canonicalize_quadruples(quads)
    except Exception as exc:
        raise RetrievalBundleError(f"record {query_id} has invalid gold: {exc}") from exc
    return query_id, content, canonical


def build_query_pool(
    records: Sequence[Mapping[str, Any]],
    *,
    source_split: str,
) -> list[dict[str, Any]]:
    """Freeze the complete canonical query/gold frame for a source split.

    This catalog is deliberately distinct from the rendered demo catalog.  In
    particular, downstream counterfactual construction needs every train gold
    tuple, not merely the fields retained by a prompt adapter.
    """

    if source_split not in {"train", "dev", "test"}:
        raise RetrievalBundleError("query-pool source split must be train/dev/test")
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        query_id, content, quads = _record(raw)
        if query_id in seen:
            raise RetrievalBundleError(f"duplicate {source_split} query-pool ID {query_id!r}")
        seen.add(query_id)
        output = serialize_quadruples(quads)
        pool.append(
            {
                "id": query_id,
                "content": content,
                "quadruples": [
                    {
                        "target": quad.target,
                        "argument": quad.argument,
                        "targeted_group": list(quad.targeted_group),
                        "hateful": quad.hateful,
                    }
                    for quad in quads
                ],
                "content_sha256": content_sha256(content),
                "gold_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
                "source_split": source_split,
            }
        )
    return pool


def _source_classes(quads: Sequence[Any], order: Sequence[str]) -> list[str]:
    present = {label for quad in quads for label in quad.targeted_group}
    return [label for label in order if label in present]


def build_demo_catalog(
    train_records: Sequence[Mapping[str, Any]],
    *,
    source_class_order: Sequence[str],
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    catalog: list[dict[str, Any]] = []
    memberships: list[list[str]] = []
    for raw in train_records:
        source_id, content, quads = _record(raw)
        output = serialize_quadruples(quads)
        content_hash = content_sha256(content)
        gold_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
        identifier = stable_demo_id(source_id, content_hash, gold_hash)
        classes = _source_classes(quads, source_class_order)
        if not classes:
            raise RetrievalBundleError(f"train record {source_id} has no retrieval source class")
        rendered = STAGE1_QUAD_JSON_EXAMPLE_PROMPT_V1.format(
            retrieve_content=content,
            retrieve_output=output,
        )
        catalog.append(
            {
                "demo_id": identifier,
                "source_record_id": source_id,
                "content": content,
                "output": output,
                "content_sha256": content_hash,
                "gold_sha256": gold_hash,
                "output_label": classes[0],
                "source_classes": classes,
                "rendered_block": rendered,
                "rendered_block_sha256": text_sha256(rendered),
                "source_split": "train",
                "train_only": True,
            }
        )
        memberships.append(classes)
    if len({row["demo_id"] for row in catalog}) != len(catalog):
        raise RetrievalBundleError("stable demo IDs are not unique")
    return catalog, memberships


def render_lexicon_evidence_block(
    *,
    term: str,
    definition: str,
    variants: Sequence[str] = (),
    usage_notes: str = "",
    ambiguity_notes: str = "",
) -> str:
    """Render one category-free terminology-understanding evidence block.

    The published main-experiment resource has no task-category field.  This
    exact block is also used for semantic retrieval, keeping retrieval and
    generation on the same category-free information surface.
    """

    word = str(term).strip()
    meaning = str(definition).strip()
    usage = str(usage_notes).strip()
    ambiguity = str(ambiguity_notes).strip()
    normalized_variants = [
        str(value).strip() for value in variants if str(value).strip()
    ]
    if not word or not meaning:
        raise RetrievalBundleError(
            "lexical evidence requires a non-empty term and definition"
        )
    rendered = f"术语：{word}\n词义说明：{meaning}"
    if usage:
        rendered += f"\n用法提示：{usage}"
    if ambiguity:
        rendered += f"\n歧义提示：{ambiguity}"
    if normalized_variants:
        rendered += "\n词形变体：" + "、".join(normalized_variants)
    return rendered


def _terminology_forbidden_key_paths(
    value: Any, *, path: str = "entry"
) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in TERMINOLOGY_TASK_FIELD_KEYS:
                paths.append(child_path)
            paths.extend(
                _terminology_forbidden_key_paths(nested, path=child_path)
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            paths.extend(
                _terminology_forbidden_key_paths(
                    nested, path=f"{path}[{index}]"
                )
            )
    return paths


def build_lexicon_catalog(terms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for term in terms:
        word = str(term.get("term", "")).strip()
        definition = str(term.get("definition", "")).strip()
        variants = [str(value).strip() for value in term.get("variants", []) if str(value).strip()]
        usage_notes = str(term.get("usage_notes", "") or "").strip()
        ambiguity_notes = str(term.get("ambiguity_notes", "") or "").strip()
        forbidden_paths = _terminology_forbidden_key_paths(term)
        if forbidden_paths:
            raise RetrievalBundleError(
                "terminology-library entries must not contain task-category fields: "
                f"{forbidden_paths}"
            )
        if not word or not definition:
            raise RetrievalBundleError("terminology entry term/definition must be non-empty")
        identifier = term.get("lexicon_id") or stable_term_evidence_id(
            word, definition, variants, usage_notes, ambiguity_notes
        )
        expected = stable_term_evidence_id(
            word, definition, variants, usage_notes, ambiguity_notes
        )
        if identifier != expected:
            raise RetrievalBundleError("lexicon stable ID mismatch")
        rendered = render_lexicon_evidence_block(
            term=word,
            definition=definition,
            variants=variants,
            usage_notes=usage_notes,
            ambiguity_notes=ambiguity_notes,
        )
        evidence_hash = content_sha256(rendered)
        catalog.append(
            {
                "lexicon_id": identifier,
                "term": word,
                "definition": definition,
                "variants": variants,
                "usage_notes": usage_notes,
                "ambiguity_notes": ambiguity_notes,
                "evidence_kind": LEXICON_EVIDENCE_KIND,
                "render_policy": LEXICON_EVIDENCE_RENDER_POLICY,
                "task_label_visibility": LEXICON_TASK_LABEL_VISIBILITY,
                "rendered_block": rendered,
                "rendered_block_sha256": text_sha256(rendered),
                "content_sha256": evidence_hash,
                "source_split": "train",
                "train_only": True,
            }
        )
    # Ordering and semantic-score alignment use only model-visible evidence.
    catalog.sort(key=lambda row: (row["content_sha256"], row["lexicon_id"]))
    if len({row["lexicon_id"] for row in catalog}) != len(catalog):
        raise RetrievalBundleError("lexicon IDs are not unique")
    if len({row["content_sha256"] for row in catalog}) != len(catalog):
        raise RetrievalBundleError(
            "terminology library contains duplicate model-visible evidence blocks"
        )
    return catalog


def _validate_scores(
    scores: np.ndarray,
    *,
    query_count: int,
    corpus_count: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (query_count, corpus_count):
        raise RetrievalBundleError(
            f"{name} score matrix shape {values.shape} != {(query_count, corpus_count)}"
        )
    if not np.isfinite(values).all():
        raise RetrievalBundleError(f"{name} score matrix contains non-finite values")
    return values


def _demo_hit(
    row: Mapping[str, Any],
    score: float,
    rank: int,
    source_class: str,
) -> RetrievalHit:
    return RetrievalHit(
        id=row["demo_id"],
        source_record_id=row["source_record_id"],
        content=row["content"],
        output=row["output"],
        content_sha256=row["content_sha256"],
        gold_sha256=row["gold_sha256"],
        score=float(score),
        rank=rank,
        source_class=source_class,
        method="train-only-bge-cosine",
        provenance={"source_split": "train", "retrieval_bundle_version": RETRIEVAL_BUNDLE_VERSION},
    )


def _lex_hit(
    row: Mapping[str, Any],
    *,
    score: float | None,
    rank: int,
    method: str,
    spans: Sequence[tuple[int, int]] = (),
) -> RetrievalHit:
    return RetrievalHit(
        id=row["lexicon_id"],
        source_record_id=row["term"],
        content=row["rendered_block"],
        output=None,
        content_sha256=row["content_sha256"],
        gold_sha256=None,
        score=score,
        rank=rank,
        source_class=LEXICON_EVIDENCE_KIND,
        method=method,
        provenance={
            "source_split": "train",
            "match_spans": [list(span) for span in spans],
            "retrieval_bundle_version": RETRIEVAL_BUNDLE_VERSION,
        },
    )


def _exact_spans(content: str, values: Sequence[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for value in values:
        start = 0
        while value:
            index = content.find(value, start)
            if index < 0:
                break
            spans.append((index, index + len(value)))
            start = index + 1
    return sorted(set(spans))


def prepare_context_bundle_from_scores(
    *,
    train_records: Sequence[Mapping[str, Any]],
    query_records: Sequence[Mapping[str, Any]],
    lexicon_terms: Sequence[Mapping[str, Any]],
    demo_scores: np.ndarray,
    lexicon_scores: np.ndarray,
    split: str,
    retrieval_config: Mapping[str, Any],
    data_locator: Mapping[str, Any] | None = None,
    lexicon_locator: Mapping[str, Any] | None = None,
    data_dependency: Mapping[str, Any] | None = None,
    lexicon_dependency: Mapping[str, Any] | None = None,
    scorer_provenance: Mapping[str, Any] | None = None,
    fit_demo_records: Sequence[Mapping[str, Any]] | None = None,
    calibration_ids: Sequence[str] | None = None,
    train_partition_dependency: Mapping[str, Any] | None = None,
    train_partition_locator: Mapping[str, Any] | None = None,
    score_evidence: Mapping[str, Any] | None = None,
    expected_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen selector to matrices computed from query content only."""

    if split not in {"train", "dev", "test"}:
        raise RetrievalBundleError("split must be train/dev/test")
    demo_records, partition_dependency, calibration = _partitioned_demo_records(
        train_records=train_records,
        fit_demo_records=fit_demo_records,
        calibration_ids=calibration_ids,
        train_partition_dependency=train_partition_dependency,
    )
    frozen_data_dependency = (
        _portable_dependency(data_dependency, artifact_kind="data")
        if data_dependency is not None
        else None
    )
    frozen_lexicon_dependency = (
        _portable_dependency(lexicon_dependency, artifact_kind="lexicon")
        if lexicon_dependency is not None
        else None
    )
    if (frozen_data_dependency is None) != (frozen_lexicon_dependency is None):
        raise RetrievalBundleError(
            "data and lexicon dependencies must be supplied together"
        )
    for locator, dependency, label in (
        (data_locator, frozen_data_dependency, "data"),
        (lexicon_locator, frozen_lexicon_dependency, "lexicon"),
        (train_partition_locator, partition_dependency, "train-partition"),
    ):
        if locator is not None and dependency is None:
            raise RetrievalBundleError(
                f"{label} locator requires a portable dependency"
            )
        if locator is not None and dependency is not None:
            _assert_locator_identity(locator, dependency, label=label)
    if partition_dependency is not None and frozen_data_dependency is None:
        raise RetrievalBundleError(
            "partitioned retrieval requires portable data/lexicon dependencies"
        )
    order = list(retrieval_config["source_class_order"])
    train_query_pool = build_query_pool(train_records, source_split="train")
    query_pool = build_query_pool(query_records, source_split=split)
    demo_catalog, memberships = build_demo_catalog(
        demo_records, source_class_order=order
    )
    lexicon_catalog = build_lexicon_catalog(lexicon_terms)
    demo_values = _validate_scores(
        demo_scores,
        query_count=len(query_records),
        corpus_count=len(demo_catalog),
        name="demo",
    )
    lex_values = _validate_scores(
        lexicon_scores,
        query_count=len(query_records),
        corpus_count=len(lexicon_catalog),
        name="lexicon",
    )
    query_ids = [str(record["id"]) for record in query_pool]
    demo_ids = [str(record["demo_id"]) for record in demo_catalog]
    lexicon_ids = [str(record["lexicon_id"]) for record in lexicon_catalog]
    score_frame = {
        "query_ids": query_ids,
        "query_texts": [str(record["content"]) for record in query_pool],
        "demo_ids": demo_ids,
        "demo_texts": [str(record["content"]) for record in demo_catalog],
        "lexicon_ids": lexicon_ids,
        "lexicon_texts": [
            str(record["rendered_block"]) for record in lexicon_catalog
        ],
    }
    if score_evidence is None:
        frozen_score_evidence = build_score_evidence(
            **score_frame,
            demo_scores=demo_values,
            lexicon_scores=lex_values,
        )
    else:
        frozen_demo_values, frozen_lexicon_values = decode_score_evidence(
            score_evidence, **score_frame
        )
        if (
            not np.array_equal(
                frozen_demo_values.view(np.uint64), demo_values.view(np.uint64)
            )
            or not np.array_equal(
                frozen_lexicon_values.view(np.uint64), lex_values.view(np.uint64)
            )
        ):
            raise RetrievalBundleError(
                "frozen score evidence differs from supplied score matrices"
            )
        del frozen_demo_values, frozen_lexicon_values
        frozen_score_evidence = dict(score_evidence)
    quota = dict(retrieval_config["allocated_class_top_k"])
    multiplier = int(retrieval_config["candidate_multiplier"])
    threshold = float(retrieval_config["similarity_threshold"])
    records: list[dict[str, Any]] = []
    expected_records: list[Any] | None = None
    if expected_bundle is not None:
        candidate_records = expected_bundle.get("records")
        if (
            not isinstance(candidate_records, list)
            or len(candidate_records) != len(query_records)
        ):
            raise RetrievalBundleError(
                "expected prepared bundle has an invalid record frame"
            )
        expected_records = candidate_records
    for query_index, raw_query in enumerate(query_records):
        query_id, query_content, query_quads = _record(raw_query)
        query_content_hash = content_sha256(query_content)
        hits_by_class: dict[str, list[RetrievalHit]] = {}
        for source_class in order:
            eligible = [index for index, classes in enumerate(memberships) if source_class in classes]
            eligible.sort(
                key=lambda index: (
                    -round_similarity_half_even(float(demo_values[query_index, index])),
                    demo_catalog[index]["demo_id"],
                )
            )
            hits_by_class[source_class] = [
                _demo_hit(
                    demo_catalog[index],
                    float(demo_values[query_index, index]),
                    rank,
                    source_class,
                )
                for rank, index in enumerate(eligible)
            ]
        demo_trace = select_demos(
            hits_by_class,
            source_class_order=order,
            allocated_class_top_k=quota,
            similarity_threshold=threshold,
            candidate_multiplier=multiplier,
            query_source_record_id=query_id if split == "train" else None,
            query_content_sha256=query_content_hash,
        )

        exact_hits: list[RetrievalHit] = []
        for lex_row in lexicon_catalog:
            spans = _exact_spans(query_content, [lex_row["term"], *lex_row["variants"]])
            if spans:
                exact_hits.append(
                    _lex_hit(
                        lex_row,
                        score=None,
                        rank=len(exact_hits),
                        method="train-only-exact-substring",
                        spans=spans,
                    )
                )
        semantic_order = sorted(
            range(len(lexicon_catalog)),
            key=lambda index: (
                -round_similarity_half_even(float(lex_values[query_index, index])),
                lexicon_catalog[index]["content_sha256"],
                lexicon_catalog[index]["lexicon_id"],
            ),
        )
        semantic_hits = [
            _lex_hit(
                lexicon_catalog[index],
                score=float(lex_values[query_index, index]),
                rank=rank,
                method="train-only-bge-cosine",
            )
            for rank, index in enumerate(semantic_order)
        ]
        lex_trace = select_lexicons(
            exact_hits,
            semantic_hits,
            exact_top_k=int(retrieval_config["lex_exact_top_k"]),
            semantic_top_k=int(retrieval_config["lex_semantic_top_k"]),
            similarity_threshold=None,
        )
        control_relevance: dict[str, Any] = {}
        if split in {"dev", "test"}:
            control_relevance = {
                "demos": [
                    {
                        "demo_id": row["demo_id"],
                        "written_similarity": round_similarity_half_even(
                            float(demo_values[query_index, corpus_index])
                        ),
                    }
                    for corpus_index, row in enumerate(demo_catalog)
                ],
                "lexicons": [
                    {
                        "lexicon_id": row["lexicon_id"],
                        "written_similarity": round_similarity_half_even(
                            float(lex_values[query_index, corpus_index])
                        ),
                    }
                    for corpus_index, row in enumerate(lexicon_catalog)
                ],
            }
        demo_payload = demo_trace.to_dict()
        lex_payload = lex_trace.to_dict()
        prepared_record = {
                "query": {
                    "id": query_id,
                    "content": query_content,
                    "content_sha256": query_content_hash,
                    "gold": [
                        {
                            "target": quad.target,
                            "argument": quad.argument,
                            "targeted_group": list(quad.targeted_group),
                            "hateful": quad.hateful,
                        }
                        for quad in query_quads
                    ],
                    "gold_sha256": hashlib.sha256(
                        serialize_quadruples(query_quads).encode("utf-8")
                    ).hexdigest(),
                },
                "selection": {
                    "demos": {
                        **demo_payload,
                        "prompt_order_before_budget": list(demo_trace.prompt_order),
                    },
                    "lexicons": {
                        **lex_payload,
                        "prompt_order_before_budget": list(lex_trace.prompt_order),
                    },
                },
                "retrieval": {
                    "schema_version": "stage1-retrieval-trace/v1",
                    "demo": demo_payload,
                    "lexicon": lex_payload,
                },
                **({"control_relevance": control_relevance} if control_relevance else {}),
            }
        if expected_records is not None:
            if prepared_record != expected_records[query_index]:
                raise RetrievalBundleError(
                    f"prepared record {query_id!r} differs from exact selector replay"
                )
        else:
            records.append(prepared_record)
    provenance = {
        "schema_version": "stage1-retrieval-provenance/v1",
        "policy_version": (
            RETRIEVAL_BUNDLE_VERSION
            if partition_dependency is not None
            else LEGACY_RETRIEVAL_BUNDLE_VERSION
        ),
        "train_only_demo_pool": True,
        "train_only_lexicon_pool": True,
        "saw_dev_test_labels_during_pool_build": False,
        "saw_model_predictions": False,
        "score_matrix": {
            "demo_sha256": _score_sha256(demo_values),
            "lexicon_sha256": _score_sha256(lex_values),
            "evidence_sha256": _canonical_sha(frozen_score_evidence),
            "query_texts_sha256": frozen_score_evidence["query_texts_sha256"],
            "demo_texts_sha256": frozen_score_evidence["demo_texts_sha256"],
            "lexicon_texts_sha256": frozen_score_evidence[
                "lexicon_texts_sha256"
            ],
        },
        "retrieval_config_sha256": _canonical_sha(dict(retrieval_config)),
        **retrieval_code_sha256s(),
        "scorer": dict(scorer_provenance or {}),
    }
    if partition_dependency is None:
        provenance["all_train_pool_relevance_complete"] = split in {"dev", "test"}
    else:
        fit_ids = [str(record["id"]) for record in demo_records]
        provenance.update(
            {
                "fit_only_demo_pool": True,
                "calibration_demo_excluded": True,
                "all_fit_demo_pool_relevance_complete": split in {"dev", "test"},
                "fit_demo_record_count": len(fit_ids),
                "fit_demo_ids_sha256": _canonical_sha(fit_ids),
                "calibration_record_count": len(calibration),
                "calibration_ids_sha256": _canonical_sha(list(calibration)),
                "train_partition_dependency": partition_dependency,
            }
        )
    result = {
        "schema_version": BUNDLE_SCHEMA,
        "split": split,
        "demo_catalog": demo_catalog,
        "lexicon_catalog": lexicon_catalog,
        "train_query_pool": train_query_pool,
        "query_pool": query_pool,
        "score_evidence": frozen_score_evidence,
        "records": expected_records if expected_records is not None else records,
        "retrieval_provenance": provenance,
    }
    if frozen_data_dependency is not None:
        result["data_dependency"] = frozen_data_dependency
        result["lexicon_dependency"] = frozen_lexicon_dependency
    if partition_dependency is not None:
        result["train_partition_dependency"] = partition_dependency
    result["bundle_sha256"] = _canonical_sha(result)
    if expected_bundle is not None and result != dict(expected_bundle):
        raise RetrievalBundleError(
            "prepared bundle differs from exact score/selector replay"
        )
    return result


def cosine_score_matrices(
    *,
    model: Any,
    train_texts: Sequence[str],
    query_texts: Sequence[str],
    lexicon_texts: Sequence[str],
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode content-only inputs and return deterministic float32 cosine matrices."""

    def encode(texts: Sequence[str]) -> np.ndarray:
        values = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts) or not np.isfinite(array).all():
            raise RetrievalBundleError("embedding backend returned an invalid matrix")
        return array

    query = encode(query_texts)
    train = encode(train_texts)
    lexicon = encode(lexicon_texts)
    return (query @ train.T).astype(np.float32), (query @ lexicon.T).astype(np.float32)


__all__ = [
    "BUNDLE_SCHEMA",
    "LEGACY_RETRIEVAL_BUNDLE_VERSION",
    "LEXICON_EVIDENCE_RENDER_POLICY",
    "LEXICON_EVIDENCE_KIND",
    "LEXICON_TASK_LABEL_VISIBILITY",
    "RETRIEVAL_BUNDLE_VERSION",
    "SCORE_EVIDENCE_SCHEMA",
    "RetrievalBundleError",
    "build_score_evidence",
    "build_demo_catalog",
    "build_lexicon_catalog",
    "build_query_pool",
    "cosine_score_matrices",
    "decode_score_evidence",
    "prepare_context_bundle_from_scores",
    "render_lexicon_evidence_block",
    "retrieval_code_sha256s",
]
