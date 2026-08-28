"""Content-addressed Stage 1 train-only evidence artifact.

The artifact copies only the selected train query, gold output, and final L/D
evidence *sets* needed by the schedule builder.  Set members are sorted by
their stable IDs; this canonical storage order is not an epoch presentation
order.  Per-epoch demo order and source dropout belong exclusively to the
downstream training schedule.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from data.context_manifest import text_sha256, validate_context_record
from data.retrieval_bundle import (
    LEXICON_EVIDENCE_RENDER_POLICY,
    LEXICON_TASK_LABEL_VISIBILITY,
)
from data.training_artifacts import (
    TrainingArtifactError,
    canonical_json_bytes,
    canonical_sha256,
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
    write_locator_ref,
)
from data.train_partition import (
    ARTIFACT_KIND as TRAIN_PARTITION_ARTIFACT_KIND,
    TrainPartitionBundle,
    load_train_partition,
    validate_train_partition_target,
)
from utils.quadruple import serialize_quadruples


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPOSITORY_ROOT / "schemas/stage1_training_evidence_v1.schema.json"
ARTIFACT_KIND = "training-evidence"
CONTEXT_ARTIFACT_KIND = "context"
BASE_MODEL_ARTIFACT_KIND = "stage1-model"
EVIDENCE_SCHEMA_VERSION = "stage1-training-evidence/v1"
RECORD_SCHEMA_VERSION = "stage1-training-evidence-record/v1"
CONFIG_SCHEMA_VERSION = "stage1-training-evidence-config/v1"
PROVENANCE_SCHEMA_VERSION = "stage1-training-evidence-provenance/v1"
EVIDENCE_SET_POLICY = "canonical-id-set-no-epoch-order/v1"
BUDGET_POLICY = "prompt-plus-gold-plus-eos-hard-fail/v1"
RENDERER_REVISION = "stage1-training-evidence-renderer/v1"
EVIDENCE_REPLAY_POLICY = "exact-context-dependency-replay/v1"
CONTEXT_POLICY_SCHEMA_VERSION = "stage1-context-policy-lineage/v1"
CANONICAL_DECIMAL_RE = re.compile(r"^[1-9][0-9]*$")
DEMO_ID_RE = re.compile(r"^demo:v1:[0-9a-f]{64}$")
LEXICON_ID_RE = re.compile(r"^lex:v2:[0-9a-f]{64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TrainingEvidenceError(TrainingArtifactError):
    """Raised when train-only evidence cannot be frozen or validated."""


def _builder_code_sha256() -> str:
    return sha256_file(Path(__file__))


def _resolve_prompt(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingEvidenceError(f"context rendering {field} is missing")
    if "{" in value or "\n" in value:
        resolved = value
    else:
        try:
            prompt_module = importlib.import_module("prompt")
            resolved = getattr(prompt_module, value)
        except (ImportError, AttributeError) as exc:
            raise TrainingEvidenceError(
                f"cannot resolve context rendering symbol {value!r}"
            ) from exc
    if not isinstance(resolved, str) or not resolved:
        raise TrainingEvidenceError(f"resolved context rendering {field} is invalid")
    return resolved.replace("\r\n", "\n")


def _dependency_identity(value: Any, *, kind: str) -> dict[str, Any]:
    """Project locator/portable refs to one machine-independent identity."""

    if not isinstance(value, Mapping):
        raise TrainingEvidenceError(f"context policy lacks {kind} dependency")
    expected = {
        "schema_version": "stage1-dependency-ref/v1",
        "artifact_kind": (
            TRAIN_PARTITION_ARTIFACT_KIND
            if kind == "train_partition"
            else kind
        ),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise TrainingEvidenceError(
                f"context policy has an invalid {kind} dependency {key}"
            )
    artifact_id = value.get("artifact_id")
    payload_hash = value.get("payload_manifest_sha256")
    if not isinstance(artifact_id, str) or not artifact_id:
        raise TrainingEvidenceError(
            f"context policy has an invalid {kind} artifact ID"
        )
    if not isinstance(payload_hash, str) or not SHA256_RE.fullmatch(payload_hash):
        raise TrainingEvidenceError(
            f"context policy has an invalid {kind} payload hash"
        )
    return {
        "schema_version": "stage1-dependency-identity/v1",
        "artifact_kind": expected["artifact_kind"],
        "artifact_id": artifact_id,
        "payload_manifest_sha256": payload_hash,
    }


def context_policy_snapshot(
    context_target: str | Path,
    *,
    expected_split: str | None = None,
    require_scientific: bool = False,
    context_meta: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Recompute the split-invariant context policy carried by a target.

    Query frames, score matrices, and selected evidence are intentionally not
    part of this projection: those remain bound by the context dependency.
    The projection is the bridge used to prove that train, dev, and sealed
    test artifacts applied the same retrieval/rendering/budget protocol.
    """

    target = Path(context_target)
    if context_meta is None:
        meta_paths = sorted(target.glob("context_manifest.*.meta.json"))
        if len(meta_paths) != 1:
            raise TrainingEvidenceError(
                "context policy requires exactly one context meta file"
            )
        raw_meta = load_json(meta_paths[0])
        if not isinstance(raw_meta, Mapping):
            raise TrainingEvidenceError("context policy meta is not an object")
        meta = dict(raw_meta)
    else:
        meta = dict(context_meta)
    if meta.get("schema_version") != "stage1-context-manifest/v1":
        raise TrainingEvidenceError("context policy has the wrong meta schema")
    split = meta.get("split")
    if split not in {"train", "dev", "test"}:
        raise TrainingEvidenceError("context policy has an invalid split")
    if expected_split is not None and split != expected_split:
        raise TrainingEvidenceError(
            f"context policy split {split!r} differs from required {expected_split!r}"
        )
    expected_kind = "test-context" if split == "test" else "context"
    if meta.get("artifact_kind") != expected_kind:
        raise TrainingEvidenceError("context meta artifact kind/split mismatch")
    if require_scientific and meta.get("scientific_eligible") is not True:
        raise TrainingEvidenceError("context policy requires a scientific artifact")

    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise TrainingEvidenceError("context policy lacks context ID inputs")
    expected_formal = meta.get("scientific_eligible") is True
    if (
        id_inputs.get("split") != split
        or id_inputs.get("artifact_kind") != expected_kind
        or id_inputs.get("formal") is not expected_formal
    ):
        raise TrainingEvidenceError("context meta and ID inputs disagree")
    builder_hash = id_inputs.get("builder_code_sha256")
    if not isinstance(builder_hash, str) or not SHA256_RE.fullmatch(builder_hash):
        raise TrainingEvidenceError("context policy lacks a builder code hash")

    config = load_json(target / "config.resolved.json")
    if not isinstance(config, Mapping):
        raise TrainingEvidenceError("context policy config is not an object")
    if config.get("schema_version") != "stage1-context-factorial-config/v1":
        raise TrainingEvidenceError("context policy has the wrong config schema")
    if id_inputs.get("context_config_sha256") != canonical_sha256(config):
        raise TrainingEvidenceError("context config hash differs from context ID inputs")
    sections: dict[str, Mapping[str, Any]] = {}
    for field in ("retrieval", "rendering", "budget", "output_protocol"):
        value = config.get(field)
        if not isinstance(value, Mapping):
            raise TrainingEvidenceError(f"context policy lacks config section {field}")
        sections[field] = value
    budget_meta = meta.get("budget")
    tokenizer_revision = sections["budget"].get("tokenizer_revision")
    if (
        not isinstance(budget_meta, Mapping)
        or not isinstance(tokenizer_revision, str)
        or not tokenizer_revision
        or budget_meta.get("tokenizer_revision") != tokenizer_revision
        or id_inputs.get("tokenizer_revision") != tokenizer_revision
    ):
        raise TrainingEvidenceError("context tokenizer policy lineage is inconsistent")

    rendering = sections["rendering"]
    if (
        rendering.get("lexicon_evidence_policy")
        != LEXICON_EVIDENCE_RENDER_POLICY
        or rendering.get("task_label_visibility")
        != LEXICON_TASK_LABEL_VISIBILITY
    ):
        raise TrainingEvidenceError(
            "context policy does not freeze category-free terminology evidence"
        )
    resolved_prompt_hashes: dict[str, str | None] = {}
    for key in ("system_prompt", "user_prompt", "example_prompt"):
        raw = rendering.get(key)
        if raw is None and key == "example_prompt":
            resolved_prompt_hashes[key] = None
            continue
        resolved_prompt_hashes[key] = text_sha256(
            _resolve_prompt(raw, field=key)
        )

    provenance = load_json(target / "provenance.json")
    if not isinstance(provenance, Mapping):
        raise TrainingEvidenceError("context policy provenance is not an object")
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise TrainingEvidenceError("context policy lacks source dependencies")
    if require_scientific and set(dependencies) != {
        "data",
        "train_partition",
        "lexicon",
    }:
        raise TrainingEvidenceError(
            "scientific context policy requires exact data/lexicon dependencies"
        )
    source_dependencies: dict[str, Any] = {}
    for kind in ("data", "train_partition", "lexicon"):
        if kind in dependencies:
            source_dependencies[kind] = _dependency_identity(
                dependencies[kind], kind=kind
            )

    prepared_meta = load_json(target / "prepared_bundle.meta.json")
    if not isinstance(prepared_meta, Mapping):
        raise TrainingEvidenceError("context policy lacks prepared-bundle metadata")
    retrieval_provenance = prepared_meta.get("retrieval_provenance")
    if not isinstance(retrieval_provenance, Mapping):
        raise TrainingEvidenceError("context policy lacks retrieval provenance")
    retrieval_engine = {
        "schema_version": retrieval_provenance.get("schema_version"),
        "policy_version": retrieval_provenance.get("policy_version"),
        "scorer": copy.deepcopy(retrieval_provenance.get("scorer", {})),
    }
    if require_scientific and (
        not isinstance(retrieval_engine["schema_version"], str)
        or not retrieval_engine["schema_version"]
        or not isinstance(retrieval_engine["policy_version"], str)
        or not retrieval_engine["policy_version"]
        or not isinstance(retrieval_engine["scorer"], Mapping)
        or not retrieval_engine["scorer"]
    ):
        raise TrainingEvidenceError(
            "scientific context policy lacks typed retrieval-engine lineage"
        )

    snapshot = {
        "schema_version": CONTEXT_POLICY_SCHEMA_VERSION,
        "context_config_schema_version": config["schema_version"],
        "context_config_sha256": canonical_sha256(config),
        "source_dependencies": source_dependencies,
        "retrieval_policy": copy.deepcopy(dict(sections["retrieval"])),
        "retrieval_engine": retrieval_engine,
        "rendering_policy": copy.deepcopy(dict(rendering)),
        "resolved_prompt_sha256": resolved_prompt_hashes,
        "budget_policy": copy.deepcopy(dict(sections["budget"])),
        "output_protocol": copy.deepcopy(dict(sections["output_protocol"])),
        "tokenizer_revision": tokenizer_revision,
        "context_builder_code_sha256": builder_hash,
    }
    return snapshot, canonical_sha256(snapshot)


@contextmanager
def _registered_evidence_tokenizer(
    base_model_dependency: Mapping[str, Any],
    *,
    workspace_root: Path,
    tokenizer: Any | None = None,
) -> Any:
    """Resolve the evidence tokenizer only from a frozen base-model dependency."""

    # Keep setup error translation *before* the generator yield.  An exception
    # raised by evidence/context replay is thrown back into this generator at
    # ``yield``; wrapping that exception as a tokenizer-resolution failure would
    # hide the real failed invariant even though the source lease cleaned up
    # correctly.
    try:
        dependency = validate_dependency_ref(
            base_model_dependency, expected_kind=BASE_MODEL_ARTIFACT_KIND
        )
        target = resolve_dependency_target(dependency, workspace_root)
        from model.stage1_registry import (
            ResolvedModelSourceContract,
            validate_model_artifact_target,
            verified_model_source_lease,
        )

        model = validate_model_artifact_target(
            target,
            workspace_root=workspace_root,
            tokenizer=tokenizer,
        )
        if (
            model.get("artifact_type") != "base"
            or model.get("checkpoint_format") != "base"
        ):
            raise TrainingEvidenceError(
                "training evidence requires the registered base model artifact"
            )
        contract = ResolvedModelSourceContract(
            workspace_root=workspace_root.resolve(),
            checkpoint_inventory=model["checkpoint_inventory"],
            tokenizer_inventory=model["tokenizer_inventory"],
            base_inventory=model["base_inventory"],
        )
    except TrainingEvidenceError:
        raise
    except Exception as exc:
        raise TrainingEvidenceError(
            f"registered evidence tokenizer validation failed: {exc}"
        ) from exc

    # Do not place a broad exception handler around this ``yield``.  The lease
    # context owns failure cleanup and post-operation source revalidation, while
    # caller exceptions must preserve their original type and message.
    with verified_model_source_lease(
        contract, source_names=("tokenizer", "base")
    ) as sources:
        try:
            if tokenizer is None:
                try:
                    from transformers import AutoTokenizer
                except ImportError as exc:
                    raise TrainingEvidenceError(
                        "transformers is required to count actual gold tokens"
                    ) from exc
                tokenizer = AutoTokenizer.from_pretrained(
                    str(sources.tokenizer_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
            if getattr(tokenizer, "eos_token_id", None) is None:
                raise TrainingEvidenceError(
                    "registered evidence tokenizer has no EOS token"
                )
        except TrainingEvidenceError:
            raise
        except Exception as exc:
            raise TrainingEvidenceError(
                f"registered evidence tokenizer construction failed: {exc}"
            ) from exc
        yield tokenizer, dependency, model


def load_frozen_evidence_tokenizer(
    evidence_target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> Any:
    """Reject the unsafe legacy API that returned a lease-scoped tokenizer.

    Tokenizer objects may retain lazy readers or backend state tied to mutable
    source files.  Returning one after source verification has ended is never a
    valid formal operation; callers must put their complete operation inside
    :func:`frozen_evidence_tokenizer_lease`.
    """

    del evidence_target, workspace_root
    raise TrainingEvidenceError(
        "returning a tokenizer outside its verified source lease is forbidden; "
        "use frozen_evidence_tokenizer_lease"
    )


@contextmanager
def frozen_evidence_tokenizer_lease(
    evidence_target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    tokenizer: Any | None = None,
) -> Any:
    """Yield the exact evidence tokenizer for one complete guarded operation."""

    root = Path(workspace_root).resolve()
    target = Path(evidence_target).resolve()
    base_model_dependency = _read_dependency(
        target, "base_model_ref.json", BASE_MODEL_ARTIFACT_KIND
    )
    with _registered_evidence_tokenizer(
        base_model_dependency,
        workspace_root=root,
        tokenizer=tokenizer,
    ) as resolved:
        yield resolved[0]


def _token_count(text: str, tokenizer: Any) -> int:
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except AttributeError:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    count = len(encoded)
    if count <= 0:
        raise TrainingEvidenceError("canonical gold output tokenized to an empty sequence")
    return count


def _read_dependency(target: Path, filename: str, expected_kind: str) -> dict[str, Any]:
    value = load_json(target / filename)
    if not isinstance(value, dict):
        raise TrainingEvidenceError(f"{filename} is not a dependency object")
    return validate_dependency_ref(value, expected_kind=expected_kind)


def _catalog_id(row: Mapping[str, Any], *, kind: str) -> str:
    keys = (f"{kind}_id", "id", "block_id", "evidence_id")
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise TrainingEvidenceError(f"{kind} catalog row lacks a stable ID")


def _load_catalog(target: Path, relative_path: str, *, kind: str) -> dict[str, dict[str, Any]]:
    path = target / relative_path
    rows = load_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = _catalog_id(row, kind=kind)
        if item_id in result:
            raise TrainingEvidenceError(f"duplicate {kind} catalog ID: {item_id}")
        result[item_id] = dict(row)
    return result


def _optional_context_blocks(target: Path) -> dict[str, dict[str, Any]]:
    path = target / "catalogs/context_blocks.jsonl"
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        identifiers = [
            value
            for key in ("block_id", "evidence_id", "demo_id", "lexicon_id", "id")
            if isinstance((value := row.get(key)), str) and value
        ]
        if not identifiers:
            raise TrainingEvidenceError("context block row lacks a stable evidence ID")
        for item_id in identifiers:
            if item_id in result and result[item_id] != row:
                raise TrainingEvidenceError(f"duplicate context block ID: {item_id}")
            result[item_id] = dict(row)
    return result


def _selected_item(
    item_id: str,
    *,
    kind: str,
    catalog: Mapping[str, Mapping[str, Any]],
    blocks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if item_id not in catalog:
        raise TrainingEvidenceError(f"selected {kind} ID is absent from train catalog: {item_id}")
    row = catalog[item_id]
    block_row = blocks.get(item_id, {})
    rendered_block = row.get("rendered_block", block_row.get("rendered_block"))
    if not isinstance(rendered_block, str) or not rendered_block:
        raise TrainingEvidenceError(f"selected {kind} lacks rendered_block: {item_id}")
    block_hash = row.get(
        "rendered_block_sha256", block_row.get("rendered_block_sha256")
    )
    actual_hash = text_sha256(rendered_block)
    if block_hash is not None and block_hash != actual_hash:
        raise TrainingEvidenceError(f"selected {kind} block hash mismatch: {item_id}")
    if kind == "demo":
        if not DEMO_ID_RE.fullmatch(item_id):
            raise TrainingEvidenceError(f"invalid demo ID: {item_id}")
        source_record_id = row.get("source_record_id")
        if not isinstance(source_record_id, str) or not CANONICAL_DECIMAL_RE.fullmatch(
            source_record_id
        ):
            raise TrainingEvidenceError(f"demo lacks canonical source_record_id: {item_id}")
        return {
            "demo_id": item_id,
            "source_record_id": source_record_id,
            "rendered_block": rendered_block.replace("\r\n", "\n"),
            "rendered_block_sha256": actual_hash,
        }
    if not LEXICON_ID_RE.fullmatch(item_id):
        raise TrainingEvidenceError(f"invalid lexicon ID: {item_id}")
    return {
        "lexicon_id": item_id,
        "rendered_block": rendered_block.replace("\r\n", "\n"),
        "rendered_block_sha256": actual_hash,
    }


def _record_semantic_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key not in {"training_evidence_build_id", "record_sha256"}
    }


def recompute_evidence_record_sha256(record: Mapping[str, Any]) -> str:
    return canonical_sha256(_record_semantic_payload(record))


def _validate_record(record: Mapping[str, Any], *, schema_path: Path) -> None:
    validate_json_schema(record, schema_path)
    query = record["query"]
    query_id = query["id"]
    if query["content_sha256"] != text_sha256(query["content"]):
        raise TrainingEvidenceError(f"query content hash mismatch: {query_id}")
    if query["gold_text"] != serialize_quadruples(query["gold"]):
        raise TrainingEvidenceError(f"canonical gold text mismatch: {query_id}")
    if query["gold_sha256"] != hashlib.sha256(
        query["gold_text"].encode("utf-8")
    ).hexdigest():
        raise TrainingEvidenceError(f"canonical gold hash mismatch: {query_id}")
    if query["actual_gold_tokens"] <= 0:
        raise TrainingEvidenceError(f"invalid actual gold token count: {query_id}")
    lexicon = record["lexicon_evidence"]
    demos = record["demo_evidence"]
    lexicon_ids = [item["lexicon_id"] for item in lexicon["items"]]
    demo_ids = [item["demo_id"] for item in demos["items"]]
    if lexicon["ids"] != sorted(lexicon_ids) or lexicon_ids != sorted(set(lexicon_ids)):
        raise TrainingEvidenceError(f"lexicon evidence is not canonical set encoding: {query_id}")
    if demos["ids"] != sorted(demo_ids) or demo_ids != sorted(set(demo_ids)):
        raise TrainingEvidenceError(f"demo evidence is not canonical set encoding: {query_id}")
    for item in [*lexicon["items"], *demos["items"]]:
        if item["rendered_block_sha256"] != text_sha256(item["rendered_block"]):
            raise TrainingEvidenceError(f"evidence block hash mismatch: {query_id}")
    budget = record["budget"]
    if query["actual_gold_tokens"] + budget["eos_tokens"] > budget[
        "completion_reserve_tokens"
    ]:
        raise TrainingEvidenceError(f"gold output exceeds completion reserve: {query_id}")
    if budget["source_cld_prompt_tokens"] + query["actual_gold_tokens"] + budget[
        "eos_tokens"
    ] > budget["max_sequence_tokens"]:
        raise TrainingEvidenceError(f"source CLD prompt plus gold overflows: {query_id}")
    if record["record_sha256"] != recompute_evidence_record_sha256(record):
        raise TrainingEvidenceError(f"training evidence record hash mismatch: {query_id}")


def _context_revision(meta: Mapping[str, Any]) -> str:
    budget = meta.get("budget")
    if not isinstance(budget, Mapping):
        raise TrainingEvidenceError("train context meta lacks budget")
    value = budget.get("tokenizer_revision")
    if not isinstance(value, str) or not value:
        raise TrainingEvidenceError("train context meta lacks tokenizer_revision")
    return value


def _build_records(
    *,
    context_target: Path,
    context_meta: Mapping[str, Any],
    context_config: Mapping[str, Any],
    tokenizer: Any,
    schema_path: Path,
    partition: TrainPartitionBundle,
) -> list[dict[str, Any]]:
    context_manifest_path = context_target / "context_manifest.train.jsonl"
    rows = load_jsonl(context_manifest_path)
    query_pool_rows = load_jsonl(
        context_target / "catalogs/query_pool.train.jsonl"
    )
    demo_catalog = _load_catalog(
        context_target, "catalogs/demo_pool.train.jsonl", kind="demo"
    )
    lexicon_catalog = _load_catalog(
        context_target, "catalogs/lexicon_pool.jsonl", kind="lexicon"
    )
    blocks = _optional_context_blocks(context_target)
    rendering_config = context_config.get("rendering")
    if not isinstance(rendering_config, Mapping):
        raise TrainingEvidenceError("context config lacks rendering settings")
    system_prompt = _resolve_prompt(
        rendering_config.get("system_prompt"), field="system_prompt"
    )
    user_prompt_template = _resolve_prompt(
        rendering_config.get("user_prompt"), field="user_prompt"
    )
    if not all(token in user_prompt_template for token in ("{lexicons}", "{examples}", "{text}")):
        raise TrainingEvidenceError("resolved user prompt lacks required placeholders")
    thinking_mode = rendering_config.get("thinking_mode")
    if thinking_mode is not False:
        raise TrainingEvidenceError("Stage 1 evidence requires thinking_mode=false")
    records: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    ordered_context_query_ids: list[str] = []
    expected_context_id = context_meta.get("context_build_id")
    partition_by_id = {row["query_id"]: row for row in partition.rows}
    source_record_by_id = {
        str(record.get("id")): record for record in partition.train_records
    }
    query_pool_by_id = {
        str(row.get("id")): row for row in query_pool_rows
    }
    ordered_partition_ids = [row["query_id"] for row in partition.rows]
    ordered_query_pool_ids = [str(row.get("id")) for row in query_pool_rows]
    if (
        ordered_query_pool_ids != ordered_partition_ids
        or len(query_pool_by_id) != len(query_pool_rows)
    ):
        raise TrainingEvidenceError(
            "train context query-pool frame differs from the immutable partition"
        )
    calibration_content_hashes = {
        row["content_sha256"]
        for row in partition.rows
        if row["partition"] == "calibration"
    }
    for context_record in rows:
        try:
            validate_context_record(context_record)
        except Exception as exc:
            raise TrainingEvidenceError(f"invalid train context record: {exc}") from exc
        if context_record.get("context_build_id") != expected_context_id:
            raise TrainingEvidenceError("train context record has the wrong context_build_id")
        query = context_record["query"]
        query_id = query.get("id")
        if not isinstance(query_id, str) or not CANONICAL_DECIMAL_RE.fullmatch(query_id):
            raise TrainingEvidenceError("train context query ID is not canonical decimal")
        if query_id in query_ids:
            raise TrainingEvidenceError(f"duplicate train context query ID: {query_id}")
        query_ids.add(query_id)
        ordered_context_query_ids.append(query_id)
        partition_row = partition_by_id.get(query_id)
        source_record = source_record_by_id.get(query_id)
        query_pool_row = query_pool_by_id.get(query_id)
        if partition_row is None:
            raise TrainingEvidenceError(
                f"train query is absent from the immutable partition: {query_id}"
            )
        if source_record is None or query_pool_row is None:
            raise TrainingEvidenceError(
                f"train query is absent from its frozen source frame: {query_id}"
            )
        if query.get("split", "train") != "train":
            raise TrainingEvidenceError(f"non-train query in training evidence input: {query_id}")
        content = query.get("content")
        gold = query.get("gold")
        if not isinstance(content, str) or not content or not isinstance(gold, list):
            raise TrainingEvidenceError(f"query content/gold is incomplete: {query_id}")
        gold_text = serialize_quadruples(gold)
        source_content = source_record.get("content")
        source_gold = source_record.get("quadruples", source_record.get("gold"))
        if (
            not isinstance(source_content, str)
            or content != source_content
            or text_sha256(content) != partition_row.get("content_sha256")
            or query.get("content_sha256") != text_sha256(content)
            or serialize_quadruples(source_gold) != gold_text
            or query.get("gold_sha256")
            != hashlib.sha256(gold_text.encode("utf-8")).hexdigest()
            or query_pool_row.get("content") != content
            or query_pool_row.get("content_sha256") != text_sha256(content)
            or query_pool_row.get("gold_sha256")
            != hashlib.sha256(gold_text.encode("utf-8")).hexdigest()
            or serialize_quadruples(query_pool_row.get("quadruples")) != gold_text
            or query_pool_row.get("source_split") != "train"
        ):
            raise TrainingEvidenceError(
                f"train context query frame differs from data/partition/catalog: {query_id}"
            )
        actual_gold_tokens = _token_count(gold_text, tokenizer)
        selection = context_record["selection"]
        lexicon_ids = selection["lexicons"]["prompt_order_final"]
        demo_ids = selection["demos"]["prompt_order_final"]
        if len(lexicon_ids) != len(set(lexicon_ids)) or len(demo_ids) != len(set(demo_ids)):
            raise TrainingEvidenceError(f"selected evidence IDs are not unique: {query_id}")
        lexicon_items = sorted(
            (
                _selected_item(
                    item_id,
                    kind="lexicon",
                    catalog=lexicon_catalog,
                    blocks=blocks,
                )
                for item_id in lexicon_ids
            ),
            key=lambda item: item["lexicon_id"],
        )
        demo_items = sorted(
            (
                _selected_item(
                    item_id,
                    kind="demo",
                    catalog=demo_catalog,
                    blocks=blocks,
                )
                for item_id in demo_ids
            ),
            key=lambda item: item["demo_id"],
        )
        for item in demo_items:
            source_id = item["source_record_id"]
            source_partition = partition_by_id.get(source_id)
            if source_partition is None:
                raise TrainingEvidenceError(
                    f"demo source is absent from the immutable partition: {source_id}"
                )
            if source_partition["partition"] != "fit":
                raise TrainingEvidenceError(
                    f"demo source is not fit-only: {source_id}"
                )
            if source_partition["content_sha256"] in calibration_content_hashes:
                raise TrainingEvidenceError(
                    f"demo source content overlaps calibration: {source_id}"
                )
        context_budget = context_record["budget"]
        record: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "training_evidence_build_id": "tevd-" + "0" * 64,
            "query": {
                "id": query_id,
                "partition": partition_row["partition"],
                "content": content.replace("\r\n", "\n"),
                "content_sha256": text_sha256(content),
                "gold": copy.deepcopy(gold),
                "gold_text": gold_text,
                "gold_sha256": hashlib.sha256(gold_text.encode("utf-8")).hexdigest(),
                "actual_gold_tokens": actual_gold_tokens,
            },
            "lexicon_evidence": {
                "ids": [item["lexicon_id"] for item in lexicon_items],
                "items": lexicon_items,
            },
            "demo_evidence": {
                "ids": [item["demo_id"] for item in demo_items],
                "items": demo_items,
            },
            "rendering": {
                "system_prompt": system_prompt,
                "user_prompt_template": user_prompt_template,
                "thinking_mode": False,
            },
            "budget": {
                "max_sequence_tokens": context_budget["max_sequence_tokens"],
                "completion_reserve_tokens": context_budget[
                    "completion_reserve_tokens"
                ],
                "source_cld_prompt_tokens": context_record["conditions"]["CLD"][
                    "chat_prompt_tokens"
                ],
                "eos_tokens": 1,
                "overflow_policy": BUDGET_POLICY,
            },
            "source_context_record_sha256": context_record["record_sha256"],
        }
        record["record_sha256"] = recompute_evidence_record_sha256(record)
        records.append(record)
    if not records:
        raise TrainingEvidenceError("train context manifest is empty")
    if (
        context_meta.get("record_count") != len(records)
        or context_meta.get("records_sha256") != sha256_file(context_manifest_path)
        or context_meta.get("ordered_query_ids_sha256")
        != canonical_sha256(ordered_context_query_ids)
    ):
        raise TrainingEvidenceError(
            "train context meta does not bind its complete manifest frame"
        )
    if ordered_context_query_ids != ordered_partition_ids:
        raise TrainingEvidenceError(
            "train context manifest order differs from the immutable train partition"
        )
    return records


def _records_content_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    semantic = [
        {
            "query_id": record["query"]["id"],
            "record_sha256": recompute_evidence_record_sha256(record),
        }
        for record in records
    ]
    return canonical_sha256(semantic)


def _partition_record_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fit_ids = [
        record["query"]["id"]
        for record in records
        if record["query"].get("partition") == "fit"
    ]
    calibration_ids = [
        record["query"]["id"]
        for record in records
        if record["query"].get("partition") == "calibration"
    ]
    if (
        not fit_ids
        or not calibration_ids
        or len(fit_ids) + len(calibration_ids) != len(records)
    ):
        raise TrainingEvidenceError(
            "training evidence partition is empty or incomplete"
        )
    return {
        "partition_counts": {
            "fit": len(fit_ids),
            "calibration": len(calibration_ids),
            "total": len(records),
        },
        "fit_query_ids_sha256": canonical_sha256(fit_ids),
        "calibration_query_ids_sha256": canonical_sha256(calibration_ids),
    }


def _partition_bundle_from_validated_target(
    *,
    partition_target: Path,
    partition_dependency: Mapping[str, Any],
    partition_report: Mapping[str, Any],
    data_dependency: Mapping[str, Any],
    workspace_root: Path,
) -> TrainPartitionBundle:
    """Materialize the already deep-validated partition without a locator file."""

    data_target = resolve_dependency_target(data_dependency, workspace_root)
    train_records_value = load_json(data_target / "train.json")
    partition_meta = load_json(partition_target / "partition.meta.json")
    partition_rows = load_jsonl(partition_target / "partition.jsonl")
    if (
        not isinstance(train_records_value, list)
        or not all(isinstance(record, Mapping) for record in train_records_value)
        or not isinstance(partition_meta, Mapping)
    ):
        raise TrainingEvidenceError(
            "validated train partition has an invalid materialized source frame"
        )
    train_records = tuple(dict(record) for record in train_records_value)
    by_id = {str(record.get("id")): record for record in train_records}
    fit_ids = tuple(str(value) for value in partition_report.get("fit_ids", []))
    calibration_ids = tuple(
        str(value) for value in partition_report.get("calibration_ids", [])
    )
    ordered_ids = tuple(str(row.get("query_id")) for row in partition_rows)
    if (
        len(by_id) != len(train_records)
        or ordered_ids != tuple(by_id)
        or set(fit_ids).union(calibration_ids) != set(ordered_ids)
        or set(fit_ids).intersection(calibration_ids)
    ):
        raise TrainingEvidenceError(
            "validated train partition cannot reconstruct its complete train frame"
        )
    return TrainPartitionBundle(
        locator=dict(partition_dependency),
        target=partition_target,
        partition_dependency=dict(partition_dependency),
        data_dependency=dict(data_dependency),
        data_target=data_target,
        meta=dict(partition_meta),
        rows=tuple(dict(row) for row in partition_rows),
        train_records=train_records,
        fit_ids=fit_ids,
        calibration_ids=calibration_ids,
        fit_records=tuple(by_id[query_id] for query_id in fit_ids),
        calibration_records=tuple(
            by_id[query_id] for query_id in calibration_ids
        ),
    )


def recompute_training_evidence_build_id(meta: Mapping[str, Any]) -> str:
    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping):
        raise TrainingEvidenceError("training evidence meta lacks id_inputs")
    return "tevd-" + canonical_sha256(id_inputs)


def _validate_target_under_lease(
    target: Path,
    *,
    workspace_root: Path,
    schema_path: Path,
    require_directory_name: bool,
    tokenizer: Any,
    registered_base_model: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ensure_exact_file_set(
        target,
        {
            "config.resolved.json",
            "base_model_ref.json",
            "context_ref.json",
            "provenance.json",
            "training_evidence.meta.json",
            "training_evidence.train.jsonl",
            "train_partition_ref.json",
            "payload_manifest.json",
        },
    )
    validate_payload_manifest(target)
    config = load_json(target / "config.resolved.json")
    meta = load_json(target / "training_evidence.meta.json")
    provenance = load_json(target / "provenance.json")
    context_dependency = load_json(target / "context_ref.json")
    base_model_dependency = load_json(target / "base_model_ref.json")
    partition_dependency = load_json(target / "train_partition_ref.json")
    records = load_jsonl(target / "training_evidence.train.jsonl")
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("replay_policy") != EVIDENCE_REPLAY_POLICY
    ):
        raise TrainingEvidenceError("invalid training evidence resolved config")
    if not isinstance(meta, dict):
        raise TrainingEvidenceError("training evidence meta is not an object")
    validate_json_schema(meta, schema_path)
    if (
        meta.get("replay_policy") != EVIDENCE_REPLAY_POLICY
        or meta.get("id_inputs", {}).get("replay_policy")
        != EVIDENCE_REPLAY_POLICY
        or meta.get("id_inputs", {}).get("config_sha256")
        != canonical_sha256(config)
    ):
        raise TrainingEvidenceError(
            "training evidence does not bind the exact context replay contract"
        )
    validate_dependency_ref(context_dependency, expected_kind=CONTEXT_ARTIFACT_KIND)
    validate_dependency_ref(
        base_model_dependency, expected_kind=BASE_MODEL_ARTIFACT_KIND
    )
    validate_dependency_ref(
        partition_dependency, expected_kind=TRAIN_PARTITION_ARTIFACT_KIND
    )
    if context_dependency != meta.get("context_dependency"):
        raise TrainingEvidenceError("context dependency mismatch between payload files")
    if (
        base_model_dependency != meta.get("base_model_dependency")
        or base_model_dependency
        != meta.get("id_inputs", {}).get("base_model_dependency")
    ):
        raise TrainingEvidenceError("base model dependency mismatch between payload files")
    context_target = resolve_dependency_target(context_dependency, workspace_root)
    context_config = load_json(context_target / "config.resolved.json")
    if not isinstance(context_config, Mapping):
        raise TrainingEvidenceError("scientific train context config is invalid")
    # ``None`` means load the exact frozen local tokenizer.  It never means
    # skip context validation or evidence replay, including when invoked by a
    # training plan, schedule, registry, or runtime validator.
    tokenizer_object = tokenizer
    base_model = registered_base_model
    context_revision = context_config.get("budget", {}).get("tokenizer_revision")
    model_revision = base_model.get("tokenizer_contract", {}).get(
        "tokenizer_revision"
    )
    if not isinstance(context_revision, str) or context_revision != model_revision:
        raise TrainingEvidenceError(
            "train context tokenizer revision differs from registered base model"
        )
    if getattr(tokenizer_object, "eos_token_id", None) is None:
        raise TrainingEvidenceError("replay tokenizer must define eos_token_id")
    try:
        from data.build_context_manifest import validate_context_target

        validated_context_meta = validate_context_target(
            context_target,
            tokenizer=tokenizer_object,
            workspace_root=workspace_root,
        )
    except Exception as exc:
        raise TrainingEvidenceError(
            f"scientific train context validation failed: {exc}"
        ) from exc
    train_context_policy, train_context_policy_sha256 = context_policy_snapshot(
        context_target,
        expected_split="train",
        require_scientific=True,
        context_meta=validated_context_meta,
    )
    frozen_tokenizer_revision = _context_revision(validated_context_meta)
    expected_config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "split": "train",
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_set_policy": EVIDENCE_SET_POLICY,
        "budget_policy": BUDGET_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": frozen_tokenizer_revision,
        "base_model_dependency": base_model_dependency,
        "train_context_policy_sha256": train_context_policy_sha256,
    }
    if config != expected_config:
        raise TrainingEvidenceError(
            "training evidence resolved config differs from frozen base/context policy"
        )
    expected_policy_fields = {
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_set_policy": EVIDENCE_SET_POLICY,
        "budget_policy": BUDGET_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": frozen_tokenizer_revision,
    }
    id_inputs = meta.get("id_inputs", {})
    if any(
        meta.get(key) != value or id_inputs.get(key) != value
        for key, value in expected_policy_fields.items()
    ):
        raise TrainingEvidenceError(
            "training evidence meta/ID policy fields differ from frozen context"
        )
    if meta.get("train_context_policy") != train_context_policy:
        raise TrainingEvidenceError("training evidence context policy snapshot mismatch")
    if meta.get("train_context_policy_sha256") != train_context_policy_sha256:
        raise TrainingEvidenceError("training evidence context policy hash mismatch")
    if (
        meta.get("id_inputs", {}).get("train_context_policy")
        != train_context_policy
        or meta.get("id_inputs", {}).get("train_context_policy_sha256")
        != train_context_policy_sha256
    ):
        raise TrainingEvidenceError(
            "training evidence ID inputs do not bind train context policy"
        )
    source_data_dependency = _read_dependency(context_target, "data_ref.json", "data")
    source_partition_dependency = _read_dependency(
        context_target, "train_partition_ref.json", TRAIN_PARTITION_ARTIFACT_KIND
    )
    source_lexicon_dependency = _read_dependency(context_target, "lexicon_ref.json", "lexicon")
    if meta.get("data_dependency") != source_data_dependency:
        raise TrainingEvidenceError("training evidence data dependency mismatch")
    if meta.get("lexicon_dependency") != source_lexicon_dependency:
        raise TrainingEvidenceError("training evidence lexicon dependency mismatch")
    if meta.get("train_partition_dependency") != partition_dependency:
        raise TrainingEvidenceError("training evidence partition dependency mismatch")
    if source_partition_dependency != partition_dependency:
        raise TrainingEvidenceError(
            "training evidence partition differs from train context lineage"
        )
    if meta.get("id_inputs", {}).get("train_partition_dependency") != partition_dependency:
        raise TrainingEvidenceError(
            "training evidence ID inputs do not bind the train partition"
        )
    partition_target = resolve_dependency_target(partition_dependency, workspace_root)
    partition_report = validate_train_partition_target(
        partition_target,
        workspace_root=workspace_root,
        expected_data_dependency=source_data_dependency,
    )
    if partition_report.get("partition_dependency") != partition_dependency:
        raise TrainingEvidenceError(
            "training evidence partition dependency cannot be reproduced"
        )
    partition = _partition_bundle_from_validated_target(
        partition_target=partition_target,
        partition_dependency=partition_dependency,
        partition_report=partition_report,
        data_dependency=source_data_dependency,
        workspace_root=workspace_root,
    )
    resolve_dependency_target(source_lexicon_dependency, workspace_root)
    build_id = recompute_training_evidence_build_id(meta)
    if meta.get("training_evidence_build_id") != build_id:
        raise TrainingEvidenceError("training evidence build ID is not reproducible")
    if require_directory_name and target.name != build_id:
        raise TrainingEvidenceError("training evidence directory name does not match build ID")
    if len(records) != meta.get("record_count"):
        raise TrainingEvidenceError("training evidence record count mismatch")
    query_ids: list[str] = []
    for record in records:
        _validate_record(record, schema_path=schema_path)
        if record.get("training_evidence_build_id") != build_id:
            raise TrainingEvidenceError("training evidence record has the wrong build ID")
        query_ids.append(record["query"]["id"])
    if query_ids != sorted(query_ids, key=int) or len(query_ids) != len(set(query_ids)):
        raise TrainingEvidenceError("training evidence records are not uniquely numeric-sorted")
    if meta.get("ordered_query_ids_sha256") != canonical_sha256(query_ids):
        raise TrainingEvidenceError("training evidence ordered query ID hash mismatch")
    if meta.get("records_content_sha256") != _records_content_sha256(records):
        raise TrainingEvidenceError("training evidence semantic record hash mismatch")
    replayed_records = _build_records(
        context_target=context_target,
        context_meta=validated_context_meta,
        context_config=context_config,
        tokenizer=tokenizer_object,
        schema_path=schema_path,
        partition=partition,
    )
    for replayed in replayed_records:
        replayed["training_evidence_build_id"] = build_id
        _validate_record(replayed, schema_path=schema_path)
    if records != replayed_records:
        raise TrainingEvidenceError(
            "training evidence differs from exact context manifest/catalog replay"
        )
    partition_summary = _partition_record_summary(records)
    if any(meta.get(key) != value for key, value in partition_summary.items()):
        raise TrainingEvidenceError("training evidence partition summary mismatch")
    if any(
        meta.get("id_inputs", {}).get(key) != value
        for key, value in partition_summary.items()
    ):
        raise TrainingEvidenceError(
            "training evidence ID inputs do not bind the partition summary"
        )
    partition_by_id = {
        row["query_id"]: row
        for row in load_jsonl(partition_target / "partition.jsonl")
    }
    if [record["query"]["id"] for record in records] != list(partition_by_id):
        raise TrainingEvidenceError(
            "training evidence query registry differs from the train partition"
        )
    calibration_content_hashes = {
        row["content_sha256"]
        for row in partition_by_id.values()
        if row["partition"] == "calibration"
    }
    for record in records:
        query_id = record["query"]["id"]
        if record["query"]["partition"] != partition_by_id[query_id]["partition"]:
            raise TrainingEvidenceError(
                f"training evidence partition label mismatch: {query_id}"
            )
        for item in record["demo_evidence"]["items"]:
            source = partition_by_id.get(item["source_record_id"])
            if (
                source is None
                or source["partition"] != "fit"
                or source["content_sha256"] in calibration_content_hashes
            ):
                raise TrainingEvidenceError(
                    f"training evidence demo is not information-isolated: {query_id}"
                )
    if meta.get("records_sha256") != sha256_file(
        target / "training_evidence.train.jsonl"
    ):
        raise TrainingEvidenceError("training evidence JSONL hash mismatch")
    if provenance != {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "training_evidence_build_id": build_id,
        "id_inputs": meta["id_inputs"],
        "base_model_dependency": base_model_dependency,
        "train_partition_dependency": partition_dependency,
        "train_context_policy": train_context_policy,
        "train_context_policy_sha256": train_context_policy_sha256,
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_builder_code_sha256": meta["evidence_builder_code_sha256"],
    }:
        raise TrainingEvidenceError("training evidence provenance mismatch")
    return meta, records


def _validate_target(
    target: Path,
    *,
    workspace_root: Path,
    schema_path: Path,
    require_directory_name: bool,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate and replay the complete evidence target inside one source lease."""

    base_model_dependency = _read_dependency(
        target, "base_model_ref.json", BASE_MODEL_ARTIFACT_KIND
    )
    with _registered_evidence_tokenizer(
        base_model_dependency,
        workspace_root=workspace_root,
        tokenizer=tokenizer,
    ) as resolved:
        tokenizer_object, resolved_dependency, base_model = resolved
        if resolved_dependency != base_model_dependency:
            raise TrainingEvidenceError(
                "registered base model dependency changed before evidence replay"
            )
        return _validate_target_under_lease(
            target,
            workspace_root=workspace_root,
            schema_path=schema_path,
            require_directory_name=require_directory_name,
            tokenizer=tokenizer_object,
            registered_base_model=base_model,
        )


def _build_training_evidence_under_lease(
    *,
    context_ref: str | Path,
    train_partition_ref: str | Path,
    base_model_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    tokenizer: Any,
    registered_base_model: Mapping[str, Any],
    registered_base_dependency: Mapping[str, Any],
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Freeze a self-contained, train-only evidence set target."""

    root = Path(workspace_root).resolve()
    schema = Path(schema_path)
    context_locator, context_target = resolve_locator_ref(
        context_ref, expected_kind=CONTEXT_ARTIFACT_KIND
    )
    context_dependency = portable_dependency(context_locator, context_target, root)
    base_locator, base_target = resolve_locator_ref(
        base_model_ref, expected_kind=BASE_MODEL_ARTIFACT_KIND
    )
    base_model_dependency = portable_dependency(base_locator, base_target, root)
    if base_model_dependency != registered_base_dependency:
        raise TrainingEvidenceError(
            "base model ref changed before training-evidence build"
        )
    data_dependency = _read_dependency(context_target, "data_ref.json", "data")
    context_partition_dependency = _read_dependency(
        context_target, "train_partition_ref.json", TRAIN_PARTITION_ARTIFACT_KIND
    )
    lexicon_dependency = _read_dependency(context_target, "lexicon_ref.json", "lexicon")
    resolve_dependency_target(data_dependency, root)
    resolve_dependency_target(lexicon_dependency, root)
    partition = load_train_partition(
        train_partition_ref,
        workspace_root=root,
        expected_data_dependency=data_dependency,
    )
    partition_dependency = partition.partition_dependency
    if context_partition_dependency != partition_dependency:
        raise TrainingEvidenceError(
            "explicit train partition ref differs from train context lineage"
        )
    context_meta = load_json(context_target / "context_manifest.train.meta.json")
    context_config = load_json(context_target / "config.resolved.json")
    if not isinstance(context_meta, dict) or context_meta.get("schema_version") != "stage1-context-manifest/v1":
        raise TrainingEvidenceError("invalid train context meta")
    if context_meta.get("context_build_id") != context_locator["artifact_id"]:
        raise TrainingEvidenceError("train context meta has the wrong context_build_id")
    if not isinstance(context_config, dict):
        raise TrainingEvidenceError("invalid context resolved config")
    tokenizer_object = tokenizer
    base_model = registered_base_model
    context_revision = context_config.get("budget", {}).get("tokenizer_revision")
    model_revision = base_model.get("tokenizer_contract", {}).get(
        "tokenizer_revision"
    )
    if not isinstance(context_revision, str) or context_revision != model_revision:
        raise TrainingEvidenceError(
            "train context tokenizer revision differs from registered base model"
        )
    if getattr(tokenizer_object, "eos_token_id", None) is None:
        raise TrainingEvidenceError("tokenizer must define eos_token_id")
    try:
        from data.build_context_manifest import validate_context_target

        validated_context_meta = validate_context_target(
            context_target,
            tokenizer=tokenizer_object,
            workspace_root=root,
        )
    except Exception as exc:
        raise TrainingEvidenceError(
            f"scientific train context validation failed: {exc}"
        ) from exc
    train_context_policy, train_context_policy_sha256 = context_policy_snapshot(
        context_target,
        expected_split="train",
        require_scientific=True,
        context_meta=validated_context_meta,
    )
    records = _build_records(
        context_target=context_target,
        context_meta=validated_context_meta,
        context_config=context_config,
        tokenizer=tokenizer_object,
        schema_path=schema,
        partition=partition,
    )
    tokenizer_revision = _context_revision(context_meta)
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "split": "train",
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_set_policy": EVIDENCE_SET_POLICY,
        "budget_policy": BUDGET_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": tokenizer_revision,
        "base_model_dependency": base_model_dependency,
        "train_context_policy_sha256": train_context_policy_sha256,
    }
    builder_hash = _builder_code_sha256()
    records_content_hash = _records_content_sha256(records)
    partition_summary = _partition_record_summary(records)
    id_inputs = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "context_dependency": context_dependency,
        "base_model_dependency": base_model_dependency,
        "data_dependency": data_dependency,
        "lexicon_dependency": lexicon_dependency,
        "train_partition_dependency": partition_dependency,
        "train_context_policy": train_context_policy,
        "train_context_policy_sha256": train_context_policy_sha256,
        "records_content_sha256": records_content_hash,
        **partition_summary,
        "config_sha256": canonical_sha256(config),
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_set_policy": EVIDENCE_SET_POLICY,
        "budget_policy": BUDGET_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": tokenizer_revision,
        "evidence_builder_code_sha256": builder_hash,
    }
    build_id = "tevd-" + canonical_sha256(id_inputs)
    for record in records:
        record["training_evidence_build_id"] = build_id
        _validate_record(record, schema_path=schema)
    # Canonical file order is numeric query ID, not record hash.
    records_bytes = b"".join(
        canonical_json_bytes(record) + b"\n"
        for record in records
    )
    meta = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "training_evidence_build_id": build_id,
        "split": "train",
        "context_dependency": context_dependency,
        "base_model_dependency": base_model_dependency,
        "data_dependency": data_dependency,
        "lexicon_dependency": lexicon_dependency,
        "train_partition_dependency": partition_dependency,
        "train_context_policy": train_context_policy,
        "train_context_policy_sha256": train_context_policy_sha256,
        "record_count": len(records),
        **partition_summary,
        "ordered_query_ids_sha256": canonical_sha256(
            [record["query"]["id"] for record in records]
        ),
        "records_content_sha256": records_content_hash,
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
        "replay_policy": EVIDENCE_REPLAY_POLICY,
        "evidence_set_policy": EVIDENCE_SET_POLICY,
        "budget_policy": BUDGET_POLICY,
        "renderer_revision": RENDERER_REVISION,
        "tokenizer_revision": tokenizer_revision,
        "evidence_builder_code_sha256": builder_hash,
        "id_inputs": id_inputs,
    }
    validate_json_schema(meta, schema)
    output_parent = (
        Path(target_root).resolve()
        if target_root is not None
        else context_target.parent.parent / "training_evidence"
    )
    target = output_parent / build_id
    staging = new_staging_directory(output_parent, build_id)
    try:
        write_canonical_json(staging / "config.resolved.json", config)
        write_canonical_json(
            staging / "base_model_ref.json", base_model_dependency
        )
        write_canonical_json(staging / "context_ref.json", context_dependency)
        write_canonical_json(
            staging / "train_partition_ref.json", partition_dependency
        )
        write_canonical_json(
            staging / "provenance.json",
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "training_evidence_build_id": build_id,
                "id_inputs": id_inputs,
                "base_model_dependency": base_model_dependency,
                "train_partition_dependency": partition_dependency,
                "train_context_policy": train_context_policy,
                "train_context_policy_sha256": train_context_policy_sha256,
                "replay_policy": EVIDENCE_REPLAY_POLICY,
                "evidence_builder_code_sha256": builder_hash,
            },
        )
        write_canonical_json(staging / "training_evidence.meta.json", meta)
        write_bytes_atomic(staging / "training_evidence.train.jsonl", records_bytes)
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    try:
        payload_hash = finalize_target_atomic(
            staging,
            target,
            validate_staging=lambda path: _validate_target(
                path,
                workspace_root=root,
                schema_path=schema,
                require_directory_name=False,
                tokenizer=tokenizer_object,
            ),
        )
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    return write_locator_ref(
        write_ref,
        artifact_kind=ARTIFACT_KIND,
        artifact_id=build_id,
        target=target,
        payload_manifest_sha256=payload_hash,
    )


def build_training_evidence(
    *,
    context_ref: str | Path,
    train_partition_ref: str | Path,
    base_model_ref: str | Path,
    write_ref: str | Path,
    workspace_root: str | Path = REPOSITORY_ROOT,
    target_root: str | Path | None = None,
    tokenizer: Any | None = None,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    """Freeze train-only evidence under one registered tokenizer/base lease."""

    root = Path(workspace_root).resolve()
    base_locator, base_target = resolve_locator_ref(
        base_model_ref, expected_kind=BASE_MODEL_ARTIFACT_KIND
    )
    base_dependency = portable_dependency(base_locator, base_target, root)
    with _registered_evidence_tokenizer(
        base_dependency,
        workspace_root=root,
        tokenizer=tokenizer,
    ) as resolved:
        tokenizer_object, resolved_dependency, base_model = resolved
        return _build_training_evidence_under_lease(
            context_ref=context_ref,
            train_partition_ref=train_partition_ref,
            base_model_ref=base_model_ref,
            write_ref=write_ref,
            workspace_root=root,
            target_root=target_root,
            tokenizer=tokenizer_object,
            registered_base_model=base_model,
            registered_base_dependency=resolved_dependency,
            schema_path=schema_path,
        )


def load_training_evidence(
    training_evidence_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    locator, target = resolve_locator_ref(
        training_evidence_ref, expected_kind=ARTIFACT_KIND
    )
    meta, records = _validate_target(
        target,
        workspace_root=Path(workspace_root).resolve(),
        schema_path=Path(schema_path),
        require_directory_name=True,
        tokenizer=tokenizer,
    )
    if locator["artifact_id"] != meta["training_evidence_build_id"]:
        raise TrainingEvidenceError("training evidence locator ID mismatch")
    if locator["payload_manifest_sha256"] != validate_payload_manifest(target):
        raise TrainingEvidenceError("training evidence locator payload mismatch")
    return locator, target, meta, records


def validate_training_evidence_target(
    target: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate an already-resolved evidence target without a locator file."""

    return _validate_target(
        Path(target),
        workspace_root=Path(workspace_root).resolve(),
        schema_path=Path(schema_path),
        require_directory_name=True,
        tokenizer=tokenizer,
    )


def validate_training_evidence(
    training_evidence_ref: str | Path,
    *,
    workspace_root: str | Path = REPOSITORY_ROOT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    locator, _target, meta, records = load_training_evidence(
        training_evidence_ref,
        workspace_root=workspace_root,
        schema_path=schema_path,
        tokenizer=tokenizer,
    )
    return {
        "schema_version": "stage1-training-evidence-validation-report/v1",
        "valid": True,
        "training_evidence_build_id": meta["training_evidence_build_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "record_count": len(records),
        "ordered_query_ids_sha256": meta["ordered_query_ids_sha256"],
    }


__all__ = [
    "ARTIFACT_KIND",
    "BUDGET_POLICY",
    "CONTEXT_POLICY_SCHEMA_VERSION",
    "EVIDENCE_REPLAY_POLICY",
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_SET_POLICY",
    "RECORD_SCHEMA_VERSION",
    "TrainingEvidenceError",
    "build_training_evidence",
    "context_policy_snapshot",
    "frozen_evidence_tokenizer_lease",
    "load_frozen_evidence_tokenizer",
    "load_training_evidence",
    "recompute_evidence_record_sha256",
    "recompute_training_evidence_build_id",
    "validate_training_evidence",
    "validate_training_evidence_target",
]
