"""Immutable Stage-1 context artifact builder over frozen retrieval evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import prompt as prompt_module
from data import context_selector as context_selector_module
from data import context_manifest as context_manifest_module
from data import retrieval_bundle as retrieval_bundle_module
from build_lex.train_only import (
    FORMAL_CAPTURE_FILES,
    PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION,
    RAW_AUDIT_FILES,
    TrainOnlyLexiconError,
    validate_lexicon_target,
)
from data.context_manifest import (
    CONDITIONS,
    canonical_sha256,
    finalize_context_budget,
    render_condition_item,
    text_sha256,
    validate_context_record,
)
from data.retrieval_bundle import (
    LEXICON_EVIDENCE_KIND,
    LEXICON_EVIDENCE_RENDER_POLICY,
    LEXICON_TASK_LABEL_VISIBILITY,
    RETRIEVAL_BUNDLE_VERSION,
    RetrievalBundleError,
    build_demo_catalog,
    build_lexicon_catalog,
    build_query_pool,
    cosine_score_matrices,
    prepare_context_bundle_from_scores,
    render_lexicon_evidence_block,
    retrieval_code_sha256s,
)
from data.stage1_data import Stage1DataError, validate_data_target
from data.train_partition import (
    TrainPartitionError,
    validate_train_partition_target,
)
from model.stage1_registry import (
    ModelRegistryError,
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
    verified_model_source_lease,
)
from utils.quadruple import canonicalize_quadruples, serialize_quadruples


PREPARED_BUNDLE_SCHEMA = "stage1-prepared-context-bundle/v1"
CONTEXT_META_SCHEMA = "stage1-context-manifest/v1"
CONTEXT_PROVENANCE_SCHEMA = "stage1-context-provenance/v1"
LOCATOR_SCHEMA = "stage1-locator-ref/v1"
DEPENDENCY_SCHEMA = "stage1-dependency-ref/v1"
PAYLOAD_SCHEMA = "stage1-payload-manifest/v1"
FROZEN_CONTEXT_POLICY_SCHEMA = "stage1-frozen-context-policy-ref/v1"
RETRIEVAL_PROVENANCE_SCHEMA = "stage1-retrieval-provenance/v1"
FORMAL_SCORER_BACKEND = "sentence-transformers-cosine/v1"
RUNTIME_SOURCE_IDENTITY_SCHEMA = "stage1-context-runtime-source-identity/v1"
RENDERING_IDENTITY_SCHEMA = "stage1-context-rendering-identity/v1"
TOKENIZER_CONSTRUCTOR_POLICY = {
    "backend": "transformers-auto-tokenizer/v1",
    "local_files_only": True,
    "trust_remote_code": False,
}
SCORER_CONSTRUCTOR_POLICY = {
    "backend": FORMAL_SCORER_BACKEND,
    "local_files_only": True,
    "trust_remote_code": False,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class ContextBuildError(ValueError):
    pass


FormalScoreReplayer = Callable[..., tuple[np.ndarray, np.ndarray]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextBuildError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                raise ContextBuildError(f"blank JSONL line at {path}:{ordinal}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContextBuildError(f"non-object JSONL row at {path}:{ordinal}")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextBuildError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_source_sha256(module: Any, *, label: str) -> str:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ContextBuildError(f"cannot resolve {label} source file")
    path = Path(source)
    if path.suffix in {".pyc", ".pyo"}:
        path = path.with_suffix(".py")
    if not path.is_file() or path.is_symlink():
        raise ContextBuildError(f"{label} source must be a regular file")
    return sha256_file(path)


def _resolved_rendering_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind resolved prompt bytes and every helper that renders them.

    Config constant names alone are not an identity: editing a prompt constant
    or one of the renderer helpers must invalidate an existing context.
    """

    rendering = config.get("rendering")
    if not isinstance(rendering, Mapping) or rendering.get("thinking_mode") is not False:
        raise ContextBuildError("Stage 1 rendering config must disable thinking")
    if (
        rendering.get("lexicon_evidence_policy")
        != LEXICON_EVIDENCE_RENDER_POLICY
        or rendering.get("task_label_visibility")
        != LEXICON_TASK_LABEL_VISIBILITY
    ):
        raise ContextBuildError(
            "Stage 1 rendering must freeze category-free terminology evidence"
        )
    prompt_keys = ("system_prompt", "user_prompt", "example_prompt")
    prompts: dict[str, Any] = {}
    for key in prompt_keys:
        name = rendering.get(key)
        value = getattr(prompt_module, str(name), None)
        if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
            raise ContextBuildError(f"cannot resolve prompt constant {name!r}")
        prompts[key] = {
            "constant_name": name,
            "utf8_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    return {
        "schema_version": RENDERING_IDENTITY_SCHEMA,
        "thinking_mode": False,
        "lexicon_evidence_policy": LEXICON_EVIDENCE_RENDER_POLICY,
        "task_label_visibility": LEXICON_TASK_LABEL_VISIBILITY,
        "prompts": prompts,
        "prompt_module_code_sha256": _module_source_sha256(
            prompt_module, label="prompt module"
        ),
        "context_renderer_code_sha256": _module_source_sha256(
            context_manifest_module, label="context renderer"
        ),
        "retrieval_renderer_code_sha256": _module_source_sha256(
            retrieval_bundle_module, label="retrieval renderer"
        ),
    }


def _full_source_inventory(
    path: str | Path, *, workspace_root: str | Path, label: str
) -> dict[str, Any]:
    try:
        return inventory_regular_file_tree(
            path,
            workspace_root=workspace_root,
            label=label,
            inventory_policy="all-regular-files/v1",
        )
    except ModelRegistryError as exc:
        raise ContextBuildError(str(exc)) from exc


def _runtime_source_identity(
    *,
    config: Mapping[str, Any],
    retrieval_provenance: Mapping[str, Any],
    workspace_root: str | Path,
) -> tuple[dict[str, Any], ResolvedModelSourceContract]:
    """Resolve full tokenizer/scorer trees from frozen logical paths."""

    budget = config.get("budget")
    if not isinstance(budget, Mapping):
        raise ContextBuildError("context config lacks a tokenizer budget")
    revision = budget.get("tokenizer_revision")
    if not isinstance(revision, str) or not revision:
        raise ContextBuildError("context budget must freeze tokenizer_revision")
    tokenizer_path = _resolve_logical_workspace_path(
        budget.get("tokenizer_path"),
        workspace_root,
        label="tokenizer path",
    )
    tokenizer_inventory = _full_source_inventory(
        tokenizer_path,
        workspace_root=workspace_root,
        label="context tokenizer",
    )
    scorer = retrieval_provenance.get("scorer")
    if not isinstance(scorer, Mapping):
        raise ContextBuildError("formal retrieval provenance lacks its scorer")
    model_path = resolve_embedding_model_path(
        scorer.get("logical_model_path"), workspace_root
    )
    scorer_inventory = _full_source_inventory(
        model_path,
        workspace_root=workspace_root,
        label="context embedding scorer",
    )
    declared_tree_hash = scorer.get("model_file_tree_sha256")
    if declared_tree_hash != embedding_model_file_tree_sha256(model_path):
        raise ContextBuildError("formal embedding model file tree changed")
    identity = {
        "schema_version": RUNTIME_SOURCE_IDENTITY_SCHEMA,
        "tokenizer": {
            "declared_revision": revision,
            "inventory": tokenizer_inventory,
            "constructor_policy": dict(TOKENIZER_CONSTRUCTOR_POLICY),
        },
        "scorer": {
            "backend": scorer.get("backend"),
            "inventory": scorer_inventory,
            "constructor_policy": dict(SCORER_CONSTRUCTOR_POLICY),
        },
    }
    contract = ResolvedModelSourceContract(
        workspace_root=Path(workspace_root).resolve(),
        checkpoint_inventory=scorer_inventory,
        tokenizer_inventory=tokenizer_inventory,
        base_inventory=scorer_inventory,
    )
    return identity, contract


def _construct_formal_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ContextBuildError(
            "transformers is required for formal context construction"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise ContextBuildError(f"formal tokenizer construction failed: {exc}") from exc


def embedding_model_file_tree_sha256(path: str | Path) -> str:
    """Hash an immutable embedding-model tree using its portable file frame."""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ContextBuildError("embedding model path must be a real directory")
    rows: list[list[Any]] = []
    for child in sorted(
        root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()
    ):
        if child.is_symlink():
            raise ContextBuildError("embedding model tree cannot contain symlinks")
        if child.is_file():
            rows.append(
                [
                    child.relative_to(root).as_posix(),
                    child.stat().st_size,
                    sha256_file(child),
                ]
            )
    if not rows:
        raise ContextBuildError("embedding model directory is empty")
    return hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _payload_manifest(target: Path) -> dict[str, Any]:
    files = []
    for path in sorted(target.rglob("*"), key=lambda item: item.relative_to(target).as_posix()):
        if path.is_symlink():
            raise ContextBuildError("context payload cannot contain symlinks")
        if not path.is_file() or path.name == "payload_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(target).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": PAYLOAD_SCHEMA, "files": files}


def _verify_payload(target: Path) -> str:
    stored = _read_json(target / "payload_manifest.json")
    if stored != _payload_manifest(target):
        raise ContextBuildError("context payload manifest mismatch")
    return sha256_file(target / "payload_manifest.json")


def _atomic_locator(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _portable_dependency(locator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": DEPENDENCY_SCHEMA,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
    }


def _validate_locator(value: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if value.get("schema_version") != LOCATOR_SCHEMA or value.get("artifact_kind") != kind:
        raise ContextBuildError(f"invalid {kind} locator")
    target = Path(str(value.get("target_path", "")))
    if (
        not target.is_absolute()
        or target.is_symlink()
        or not target.is_dir()
        or target.name != value.get("artifact_id")
    ):
        raise ContextBuildError(f"unresolvable {kind} locator")
    if value.get("payload_manifest_sha256") != _verify_payload(target):
        raise ContextBuildError(f"{kind} locator payload hash mismatch")
    return _portable_dependency(value)


def _catalog(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    key = "lexicon_id" if kind == "lexicon" else "demo_id"
    prefix = "lex:v2:" if kind == "lexicon" else "demo:v1:"
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        identifier = row.get(key)
        if not isinstance(identifier, str) or not identifier.startswith(prefix):
            raise ContextBuildError(f"invalid {kind} ID")
        block = row.get("rendered_block")
        if not isinstance(block, str) or not block:
            raise ContextBuildError(f"{kind} row lacks rendered_block")
        row["rendered_block_sha256"] = text_sha256(block)
        if row.get("source_split", row.get("split", "train")) != "train" or row.get("train_only") is False:
            raise ContextBuildError(f"{kind} catalog is not train-only")
        row["source_split"] = "train"
        row["train_only"] = True
        result.append(row)
    if kind == "lexicon":
        result.sort(key=lambda row: (row.get("content_sha256", ""), row[key]))
    else:
        result.sort(key=lambda row: row[key])
    if len({row[key] for row in result}) != len(result):
        raise ContextBuildError(f"duplicate {kind} ID")
    return result


def _validate_category_free_terminology_catalog(rows: Any) -> None:
    if not isinstance(rows, list) or not rows:
        raise ContextBuildError("formal lexicon catalog must be a non-empty array")
    visible_hashes: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ContextBuildError("formal lexicon catalog contains a non-object row")
        if "category" in raw or "categories" in raw:
            raise ContextBuildError(
                "formal terminology catalog contains a task-category field"
            )
        variants = raw.get("variants", []) or []
        if not isinstance(variants, list) or any(
            not isinstance(value, str) for value in variants
        ):
            raise ContextBuildError("formal lexicon variants must be strings")
        for field in ("usage_notes", "ambiguity_notes"):
            if not isinstance(raw.get(field, ""), str):
                raise ContextBuildError(
                    f"formal terminology {field} must be text"
                )
        if (
            raw.get("render_policy") != LEXICON_EVIDENCE_RENDER_POLICY
            or raw.get("task_label_visibility")
            != LEXICON_TASK_LABEL_VISIBILITY
            or raw.get("evidence_kind") != LEXICON_EVIDENCE_KIND
        ):
            raise ContextBuildError(
                "formal lexicon catalog exposes an unfrozen evidence policy"
            )
        expected = render_lexicon_evidence_block(
            term=str(raw.get("term", "")),
            definition=str(raw.get("definition", "")),
            variants=variants,
            usage_notes=str(raw.get("usage_notes", "")),
            ambiguity_notes=str(raw.get("ambiguity_notes", "")),
        )
        if raw.get("rendered_block") != expected:
            raise ContextBuildError(
                "formal lexicon block is not canonical category-free terminology evidence"
            )
        visible_hash = text_sha256(expected)
        if raw.get("content_sha256") != visible_hash:
            raise ContextBuildError("formal lexicon evidence content hash mismatch")
        if visible_hash in visible_hashes:
            raise ContextBuildError(
                "formal lexicon catalog duplicates model-visible evidence"
            )
        visible_hashes.add(visible_hash)


def _query_pool(
    rows: Any,
    *,
    source_split: str,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ContextBuildError(f"{source_split} query pool must be a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ContextBuildError(f"{source_split} query-pool row must be an object")
        query_id = str(raw.get("id", ""))
        content = raw.get("content")
        if not re.fullmatch(r"[1-9][0-9]*", query_id) or not isinstance(content, str) or not content:
            raise ContextBuildError(f"invalid {source_split} query-pool identity/content")
        if query_id in seen:
            raise ContextBuildError(f"duplicate {source_split} query-pool ID")
        seen.add(query_id)
        try:
            quads = canonicalize_quadruples(raw.get("quadruples"))
        except Exception as exc:
            raise ContextBuildError(
                f"invalid {source_split} query-pool gold for {query_id}: {exc}"
            ) from exc
        output = serialize_quadruples(quads)
        expected = {
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
            "content_sha256": text_sha256(content),
            "gold_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "source_split": source_split,
        }
        if dict(raw) != expected:
            raise ContextBuildError(f"non-canonical {source_split} query-pool row {query_id}")
        result.append(expected)
    return result


def _validate_query_pool_lineage(
    *,
    train_pool: Sequence[Mapping[str, Any]],
    split_pool: Sequence[Mapping[str, Any]],
    split: str,
    records: Sequence[Mapping[str, Any]],
    demo_catalog: Sequence[Mapping[str, Any]],
) -> None:
    train_by_id = {row["id"]: row for row in train_pool}
    demo_by_source = {row["source_record_id"]: row for row in demo_catalog}
    if not demo_by_source or not set(demo_by_source).issubset(train_by_id):
        raise ContextBuildError("demo catalog is not a non-empty subset of train query pool")
    for query_id in demo_by_source:
        row = train_by_id[query_id]
        demo = demo_by_source[query_id]
        if (
            row["content"] != demo.get("content")
            or row["content_sha256"] != demo.get("content_sha256")
            or row["gold_sha256"] != demo.get("gold_sha256")
        ):
            raise ContextBuildError("train query pool and demo catalog lineage mismatch")
    if split == "train" and list(split_pool) != list(train_pool):
        raise ContextBuildError("train context query pool must equal the frozen train pool")
    split_by_id = {row["id"]: row for row in split_pool}
    record_ids = [str(record.get("query", {}).get("id", "")) for record in records]
    if record_ids != [row["id"] for row in split_pool]:
        raise ContextBuildError("context record order differs from split query pool")
    for record in records:
        query = record["query"]
        frozen = split_by_id[str(query["id"])]
        if (
            query.get("content") != frozen["content"]
            or query.get("content_sha256") != frozen["content_sha256"]
            or query.get("gold_sha256") != frozen["gold_sha256"]
            or canonicalize_quadruples(query.get("gold"))
            != canonicalize_quadruples(frozen["quadruples"])
        ):
            raise ContextBuildError("context query disagrees with frozen query pool")


def _prepared_bundle_file_sha256(bundle: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(bundle) + b"\n").hexdigest()


def _replay_prepared_records(
    *,
    prepared_records: Sequence[Mapping[str, Any]],
    context_build_id: str,
    lexicon_catalog: Mapping[str, Mapping[str, Any]],
    demo_catalog: Mapping[str, Mapping[str, Any]],
    system_prompt: str,
    user_prompt_template: str,
    tokenizer: Any,
    budget: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay the sole permitted prepared-record -> frozen-context transform."""

    replayed: list[dict[str, Any]] = []
    generated_fields = {
        "context_build_id",
        "budget",
        "conditions",
        "schema_version",
        "record_sha256",
    }
    for ordinal, raw in enumerate(prepared_records):
        if not isinstance(raw, Mapping):
            raise ContextBuildError(
                f"prepared context record {ordinal} must be an object"
            )
        if generated_fields.intersection(raw):
            raise ContextBuildError(
                f"prepared context record {ordinal} contains finalized fields"
            )
        selection = raw.get("selection")
        if not isinstance(selection, Mapping):
            raise ContextBuildError(
                f"prepared context record {ordinal} lacks selection"
            )
        for kind in ("demos", "lexicons"):
            branch = selection.get(kind)
            if not isinstance(branch, Mapping):
                raise ContextBuildError(
                    f"prepared context record {ordinal} lacks {kind} selection"
                )
            if {"prompt_order_final", "budget_dropped_ids"}.intersection(
                branch
            ):
                raise ContextBuildError(
                    f"prepared context record {ordinal} contains finalized selection fields"
                )
        base = copy.deepcopy(dict(raw))
        base["context_build_id"] = context_build_id
        replayed.append(
            finalize_context_budget(
                base,
                lexicon_catalog=lexicon_catalog,
                demo_catalog=demo_catalog,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                tokenizer=tokenizer,
                max_sequence_tokens=int(budget["max_sequence_tokens"]),
                completion_reserve_tokens=int(
                    budget["completion_reserve_tokens"]
                ),
                trim_policy=str(budget["trim_policy"]),
            )
        )
    return replayed


def _resolved_prompts(config: Mapping[str, Any]) -> tuple[str, str]:
    rendering = config.get("rendering")
    if not isinstance(rendering, Mapping) or rendering.get("thinking_mode") is not False:
        raise ContextBuildError("Stage 1 rendering config must disable thinking")
    names = (rendering.get("system_prompt"), rendering.get("user_prompt"))
    values = []
    for name in names:
        value = getattr(prompt_module, str(name), None)
        if not isinstance(value, str) or not value:
            raise ContextBuildError(f"cannot resolve prompt constant {name!r}")
        values.append(value)
    return values[0], values[1]


def _validate_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    declared = bundle.get("bundle_sha256")
    if not isinstance(declared, str) or not SHA256_RE.fullmatch(declared):
        raise ContextBuildError("prepared bundle lacks a valid bundle_sha256")
    unhashed = dict(bundle)
    unhashed.pop("bundle_sha256", None)
    if canonical_sha256(unhashed) != declared:
        raise ContextBuildError("prepared bundle_sha256 does not match bundle content")
    return declared


def _assert_bundle_has_no_locator_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, inner in value.items():
            key = str(raw_key).lower()
            if (
                key == "target_path"
                or key.endswith("_locator")
                or "locator_sha256" in key
            ):
                raise ContextBuildError(
                    "prepared context bundle must not persist runtime locator refs"
                )
            _assert_bundle_has_no_locator_material(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _assert_bundle_has_no_locator_material(inner)


def _resolve_logical_workspace_path(
    logical_path: Any,
    workspace_root: str | Path,
    *,
    label: str,
) -> Path:
    logical = Path(logical_path) if isinstance(logical_path, str) else None
    if (
        not isinstance(logical_path, str)
        or not logical_path
        or logical is None
        or logical.as_posix() in {"", "."}
        or logical_path != logical.as_posix()
        or logical.is_absolute()
        or ".." in logical.parts
    ):
        raise ContextBuildError(f"{label} is not a portable workspace path")
    root = Path(workspace_root).resolve()
    lexical = root / logical
    cursor = root
    for part in logical.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ContextBuildError(f"{label} cannot traverse symlinks")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContextBuildError(f"{label} escapes workspace_root") from exc
    return resolved


def resolve_embedding_model_path(
    logical_path: str, workspace_root: str | Path
) -> Path:
    """Resolve a canonical, symlink-free embedding model workspace path."""

    return _resolve_logical_workspace_path(
        logical_path, workspace_root, label="embedding model path"
    )


def _validate_formal_retrieval_provenance(
    provenance: Any,
    *,
    split: str,
    workspace_root: str | Path,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ContextBuildError("formal bundle lacks retrieval provenance")
    provenance_keys = {
        "schema_version",
        "policy_version",
        "train_only_demo_pool",
        "train_only_lexicon_pool",
        "fit_only_demo_pool",
        "calibration_demo_excluded",
        "all_fit_demo_pool_relevance_complete",
        "fit_demo_record_count",
        "fit_demo_ids_sha256",
        "calibration_record_count",
        "calibration_ids_sha256",
        "train_partition_dependency",
        "saw_dev_test_labels_during_pool_build",
        "saw_model_predictions",
        "score_matrix",
        "retrieval_config_sha256",
        "retrieval_builder_code_sha256",
        "selector_code_sha256",
        "scorer",
    }
    if set(provenance) != provenance_keys:
        raise ContextBuildError("formal retrieval provenance fields are not canonical")
    required_flags = {
        "train_only_demo_pool": True,
        "train_only_lexicon_pool": True,
        "fit_only_demo_pool": True,
        "calibration_demo_excluded": True,
        "saw_dev_test_labels_during_pool_build": False,
        "saw_model_predictions": False,
    }
    if provenance.get("schema_version") != RETRIEVAL_PROVENANCE_SCHEMA:
        raise ContextBuildError("formal retrieval provenance schema is invalid")
    if provenance.get("policy_version") != RETRIEVAL_BUNDLE_VERSION:
        raise ContextBuildError("formal retrieval policy is not the frozen cosine policy")
    for key, expected in required_flags.items():
        if provenance.get(key) is not expected:
            raise ContextBuildError(f"formal retrieval provenance fails {key}")
    expected_complete = split in {"dev", "test"}
    if provenance.get("all_fit_demo_pool_relevance_complete") is not expected_complete:
        raise ContextBuildError(
            "all-fit-demo-pool relevance completeness disagrees with split policy"
        )
    for key in ("fit_demo_record_count", "calibration_record_count"):
        value = provenance.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContextBuildError(f"formal retrieval provenance has invalid {key}")
    for key in ("fit_demo_ids_sha256", "calibration_ids_sha256"):
        if not isinstance(provenance.get(key), str) or not SHA256_RE.fullmatch(
            str(provenance.get(key))
        ):
            raise ContextBuildError(f"formal retrieval provenance has invalid {key}")
    partition_dependency = provenance.get("train_partition_dependency")
    if (
        not isinstance(partition_dependency, Mapping)
        or set(partition_dependency)
        != {
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
            "logical_repo_path",
        }
        or partition_dependency.get("schema_version") != DEPENDENCY_SCHEMA
        or partition_dependency.get("artifact_kind") != "train-partition"
        or not isinstance(partition_dependency.get("artifact_id"), str)
        or not str(partition_dependency.get("artifact_id")).startswith("tpart-")
        or not isinstance(partition_dependency.get("payload_manifest_sha256"), str)
        or not SHA256_RE.fullmatch(
            str(partition_dependency.get("payload_manifest_sha256"))
        )
        or not isinstance(partition_dependency.get("logical_repo_path"), str)
        or not partition_dependency.get("logical_repo_path")
    ):
        raise ContextBuildError("formal retrieval provenance has invalid partition dependency")
    score_matrix = provenance.get("score_matrix")
    score_matrix_keys = {
        "demo_sha256",
        "lexicon_sha256",
        "evidence_sha256",
        "query_texts_sha256",
        "demo_texts_sha256",
        "lexicon_texts_sha256",
    }
    if not isinstance(score_matrix, Mapping) or set(score_matrix) != score_matrix_keys:
        raise ContextBuildError("formal retrieval score-matrix provenance is invalid")
    if any(
        not isinstance(score_matrix.get(key), str)
        or not SHA256_RE.fullmatch(str(score_matrix.get(key)))
        for key in score_matrix_keys
    ):
        raise ContextBuildError("formal retrieval score-matrix hashes are invalid")
    try:
        expected_code_hashes = retrieval_code_sha256s()
    except RetrievalBundleError as exc:
        raise ContextBuildError(str(exc)) from exc
    if any(
        provenance.get(key) != value
        for key, value in expected_code_hashes.items()
    ):
        raise ContextBuildError("formal retrieval implementation code changed")
    if (
        not isinstance(provenance.get("retrieval_config_sha256"), str)
        or not SHA256_RE.fullmatch(str(provenance.get("retrieval_config_sha256")))
    ):
        raise ContextBuildError("formal retrieval config hash is invalid")

    scorer = provenance.get("scorer")
    scorer_keys = {
        "backend",
        "logical_model_path",
        "model_file_tree_sha256",
        "device_class",
        "batch_size",
    }
    if not isinstance(scorer, Mapping) or set(scorer) != scorer_keys:
        raise ContextBuildError("formal retrieval scorer provenance is not canonical")
    if scorer.get("backend") != FORMAL_SCORER_BACKEND:
        raise ContextBuildError(
            f"formal retrieval scorer must use {FORMAL_SCORER_BACKEND}"
        )
    if scorer.get("device_class") not in {"cpu", "cuda"}:
        raise ContextBuildError("formal retrieval scorer device class is invalid")
    batch_size = scorer.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ContextBuildError("formal retrieval scorer batch_size is invalid")
    declared_tree_hash = scorer.get("model_file_tree_sha256")
    if not isinstance(declared_tree_hash, str) or not SHA256_RE.fullmatch(
        declared_tree_hash
    ):
        raise ContextBuildError("formal retrieval scorer model tree hash is invalid")
    model_path = resolve_embedding_model_path(
        scorer.get("logical_model_path"),
        workspace_root,
    )
    actual_tree_hash = embedding_model_file_tree_sha256(model_path)
    if actual_tree_hash != declared_tree_hash:
        raise ContextBuildError("formal embedding model file tree changed")
    return {
        "model_path": model_path,
        "model_file_tree_sha256": actual_tree_hash,
    }


def _validate_prepared_bundle(
    bundle: Mapping[str, Any],
    *,
    formal: bool,
    sealed: bool = False,
    workspace_root: str | Path = REPOSITORY_ROOT,
) -> None:
    _validate_bundle_sha256(bundle)
    _assert_bundle_has_no_locator_material(bundle)
    if bundle.get("schema_version") != PREPARED_BUNDLE_SCHEMA:
        raise ContextBuildError("unsupported prepared context bundle")
    if bundle.get("split") not in {"train", "dev", "test"}:
        raise ContextBuildError("prepared context split is invalid")
    records = bundle.get("records")
    if not isinstance(records, list) or not records:
        raise ContextBuildError("prepared context records must be non-empty")
    if formal:
        formal_bundle_fields = {
            "schema_version",
            "split",
            "demo_catalog",
            "lexicon_catalog",
            "train_query_pool",
            "query_pool",
            "score_evidence",
            "records",
            "retrieval_provenance",
            "data_dependency",
            "train_partition_dependency",
            "lexicon_dependency",
            "bundle_sha256",
        }
        if set(bundle) != formal_bundle_fields:
            raise ContextBuildError(
                "formal prepared bundle fields are not canonical"
            )
        _validate_category_free_terminology_catalog(bundle.get("lexicon_catalog"))
        dependencies = {
            "data": bundle.get("data_dependency"),
            "train-partition": bundle.get("train_partition_dependency"),
            "lexicon": bundle.get("lexicon_dependency"),
        }
        for kind, dependency in dependencies.items():
            if not isinstance(dependency, Mapping):
                raise ContextBuildError(
                    f"formal bundle lacks portable {kind} dependency"
                )
            _resolve_workspace_dependency(
                dependency, workspace_root, kind=kind
            )
        _validate_formal_retrieval_provenance(
            bundle.get("retrieval_provenance"),
            split=str(bundle.get("split")),
            workspace_root=workspace_root,
        )
        if bundle.get("train_partition_dependency") != bundle.get(
            "retrieval_provenance", {}
        ).get("train_partition_dependency"):
            raise ContextBuildError(
                "prepared bundle partition dependency disagrees with retrieval provenance"
            )


def _workspace_dependency(
    locator: Mapping[str, Any], target: Path, workspace_root: str | Path
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    try:
        logical = target.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContextBuildError("sealed dependency must live below workspace_root") from exc
    resolved = _resolve_logical_workspace_path(
        logical, root, label="dependency target path"
    )
    if resolved != target.resolve():
        raise ContextBuildError("dependency target path is not canonical")
    return {
        "schema_version": DEPENDENCY_SCHEMA,
        "artifact_kind": locator["artifact_kind"],
        "artifact_id": locator["artifact_id"],
        "payload_manifest_sha256": locator["payload_manifest_sha256"],
        "logical_repo_path": logical,
    }


def _resolve_workspace_dependency(
    dependency: Mapping[str, Any], workspace_root: str | Path, *, kind: str
) -> Path:
    required = {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "payload_manifest_sha256",
        "logical_repo_path",
    }
    if (
        set(dependency) != required
        or dependency.get("schema_version") != DEPENDENCY_SCHEMA
        or dependency.get("artifact_kind") != kind
    ):
        raise ContextBuildError(f"invalid portable {kind} dependency")
    root = Path(workspace_root).resolve()
    target = _resolve_logical_workspace_path(
        dependency.get("logical_repo_path"),
        root,
        label="portable dependency path",
    )
    if (
        not target.is_dir()
        or target.name != dependency.get("artifact_id")
        or _verify_payload(target) != dependency.get("payload_manifest_sha256")
    ):
        raise ContextBuildError(f"portable {kind} dependency cannot be verified")
    return target


def _source_class_order(config: Mapping[str, Any]) -> list[str]:
    retrieval = config.get("retrieval")
    order = retrieval.get("source_class_order") if isinstance(retrieval, Mapping) else None
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(value, str) or not value for value in order)
        or len(order) != len(set(order))
    ):
        raise ContextBuildError("context config lacks canonical retrieval.source_class_order")
    return list(order)


def _validate_formal_retrieval_config(
    config: Mapping[str, Any], provenance: Mapping[str, Any]
) -> Mapping[str, Any]:
    retrieval = config.get("retrieval")
    expected_fields = {
        "demo_policy",
        "demo_top_k",
        "candidate_multiplier",
        "source_class_order",
        "class_weights",
        "weights_reverse",
        "allocated_class_top_k",
        "similarity_threshold",
        "threshold_comparator",
        "source_rank_policy",
        "selection_score_policy",
        "lexicon_selection_policy",
        "lex_exact_top_k",
        "lex_semantic_top_k",
        "score_round_digits",
        "seed",
    }
    if not isinstance(retrieval, Mapping) or set(retrieval) != expected_fields:
        raise ContextBuildError("formal retrieval config fields are not canonical")
    policy_values = {
        "demo_policy": context_selector_module.SELECTION_POLICY,
        "source_rank_policy": context_selector_module.SOURCE_RANK_POLICY,
        "selection_score_policy": context_selector_module.SELECTION_SCORE_POLICY,
        "lexicon_selection_policy": context_selector_module.LEXICON_SELECTION_POLICY,
        "threshold_comparator": ">=",
        "score_round_digits": 8,
        "weights_reverse": False,
    }
    if any(retrieval.get(key) != value for key, value in policy_values.items()):
        raise ContextBuildError("formal retrieval config policy is not implemented")
    order = _source_class_order(config)
    quota = retrieval.get("allocated_class_top_k")
    weights = retrieval.get("class_weights")
    if (
        not isinstance(quota, Mapping)
        or set(quota) != set(order)
        or any(
            isinstance(quota.get(key), bool)
            or not isinstance(quota.get(key), int)
            or int(quota[key]) < 0
            for key in order
        )
        or not isinstance(weights, Mapping)
        or set(weights) != set(order)
        or any(
            isinstance(weights.get(key), bool)
            or not isinstance(weights.get(key), (int, float))
            or not math.isfinite(float(weights[key]))
            or float(weights[key]) < 0
            for key in order
        )
    ):
        raise ContextBuildError("formal retrieval quota/weight frame is invalid")
    demo_top_k = retrieval.get("demo_top_k")
    candidate_multiplier = retrieval.get("candidate_multiplier")
    exact_top_k = retrieval.get("lex_exact_top_k")
    semantic_top_k = retrieval.get("lex_semantic_top_k")
    if (
        isinstance(demo_top_k, bool)
        or not isinstance(demo_top_k, int)
        or demo_top_k != sum(int(quota[key]) for key in order)
        or isinstance(candidate_multiplier, bool)
        or not isinstance(candidate_multiplier, int)
        or candidate_multiplier <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (exact_top_k, semantic_top_k)
        )
    ):
        raise ContextBuildError("formal retrieval top-k policy is invalid")
    threshold = retrieval.get("similarity_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or isinstance(retrieval.get("seed"), bool)
        or not isinstance(retrieval.get("seed"), int)
    ):
        raise ContextBuildError("formal retrieval numeric policy is invalid")
    if provenance.get("retrieval_config_sha256") != canonical_sha256(
        dict(retrieval)
    ):
        raise ContextBuildError("formal retrieval config hash mismatch")
    return retrieval


def _formal_source_snapshot(
    *,
    data_target: Path,
    partition_target: Path,
    lexicon_target: Path,
    split: str,
    source_class_order: Sequence[str],
    retrieval_provenance: Mapping[str, Any],
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Deep-validate upstream artifacts and reconstruct the only valid catalogs."""

    try:
        guard_before = {
            "data_payload_manifest_sha256": _verify_payload(data_target),
            "train_partition_payload_manifest_sha256": _verify_payload(
                partition_target
            ),
            "lexicon_payload_manifest_sha256": _verify_payload(lexicon_target),
        }
        scorer = _validate_formal_retrieval_provenance(
            retrieval_provenance,
            split=split,
            workspace_root=workspace_root,
        )
        guard_before["model_file_tree_sha256"] = scorer[
            "model_file_tree_sha256"
        ]
        data_report = validate_data_target(data_target)
        data_dependency = {
            "schema_version": DEPENDENCY_SCHEMA,
            "artifact_kind": "data",
            "artifact_id": data_report["data_build_id"],
            "payload_manifest_sha256": data_report["payload_manifest_sha256"],
        }
        partition_report = validate_train_partition_target(
            partition_target,
            workspace_root=workspace_root,
            expected_data_dependency=data_dependency,
        )
        full_partition_dependency = partition_report.get("partition_dependency")
        if not isinstance(full_partition_dependency, Mapping):
            raise ContextBuildError(
                "validated train partition lacks its dependency"
            )
        partition_dependency = dict(full_partition_dependency)
        lexicon_manifest = validate_lexicon_target(
            lexicon_target, workspace_root=workspace_root
        )
        if lexicon_manifest.get("source_mode") != "data_ref+train_partition":
            raise ContextBuildError(
                "formal context requires a formally authorized data-ref lexicon"
            )
        if lexicon_manifest.get("data_build_id") != data_report.get("data_build_id"):
            raise ContextBuildError("formal lexicon is bound to a different data artifact")
        if lexicon_manifest.get("train_partition_dependency") != partition_dependency:
            raise ContextBuildError(
                "formal lexicon is bound to a different train partition"
            )
        if (
            lexicon_manifest.get("schema_version")
            == PILOT_GATED_TERMINOLOGY_MANIFEST_VERSION
        ):
            expected_lexicon_files = {
                "data_ref.json",
                "train_partition_ref.json",
                "lexicon.json",
                "manifest.json",
                "provenance.json",
                "lifecycle.json",
                "pipeline_config.json",
                "span_pilot_decision.json",
                "full_span_audit_decision.json",
                "resolution_gate.json",
                "merged_resolution.json",
                "human_resolution.json",
                "payload_manifest.json",
            }
        else:
            expected_lexicon_files = {
                "data_ref.json",
                "train_partition_ref.json",
                "lexicon.json",
                "manifest.json",
                "provenance.json",
                "payload_manifest.json",
                *RAW_AUDIT_FILES,
                *FORMAL_CAPTURE_FILES.values(),
            }
        actual_lexicon_files = {
            path.relative_to(lexicon_target).as_posix()
            for path in lexicon_target.rglob("*")
            if path.is_file()
        }
        if actual_lexicon_files != expected_lexicon_files:
            raise ContextBuildError("formal lexicon target file set is not canonical")
        lexicon_data_ref = _read_json(lexicon_target / "data_ref.json")
        lexicon_partition_ref = _read_json(
            lexicon_target / "train_partition_ref.json"
        )
        data_portable_dependency = _workspace_dependency(
            {
                "artifact_kind": "data",
                "artifact_id": data_report["data_build_id"],
                "payload_manifest_sha256": data_report[
                    "payload_manifest_sha256"
                ],
            },
            data_target,
            workspace_root,
        )
        if (
            not isinstance(lexicon_data_ref, Mapping)
            or dict(lexicon_data_ref) != data_portable_dependency
        ):
            raise ContextBuildError(
                "formal lexicon does not bind the exact selected data payload"
            )
        if (
            not isinstance(lexicon_partition_ref, Mapping)
            or dict(lexicon_partition_ref) != partition_dependency
        ):
            raise ContextBuildError(
                "formal lexicon does not bind the exact selected train partition"
            )

        train_records = _read_json(data_target / "train.json")
        split_records = _read_json(data_target / f"{split}.json")
        lexicon_document = _read_json(lexicon_target / "lexicon.json")
        terms = (
            lexicon_document.get("terms")
            if isinstance(lexicon_document, Mapping)
            else None
        )
        if not isinstance(train_records, list) or not isinstance(split_records, list):
            raise ContextBuildError("validated data target split payload is not an array")
        if not isinstance(terms, list):
            raise ContextBuildError("validated lexicon target terms payload is not an array")
        fit_ids = partition_report.get("fit_ids")
        calibration_ids = partition_report.get("calibration_ids")
        if (
            not isinstance(fit_ids, list)
            or not isinstance(calibration_ids, list)
        ):
            raise ContextBuildError("validated train partition lacks reconstructed pools")
        train_by_id = {str(record.get("id")): record for record in train_records}
        fit_records = [train_by_id[str(identifier)] for identifier in fit_ids]
        demo_catalog, _ = build_demo_catalog(
            fit_records, source_class_order=source_class_order
        )
        expected = {
            "train_query_pool": _query_pool(
                build_query_pool(train_records, source_split="train"),
                source_split="train",
            ),
            "split_query_pool": _query_pool(
                build_query_pool(split_records, source_split=split),
                source_split=split,
            ),
            "demo_catalog": _catalog(demo_catalog, "demo"),
            "lexicon_catalog": _catalog(build_lexicon_catalog(terms), "lexicon"),
            "fit_ids": list(fit_ids),
            "calibration_ids": list(calibration_ids),
            "partition_dependency": partition_dependency,
            "train_records": train_records,
            "fit_records": fit_records,
            "split_records": split_records,
            "lexicon_terms": terms,
            "scorer": scorer,
        }
        guard_after = {
            "data_payload_manifest_sha256": _verify_payload(data_target),
            "train_partition_payload_manifest_sha256": _verify_payload(
                partition_target
            ),
            "lexicon_payload_manifest_sha256": _verify_payload(lexicon_target),
            "model_file_tree_sha256": embedding_model_file_tree_sha256(
                scorer["model_path"]
            ),
        }
    except (
        Stage1DataError,
        TrainOnlyLexiconError,
        TrainPartitionError,
        RetrievalBundleError,
    ) as exc:
        raise ContextBuildError(f"formal context upstream validation failed: {exc}") from exc
    if guard_after != guard_before:
        raise ContextBuildError("formal context dependency changed during validation")
    expected["guard"] = guard_before
    return expected


def _assert_source_snapshot_matches_bundle(
    snapshot: Mapping[str, Any],
    *,
    train_query_pool: Sequence[Mapping[str, Any]],
    split_query_pool: Sequence[Mapping[str, Any]],
    demo_catalog: Sequence[Mapping[str, Any]],
    lexicon_catalog: Sequence[Mapping[str, Any]],
    retrieval_provenance: Mapping[str, Any],
) -> None:
    comparisons = (
        ("train query pool", train_query_pool, snapshot["train_query_pool"]),
        ("split query pool", split_query_pool, snapshot["split_query_pool"]),
        ("demo catalog", demo_catalog, snapshot["demo_catalog"]),
        ("lexicon catalog", lexicon_catalog, snapshot["lexicon_catalog"]),
    )
    for label, actual, expected in comparisons:
        if list(actual) != list(expected):
            raise ContextBuildError(
                f"formal prepared {label} differs from validated upstream artifacts"
            )
    if (
        retrieval_provenance.get("train_partition_dependency")
        != snapshot["partition_dependency"]
        or retrieval_provenance.get("fit_demo_record_count")
        != len(snapshot["fit_ids"])
        or retrieval_provenance.get("fit_demo_ids_sha256")
        != canonical_sha256(snapshot["fit_ids"])
        or retrieval_provenance.get("calibration_record_count")
        != len(snapshot["calibration_ids"])
        or retrieval_provenance.get("calibration_ids_sha256")
        != canonical_sha256(snapshot["calibration_ids"])
    ):
        raise ContextBuildError(
            "formal retrieval partition provenance differs from validated partition"
        )


def _default_formal_score_replayer(
    *,
    model_path: Path,
    device_class: str,
    batch_size: int,
    train_texts: Sequence[str],
    query_texts: Sequence[str],
    lexicon_texts: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the frozen cosine scorer from the local immutable model."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ContextBuildError(
            "sentence-transformers is required for formal retrieval replay"
        ) from exc
    device = "cuda:0" if device_class == "cuda" else "cpu"
    try:
        model = SentenceTransformer(
            str(model_path),
            device=device,
            local_files_only=True,
            trust_remote_code=False,
        )
        return cosine_score_matrices(
            model=model,
            train_texts=train_texts,
            query_texts=query_texts,
            lexicon_texts=lexicon_texts,
            batch_size=batch_size,
        )
    except RetrievalBundleError:
        raise
    except Exception as exc:
        raise ContextBuildError(
            f"formal embedding scorer replay failed: {exc}"
        ) from exc


def _memoized_score_replayer(
    replayer: FormalScoreReplayer,
) -> FormalScoreReplayer:
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def replay(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        key = canonical_sha256(
            {
                "model_path": str(kwargs["model_path"]),
                "device_class": kwargs["device_class"],
                "batch_size": kwargs["batch_size"],
                "train_texts": [text_sha256(value) for value in kwargs["train_texts"]],
                "query_texts": [text_sha256(value) for value in kwargs["query_texts"]],
                "lexicon_texts": [
                    text_sha256(value) for value in kwargs["lexicon_texts"]
                ],
            }
        )
        if key not in cache:
            values = replayer(**kwargs)
            if not isinstance(values, tuple) or len(values) != 2:
                raise ContextBuildError(
                    "formal score replayer must return demo/lexicon matrices"
                )
            cache[key] = values
        return cache[key]

    return replay


def _replay_formal_retrieval_bundle(
    *,
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    score_replayer: FormalScoreReplayer,
) -> None:
    """Prove score bytes and every selector trace from immutable inputs."""

    provenance = bundle.get("retrieval_provenance")
    if not isinstance(provenance, Mapping):  # pragma: no cover - validated first
        raise ContextBuildError("formal retrieval provenance is missing")
    retrieval_config = _validate_formal_retrieval_config(config, provenance)
    scorer = provenance.get("scorer")
    if not isinstance(scorer, Mapping):  # pragma: no cover - validated first
        raise ContextBuildError("formal scorer provenance is missing")
    train_texts = [str(row["content"]) for row in snapshot["fit_records"]]
    query_texts = [str(row["content"]) for row in snapshot["split_records"]]
    lexicon_texts = [
        str(row["rendered_block"]) for row in snapshot["lexicon_catalog"]
    ]
    try:
        demo_scores, lexicon_scores = score_replayer(
            model_path=snapshot["scorer"]["model_path"],
            device_class=str(scorer["device_class"]),
            batch_size=int(scorer["batch_size"]),
            train_texts=train_texts,
            query_texts=query_texts,
            lexicon_texts=lexicon_texts,
        )
        replayed = prepare_context_bundle_from_scores(
            train_records=snapshot["train_records"],
            query_records=snapshot["split_records"],
            lexicon_terms=snapshot["lexicon_terms"],
            demo_scores=demo_scores,
            lexicon_scores=lexicon_scores,
            split=str(bundle["split"]),
            retrieval_config=retrieval_config,
            data_dependency=bundle["data_dependency"],
            lexicon_dependency=bundle["lexicon_dependency"],
            scorer_provenance=scorer,
            fit_demo_records=snapshot["fit_records"],
            calibration_ids=snapshot["calibration_ids"],
            train_partition_dependency=bundle["train_partition_dependency"],
            score_evidence=bundle.get("score_evidence"),
            expected_bundle=bundle,
        )
    except RetrievalBundleError as exc:
        if "score evidence differs" in str(exc):
            raise ContextBuildError(
                "formal score evidence differs from frozen scorer replay"
            ) from exc
        if "exact selector replay" in str(exc):
            raise ContextBuildError(
                "formal prepared records/selector differ from exact score replay"
            ) from exc
        raise ContextBuildError(f"formal retrieval replay failed: {exc}") from exc
    if replayed.get("records") != bundle.get("records"):
        raise ContextBuildError(
            "formal prepared records/selector differ from exact score replay"
        )
    if replayed != dict(bundle):
        raise ContextBuildError(
            "formal prepared bundle differs from exact score/selector replay"
        )


def _sealed_source_targets(
    lineage: Mapping[str, Any], workspace_root: str | Path
) -> tuple[Path, Path, Path]:
    data_target = _resolve_workspace_dependency(
        lineage["data_dependency"], workspace_root, kind="data"
    )
    frozen_target = _resolve_workspace_dependency(
        lineage["frozen_dev_context_dependency"], workspace_root, kind="context"
    )
    frozen_meta_paths = list(frozen_target.glob("context_manifest.*.meta.json"))
    if len(frozen_meta_paths) != 1:
        raise ContextBuildError("frozen dev context has no unique meta payload")
    frozen_meta = _read_json(frozen_meta_paths[0])
    frozen_inputs = frozen_meta.get("id_inputs") if isinstance(frozen_meta, Mapping) else None
    source_refs = (
        frozen_inputs.get("source_dependency_refs")
        if isinstance(frozen_inputs, Mapping)
        else None
    )
    if (
        not isinstance(source_refs, Mapping)
        or not isinstance(source_refs.get("lexicon"), Mapping)
        or not isinstance(source_refs.get("train_partition"), Mapping)
    ):
        raise ContextBuildError(
            "frozen dev context lacks its partition/lexicon source refs"
        )
    lexicon_ref = source_refs["lexicon"]
    compact_lexicon = {
        key: lexicon_ref[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "artifact_id",
            "payload_manifest_sha256",
        )
    }
    if compact_lexicon != lineage.get("lexicon_dependency"):
        raise ContextBuildError("sealed lexicon source ref differs from frozen lineage")
    lexicon_target = _resolve_workspace_dependency(
        lexicon_ref, workspace_root, kind="lexicon"
    )
    partition_ref = source_refs["train_partition"]
    if partition_ref != lineage.get("train_partition_dependency"):
        raise ContextBuildError(
            "sealed train-partition source ref differs from frozen lineage"
        )
    partition_target = _resolve_workspace_dependency(
        partition_ref, workspace_root, kind="train-partition"
    )
    return data_target, partition_target, lexicon_target


def _retrieval_policy_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projection = dict(value)
    projection.pop("score_matrix", None)
    return projection


def _build_prepared_context_artifact_with_tokenizer(
    *,
    prepared_bundle: str | Path | Mapping[str, Any],
    config: str | Path | Mapping[str, Any],
    tokenizer: Any,
    write_ref: str | Path | None,
    formal: bool,
    data_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    lexicon_ref: str | Path | None = None,
    target_root: str | Path | None = None,
    sealed_lineage: Mapping[str, Any] | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
    runtime_source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze precomputed one-pass retrieval evidence into all four conditions."""

    raw_bundle = _read_json(prepared_bundle) if not isinstance(prepared_bundle, Mapping) else dict(prepared_bundle)
    raw_config = _read_json(config) if not isinstance(config, Mapping) else dict(config)
    sealed = sealed_lineage is not None
    score_replayer: FormalScoreReplayer | None = None
    if formal:
        score_replayer = _memoized_score_replayer(
            formal_score_replayer or _default_formal_score_replayer
        )
    _validate_prepared_bundle(
        raw_bundle,
        formal=formal,
        sealed=sealed,
        workspace_root=workspace_root,
    )
    if raw_bundle.get("split") == "test" and not sealed:
        raise ContextBuildError("test context can only be created through seal-test")
    if sealed and (not formal or raw_bundle.get("split") != "test"):
        raise ContextBuildError("sealed context build must be a formal test artifact")
    explicit_data_locator: Mapping[str, Any] | None = None
    explicit_partition_locator: Mapping[str, Any] | None = None
    explicit_lexicon_locator: Mapping[str, Any] | None = None
    if formal and not sealed:
        if data_ref is None or train_partition_ref is None or lexicon_ref is None:
            raise ContextBuildError(
                "formal context build requires explicit data/partition/lexicon refs"
            )
        candidates = {
            "data": _read_json(data_ref),
            "train-partition": _read_json(train_partition_ref),
            "lexicon": _read_json(lexicon_ref),
        }
        if any(not isinstance(value, Mapping) for value in candidates.values()):
            raise ContextBuildError("formal runtime refs must contain objects")
        explicit_data_locator = candidates["data"]
        explicit_partition_locator = candidates["train-partition"]
        explicit_lexicon_locator = candidates["lexicon"]
    elif not formal and any(
        value is not None for value in (data_ref, train_partition_ref, lexicon_ref)
    ):
        raise ContextBuildError(
            "engineering context builds must not claim formal runtime refs"
        )
    if raw_config.get("schema_version") != "stage1-context-factorial-config/v1":
        raise ContextBuildError("unsupported context config")
    system_prompt, user_prompt = _resolved_prompts(raw_config)
    rendering_identity = _resolved_rendering_identity(raw_config)
    budget = raw_config.get("budget", {})
    revision = budget.get("tokenizer_revision")
    if not isinstance(revision, str) or not revision:
        raise ContextBuildError("context budget must freeze tokenizer_revision")

    lexicon_catalog = _catalog(raw_bundle.get("lexicon_catalog", []), "lexicon")
    demo_catalog = _catalog(raw_bundle.get("demo_catalog", []), "demo")
    train_query_pool = _query_pool(raw_bundle.get("train_query_pool"), source_split="train")
    split_query_pool = _query_pool(raw_bundle.get("query_pool"), source_split=raw_bundle["split"])
    lexicon_map = {row["lexicon_id"]: row for row in lexicon_catalog}
    demo_map = {row["demo_id"]: row for row in demo_catalog}
    dependencies: dict[str, Any] = {}
    source_dependency_refs: dict[str, Any] = {}
    data_target: Path | None = None
    partition_target: Path | None = None
    lexicon_target: Path | None = None
    if sealed:
        frozen = dict(sealed_lineage or {})
        if frozen.get("schema_version") != FROZEN_CONTEXT_POLICY_SCHEMA:
            raise ContextBuildError("invalid frozen context policy lineage")
        dependencies["data"] = dict(frozen["data_dependency"])
        dependencies["train_partition"] = dict(
            frozen["train_partition_dependency"]
        )
        dependencies["lexicon"] = dict(frozen["lexicon_dependency"])
        data_target, partition_target, lexicon_target = _sealed_source_targets(
            frozen, workspace_root
        )
    elif formal:
        dependencies["data"] = _validate_locator(explicit_data_locator, "data")
        _validate_locator(explicit_partition_locator, "train-partition")
        dependencies["lexicon"] = _validate_locator(
            explicit_lexicon_locator, "lexicon"
        )
        data_target = Path(str(explicit_data_locator["target_path"]))
        partition_target = Path(
            str(explicit_partition_locator["target_path"])
        )
        lexicon_target = Path(str(explicit_lexicon_locator["target_path"]))
        source_dependency_refs["data"] = _workspace_dependency(
            explicit_data_locator,
            data_target,
            workspace_root,
        )
        source_dependency_refs["lexicon"] = _workspace_dependency(
            explicit_lexicon_locator,
            lexicon_target,
            workspace_root,
        )
        source_dependency_refs["train_partition"] = _workspace_dependency(
            explicit_partition_locator,
            partition_target,
            workspace_root,
        )
        dependencies["train_partition"] = dict(
            source_dependency_refs["train_partition"]
        )
        prepared_dependencies = {
            "data": raw_bundle["data_dependency"],
            "train_partition": raw_bundle["train_partition_dependency"],
            "lexicon": raw_bundle["lexicon_dependency"],
        }
        if any(
            prepared_dependencies[key] != source_dependency_refs[key]
            for key in prepared_dependencies
        ):
            raise ContextBuildError(
                "explicit runtime refs differ from prepared portable dependencies"
            )
    formal_snapshot: dict[str, Any] | None = None
    if formal:
        if data_target is None or partition_target is None or lexicon_target is None:
            raise ContextBuildError("formal context source targets cannot be resolved")
        formal_snapshot = _formal_source_snapshot(
            data_target=data_target,
            partition_target=partition_target,
            lexicon_target=lexicon_target,
            split=str(raw_bundle["split"]),
            source_class_order=_source_class_order(raw_config),
            retrieval_provenance=raw_bundle["retrieval_provenance"],
            workspace_root=workspace_root,
        )
        _assert_source_snapshot_matches_bundle(
            formal_snapshot,
            train_query_pool=train_query_pool,
            split_query_pool=split_query_pool,
            demo_catalog=demo_catalog,
            lexicon_catalog=lexicon_catalog,
            retrieval_provenance=raw_bundle["retrieval_provenance"],
        )
        if score_replayer is None:  # pragma: no cover - formal invariant
            raise ContextBuildError("formal score replayer is unavailable")
        _replay_formal_retrieval_bundle(
            bundle=raw_bundle,
            config=raw_config,
            snapshot=formal_snapshot,
            score_replayer=score_replayer,
        )
    id_inputs = {
        "schema_version": "stage1-context/v1",
        "artifact_kind": "test-context" if raw_bundle["split"] == "test" else "context",
        "split": raw_bundle["split"],
        "formal": formal,
        "dependencies": dependencies,
        "bundle_sha256": raw_bundle["bundle_sha256"],
        "prepared_bundle_sha256": canonical_sha256(raw_bundle),
        "context_config_sha256": canonical_sha256(raw_config),
        "tokenizer_revision": revision,
        "rendering_identity": rendering_identity,
        "runtime_source_identity": (
            dict(runtime_source_identity)
            if formal and isinstance(runtime_source_identity, Mapping)
            else {
                "schema_version": "stage1-context-engineering-tokenizer-injection/v1",
                "scientific_eligible": False,
            }
        ),
        "builder_code_sha256": sha256_file(__file__),
    }
    if formal and not isinstance(runtime_source_identity, Mapping):
        raise ContextBuildError("formal context lacks verified runtime source identity")
    if formal:
        id_inputs["prepared_bundle_file_sha256"] = (
            _prepared_bundle_file_sha256(raw_bundle)
        )
        id_inputs["retrieval_score_inputs"] = dict(
            raw_bundle["retrieval_provenance"]["score_matrix"]
        )
    if sealed:
        id_inputs["frozen_policy_ref"] = dict(sealed_lineage or {})
        if (
            id_inputs["frozen_policy_ref"].get("rendering_identity_sha256")
            != canonical_sha256(rendering_identity)
            or id_inputs["frozen_policy_ref"].get(
                "runtime_source_identity_sha256"
            )
            != canonical_sha256(id_inputs["runtime_source_identity"])
        ):
            raise ContextBuildError(
                "sealed tokenizer/prompt/renderer/scorer identity differs from frozen dev"
            )
    elif formal:
        id_inputs["source_dependency_refs"] = source_dependency_refs
    build_id = "ctx-" + canonical_sha256(id_inputs)
    records = _replay_prepared_records(
        prepared_records=raw_bundle["records"],
        context_build_id=build_id,
        lexicon_catalog=lexicon_map,
        demo_catalog=demo_map,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
        tokenizer=tokenizer,
        budget=budget,
    )
    query_ids = [record["query"]["id"] for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ContextBuildError("context query IDs must be unique")
    _validate_query_pool_lineage(
        train_pool=train_query_pool,
        split_pool=split_query_pool,
        split=raw_bundle["split"],
        records=records,
        demo_catalog=demo_catalog,
    )

    artifact_root = raw_config.get("artifact_root")
    if target_root is None:
        if not isinstance(artifact_root, str) or not artifact_root:
            raise ContextBuildError("context build requires target_root or artifact_root")
        root = Path(__file__).resolve().parents[2] / artifact_root
        directory = "test_contexts" if raw_bundle["split"] == "test" else "contexts"
        target_parent = (root / directory).resolve()
    else:
        target_parent = Path(target_root).resolve()
    target_parent.mkdir(parents=True, exist_ok=True)
    target = target_parent / build_id
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=target_parent))
    split = raw_bundle["split"]
    try:
        _write_json(temporary / "config.resolved.json", raw_config)
        prepared_meta_document = {
            "schema_version": PREPARED_BUNDLE_SCHEMA,
            "bundle_sha256": raw_bundle["bundle_sha256"],
            "prepared_bundle_sha256": canonical_sha256(raw_bundle),
            "retrieval_provenance": raw_bundle.get("retrieval_provenance"),
        }
        if formal:
            _write_json(temporary / "prepared_bundle.json", raw_bundle)
            prepared_meta_document["prepared_bundle_file_sha256"] = sha256_file(
                temporary / "prepared_bundle.json"
            )
        _write_json(
            temporary / "prepared_bundle.meta.json", prepared_meta_document
        )
        _write_jsonl(temporary / "catalogs" / "lexicon_pool.jsonl", lexicon_catalog)
        _write_jsonl(temporary / "catalogs" / "demo_pool.train.jsonl", demo_catalog)
        _write_jsonl(temporary / "catalogs" / "query_pool.train.jsonl", train_query_pool)
        if split != "train":
            _write_jsonl(
                temporary / "catalogs" / f"query_pool.{split}.jsonl", split_query_pool
            )
        _write_jsonl(temporary / f"context_manifest.{split}.jsonl", records)
        records_hash = sha256_file(temporary / f"context_manifest.{split}.jsonl")
        meta = {
            "schema_version": CONTEXT_META_SCHEMA,
            "context_build_id": build_id,
            "artifact_kind": id_inputs["artifact_kind"],
            "split": split,
            "scientific_eligible": bool(formal),
            "record_count": len(records),
            "ordered_query_ids_sha256": canonical_sha256(query_ids),
            "records_sha256": records_hash,
            "sources": {
                "demo_pool": {
                    "split": "train",
                    "partition": "fit" if formal else "legacy-full-train",
                    "record_count": len(demo_catalog),
                },
                "lexicon_pool": {
                    "train_only_verified": True,
                    "partition": "fit" if formal else "legacy-full-train",
                    "record_count": len(lexicon_catalog),
                },
                "train_query_pool": {
                    "split": "train",
                    "record_count": len(train_query_pool),
                    "sha256": sha256_file(temporary / "catalogs" / "query_pool.train.jsonl"),
                },
                "split_query_pool": {
                    "split": split,
                    "record_count": len(split_query_pool),
                    "sha256": sha256_file(
                        temporary / "catalogs" / f"query_pool.{split}.jsonl"
                    ),
                },
            },
            "budget": {
                "tokenizer_revision": revision,
                "max_sequence_tokens": budget["max_sequence_tokens"],
                "completion_reserve_tokens": budget["completion_reserve_tokens"],
                "trim_policy": budget["trim_policy"],
            },
            "nondegenerate_masks": {
                "L_nondegenerate": [record["query"]["id"] for record in records if record["selection"]["lexicons"]["prompt_order_final"]],
                "D_nondegenerate": [record["query"]["id"] for record in records if record["selection"]["demos"]["prompt_order_final"]],
                "LD_nondegenerate": [record["query"]["id"] for record in records if record["selection"]["lexicons"]["prompt_order_final"] and record["selection"]["demos"]["prompt_order_final"]],
            },
            "rendering_identity": rendering_identity,
            "runtime_source_identity": id_inputs["runtime_source_identity"],
            "id_inputs": id_inputs,
        }
        _write_json(temporary / f"context_manifest.{split}.meta.json", meta)
        _write_json(temporary / "provenance.json", {
            "schema_version": CONTEXT_PROVENANCE_SCHEMA,
            "context_build_id": build_id,
            "dependencies": dependencies,
            "retrieval_provenance": raw_bundle.get("retrieval_provenance"),
            "saw_model_predictions": False,
            "builder_code_sha256": id_inputs["builder_code_sha256"],
            "rendering_identity": rendering_identity,
            "runtime_source_identity": id_inputs["runtime_source_identity"],
        })
        if sealed:
            _write_json(
                temporary / "data_ref.json", dict(sealed_lineage["data_dependency"])
            )
            _write_json(
                temporary / "lexicon_ref.json",
                dict(sealed_lineage["lexicon_dependency"]),
            )
            _write_json(
                temporary / "train_partition_ref.json",
                dict(sealed_lineage["train_partition_dependency"]),
            )
            _write_json(
                temporary / "frozen_policy_ref.json", dict(sealed_lineage)
            )
        elif formal:
            _write_json(temporary / "data_ref.json", source_dependency_refs["data"])
            _write_json(
                temporary / "lexicon_ref.json", source_dependency_refs["lexicon"]
            )
            _write_json(
                temporary / "train_partition_ref.json",
                source_dependency_refs["train_partition"],
            )
        for condition in CONDITIONS:
            _write_json(
                temporary / "conditions" / "runner" / condition / f"{split}.json",
                [render_condition_item(record, condition) for record in records],
            )
        if formal:
            refreshed = _formal_source_snapshot(
                data_target=data_target,
                partition_target=partition_target,
                lexicon_target=lexicon_target,
                split=split,
                source_class_order=_source_class_order(raw_config),
                retrieval_provenance=raw_bundle["retrieval_provenance"],
                workspace_root=workspace_root,
            )
            _assert_source_snapshot_matches_bundle(
                refreshed,
                train_query_pool=train_query_pool,
                split_query_pool=split_query_pool,
                demo_catalog=demo_catalog,
                lexicon_catalog=lexicon_catalog,
                retrieval_provenance=raw_bundle["retrieval_provenance"],
            )
            if refreshed["guard"] != formal_snapshot["guard"]:
                raise ContextBuildError(
                    "formal context dependency changed while building artifact"
                )
        _write_json(temporary / "payload_manifest.json", _payload_manifest(temporary))
        _validate_context_target_with_tokenizer(
            temporary,
            tokenizer=tokenizer,
            require_directory_name=False,
            workspace_root=workspace_root,
            formal_score_replayer=score_replayer,
            runtime_source_identity=runtime_source_identity,
        )
        if target.exists():
            _validate_context_target_with_tokenizer(
                target,
                tokenizer=tokenizer,
                workspace_root=workspace_root,
                formal_score_replayer=score_replayer,
                runtime_source_identity=runtime_source_identity,
            )
            if _read_json(target / "payload_manifest.json") != _read_json(temporary / "payload_manifest.json"):
                raise ContextBuildError("same context ID produced a different payload")
        else:
            os.replace(temporary, target)
        _validate_context_target_with_tokenizer(
            target,
            tokenizer=tokenizer,
            workspace_root=workspace_root,
            formal_score_replayer=score_replayer,
            runtime_source_identity=runtime_source_identity,
        )
        if formal:
            final_snapshot = _formal_source_snapshot(
                data_target=data_target,
                partition_target=partition_target,
                lexicon_target=lexicon_target,
                split=split,
                source_class_order=_source_class_order(raw_config),
                retrieval_provenance=raw_bundle["retrieval_provenance"],
                workspace_root=workspace_root,
            )
            if final_snapshot["guard"] != formal_snapshot["guard"]:
                raise ContextBuildError(
                    "formal context dependency changed before locator publication"
                )
        locator = {
            "schema_version": LOCATOR_SCHEMA,
            "artifact_kind": id_inputs["artifact_kind"],
            "artifact_id": build_id,
            "target_path": str(target.resolve()),
            "payload_manifest_sha256": sha256_file(target / "payload_manifest.json"),
        }
        if write_ref is not None:
            _atomic_locator(Path(write_ref), locator)
        return locator
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_prepared_context_artifact(
    *,
    prepared_bundle: str | Path | Mapping[str, Any],
    config: str | Path | Mapping[str, Any],
    tokenizer: Any | None = None,
    write_ref: str | Path | None,
    formal: bool,
    data_ref: str | Path | None = None,
    train_partition_ref: str | Path | None = None,
    lexicon_ref: str | Path | None = None,
    target_root: str | Path | None = None,
    sealed_lineage: Mapping[str, Any] | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
) -> dict[str, Any]:
    """Build a context under a typed, full-tree formal source lease.

    Caller-provided tokenizers and score replayers are engineering-only.  A
    scientific artifact always constructs both backends from the logical paths
    frozen in its config/provenance while the source trees remain leased.
    """

    if not formal:
        if tokenizer is None:
            raise ContextBuildError(
                "engineering context requires an explicit test-only tokenizer"
            )
        if formal_score_replayer is not None:
            raise ContextBuildError(
                "engineering context must not claim a formal score replayer"
            )
        return _build_prepared_context_artifact_with_tokenizer(
            prepared_bundle=prepared_bundle,
            config=config,
            tokenizer=tokenizer,
            write_ref=write_ref,
            formal=False,
            data_ref=data_ref,
            train_partition_ref=train_partition_ref,
            lexicon_ref=lexicon_ref,
            target_root=target_root,
            sealed_lineage=sealed_lineage,
            workspace_root=workspace_root,
            runtime_source_identity=None,
        )
    if tokenizer is not None or formal_score_replayer is not None:
        raise ContextBuildError(
            "scientific context forbids caller-injected tokenizer/score replayer"
        )
    raw_bundle = (
        _read_json(prepared_bundle)
        if not isinstance(prepared_bundle, Mapping)
        else dict(prepared_bundle)
    )
    raw_config = (
        _read_json(config) if not isinstance(config, Mapping) else dict(config)
    )
    _validate_prepared_bundle(
        raw_bundle,
        formal=True,
        sealed=sealed_lineage is not None,
        workspace_root=workspace_root,
    )
    # The source lease deliberately monitors model-path ancestors.  Create all
    # unrelated output parents before entering it so artifact publication does
    # not look like a model namespace swap.
    if target_root is not None:
        Path(target_root).resolve().mkdir(parents=True, exist_ok=True)
    else:
        artifact_root = raw_config.get("artifact_root")
        if not isinstance(artifact_root, str) or not artifact_root:
            raise ContextBuildError("context build requires target_root or artifact_root")
        directory = (
            "test_contexts"
            if raw_bundle.get("split") == "test"
            else "contexts"
        )
        (REPOSITORY_ROOT / artifact_root / directory).resolve().mkdir(
            parents=True, exist_ok=True
        )
    if write_ref is not None:
        Path(write_ref).parent.mkdir(parents=True, exist_ok=True)
    retrieval_provenance = raw_bundle.get("retrieval_provenance")
    if not isinstance(retrieval_provenance, Mapping):
        raise ContextBuildError("formal prepared bundle lacks retrieval provenance")
    runtime_identity, contract = _runtime_source_identity(
        config=raw_config,
        retrieval_provenance=retrieval_provenance,
        workspace_root=workspace_root,
    )
    try:
        with verified_model_source_lease(
            contract, source_names=("tokenizer", "checkpoint")
        ) as paths:
            frozen_tokenizer = _construct_formal_tokenizer(paths.tokenizer_path)
            return _build_prepared_context_artifact_with_tokenizer(
                prepared_bundle=raw_bundle,
                config=raw_config,
                tokenizer=frozen_tokenizer,
                write_ref=write_ref,
                formal=True,
                data_ref=data_ref,
                train_partition_ref=train_partition_ref,
                lexicon_ref=lexicon_ref,
                target_root=target_root,
                sealed_lineage=sealed_lineage,
                workspace_root=workspace_root,
                formal_score_replayer=_default_formal_score_replayer,
                runtime_source_identity=runtime_identity,
            )
    except ModelRegistryError as exc:
        raise ContextBuildError(f"formal context source lease failed: {exc}") from exc


def seal_test_context_artifact(
    *,
    frozen_context_ref: str | Path,
    data_ref: str | Path,
    train_partition_ref: str | Path,
    prepared_bundle: str | Path | Mapping[str, Any],
    tokenizer: Any | None = None,
    write_ref: str | Path | None,
    target_root: str | Path | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
) -> dict[str, Any]:
    """Build test context once from the exact frozen dev policy/code lineage."""

    if tokenizer is not None or formal_score_replayer is not None:
        raise ContextBuildError(
            "scientific seal-test forbids caller-injected tokenizer/score replayer"
        )

    root = Path(workspace_root).resolve()
    frozen_locator = _read_json(frozen_context_ref)
    if not isinstance(frozen_locator, Mapping):
        raise ContextBuildError("frozen context ref must be an object")
    _validate_locator(frozen_locator, "context")
    frozen_target = Path(str(frozen_locator["target_path"]))
    frozen_meta = validate_context_target(
        frozen_target,
        workspace_root=root,
    )
    if frozen_meta.get("split") != "dev" or frozen_meta.get("scientific_eligible") is not True:
        raise ContextBuildError("seal-test requires a scientific frozen dev context")
    frozen_inputs = frozen_meta.get("id_inputs")
    if not isinstance(frozen_inputs, Mapping) or frozen_inputs.get(
        "builder_code_sha256"
    ) != sha256_file(__file__):
        raise ContextBuildError("frozen dev context builder code differs from seal-test code")
    frozen_config = _read_json(frozen_target / "config.resolved.json")
    frozen_provenance = _read_json(frozen_target / "provenance.json")
    frozen_dependencies = frozen_provenance.get("dependencies")
    if not isinstance(frozen_dependencies, Mapping) or set(frozen_dependencies) != {
        "data",
        "train_partition",
        "lexicon",
    }:
        raise ContextBuildError(
            "frozen dev context lacks exact data/partition/lexicon lineage"
        )
    data_locator = _read_json(data_ref)
    if not isinstance(data_locator, Mapping):
        raise ContextBuildError("sealed data ref must be an object")
    compact_data_dependency = _validate_locator(data_locator, "data")
    if compact_data_dependency != frozen_dependencies["data"]:
        raise ContextBuildError("sealed data ref differs from frozen dev data dependency")
    data_target = Path(str(data_locator["target_path"]))
    data_dependency = _workspace_dependency(data_locator, data_target, root)
    partition_locator = _read_json(train_partition_ref)
    if not isinstance(partition_locator, Mapping):
        raise ContextBuildError("sealed train-partition ref must be an object")
    compact_partition_dependency = _validate_locator(
        partition_locator, "train-partition"
    )
    partition_target = Path(str(partition_locator["target_path"]))
    partition_dependency = _workspace_dependency(
        partition_locator, partition_target, root
    )
    if partition_dependency != frozen_dependencies["train_partition"]:
        raise ContextBuildError(
            "sealed train-partition ref differs from frozen dev dependency"
        )

    raw_bundle = (
        _read_json(prepared_bundle)
        if not isinstance(prepared_bundle, Mapping)
        else dict(prepared_bundle)
    )
    _validate_prepared_bundle(
        raw_bundle, formal=True, sealed=True, workspace_root=root
    )
    if raw_bundle.get("split") != "test":
        raise ContextBuildError("sealed context prepared bundle must use split=test")
    frozen_demo = _catalog(
        _read_jsonl(frozen_target / "catalogs/demo_pool.train.jsonl"), "demo"
    )
    frozen_lexicon = _catalog(
        _read_jsonl(frozen_target / "catalogs/lexicon_pool.jsonl"), "lexicon"
    )
    frozen_train = _query_pool(
        _read_jsonl(frozen_target / "catalogs/query_pool.train.jsonl"),
        source_split="train",
    )
    if _catalog(raw_bundle.get("demo_catalog", []), "demo") != frozen_demo:
        raise ContextBuildError("sealed test demo catalog differs from frozen dev catalog")
    if _catalog(raw_bundle.get("lexicon_catalog", []), "lexicon") != frozen_lexicon:
        raise ContextBuildError("sealed test lexicon catalog differs from frozen dev catalog")
    if _query_pool(raw_bundle.get("train_query_pool"), source_split="train") != frozen_train:
        raise ContextBuildError("sealed test train query pool differs from frozen dev lineage")
    frozen_prepared = _read_json(frozen_target / "prepared_bundle.meta.json")
    test_retrieval = raw_bundle.get("retrieval_provenance")
    dev_retrieval = frozen_prepared.get("retrieval_provenance")
    if not isinstance(test_retrieval, Mapping) or not isinstance(dev_retrieval, Mapping):
        raise ContextBuildError("sealed/dev retrieval provenance is missing")
    if _retrieval_policy_projection(test_retrieval) != _retrieval_policy_projection(
        dev_retrieval
    ):
        raise ContextBuildError("sealed test retrieval policy/scorer differs from frozen dev")
    test_data_path = data_target / "test.json"
    if not test_data_path.is_file():
        raise ContextBuildError("sealed data target lacks test.json")
    test_records = _read_json(test_data_path)
    if not isinstance(test_records, list):
        raise ContextBuildError("sealed data test split must be an array")
    if _query_pool(
        raw_bundle.get("query_pool"), source_split="test"
    ) != build_query_pool(test_records, source_split="test"):
        raise ContextBuildError("sealed test query frame differs from frozen data target")
    if raw_bundle.get("data_dependency") != data_dependency:
        raise ContextBuildError("prepared sealed data dependency is inconsistent")
    if raw_bundle.get("lexicon_dependency") != frozen_inputs.get(
        "source_dependency_refs", {}
    ).get("lexicon"):
        raise ContextBuildError("prepared sealed lexicon dependency is inconsistent")
    if raw_bundle.get("train_partition_dependency") != partition_dependency:
        raise ContextBuildError(
            "prepared sealed train-partition dependency is inconsistent"
        )
    lineage = {
        "schema_version": FROZEN_CONTEXT_POLICY_SCHEMA,
        "frozen_dev_context_dependency": _workspace_dependency(
            frozen_locator, frozen_target, root
        ),
        "data_dependency": data_dependency,
        "train_partition_dependency": partition_dependency,
        "lexicon_dependency": dict(frozen_dependencies["lexicon"]),
        "dev_context_config_sha256": canonical_sha256(frozen_config),
        "retrieval_policy_sha256": canonical_sha256(
            _retrieval_policy_projection(dev_retrieval)
        ),
        "rendering_policy_sha256": canonical_sha256(frozen_config["rendering"]),
        "budget_policy_sha256": canonical_sha256(frozen_config["budget"]),
        "output_protocol_sha256": canonical_sha256(frozen_config["output_protocol"]),
        "context_builder_code_sha256": frozen_inputs["builder_code_sha256"],
        "rendering_identity_sha256": canonical_sha256(
            frozen_inputs["rendering_identity"]
        ),
        "runtime_source_identity_sha256": canonical_sha256(
            frozen_inputs["runtime_source_identity"]
        ),
    }
    return build_prepared_context_artifact(
        prepared_bundle=raw_bundle,
        config=frozen_config,
        write_ref=write_ref,
        formal=True,
        target_root=target_root,
        sealed_lineage=lineage,
        workspace_root=root,
    )


def _validate_context_target_with_tokenizer(
    target_dir: str | Path,
    *,
    tokenizer: Any,
    require_directory_name: bool = True,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
    runtime_source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = Path(target_dir)
    if target.is_symlink() or not target.is_dir():
        raise ContextBuildError("context target must be a real directory")
    _verify_payload(target)
    metas = list(target.glob("context_manifest.*.meta.json"))
    if len(metas) != 1:
        raise ContextBuildError("context target must contain exactly one split meta")
    meta = _read_json(metas[0])
    if meta.get("schema_version") != CONTEXT_META_SCHEMA:
        raise ContextBuildError("wrong context meta schema")
    build_id = meta.get("context_build_id")
    if not isinstance(build_id, str) or not re.fullmatch(r"ctx-[0-9a-f]{64}", build_id):
        raise ContextBuildError("invalid context build ID")
    if require_directory_name and target.name != build_id:
        raise ContextBuildError("context target directory/ID mismatch")
    id_inputs = meta.get("id_inputs")
    if not isinstance(id_inputs, Mapping) or "ctx-" + canonical_sha256(id_inputs) != build_id:
        raise ContextBuildError("context build ID cannot be recomputed")
    if id_inputs.get("builder_code_sha256") != sha256_file(__file__):
        raise ContextBuildError("context builder implementation changed")
    split = meta.get("split")
    if split not in {"train", "dev", "test"}:
        raise ContextBuildError("invalid context split")
    expected_kind = "test-context" if split == "test" else "context"
    if (
        meta.get("artifact_kind") != expected_kind
        or id_inputs.get("artifact_kind") != expected_kind
        or id_inputs.get("split") != split
        or meta.get("scientific_eligible") is not bool(id_inputs.get("formal"))
    ):
        raise ContextBuildError("context kind/split/scientific lineage mismatch")
    stored_config = _read_json(target / "config.resolved.json")
    prepared_meta = _read_json(target / "prepared_bundle.meta.json")
    provenance = _read_json(target / "provenance.json")
    scientific = meta.get("scientific_eligible") is True
    current_rendering_identity = _resolved_rendering_identity(stored_config)
    if (
        id_inputs.get("rendering_identity") != current_rendering_identity
        or meta.get("rendering_identity") != current_rendering_identity
    ):
        raise ContextBuildError("context resolved prompt/renderer identity changed")
    expected_runtime_identity: Mapping[str, Any]
    if scientific:
        if not isinstance(runtime_source_identity, Mapping):
            raise ContextBuildError("scientific context lacks a verified source lease")
        expected_runtime_identity = runtime_source_identity
    else:
        expected_runtime_identity = {
            "schema_version": "stage1-context-engineering-tokenizer-injection/v1",
            "scientific_eligible": False,
        }
    if (
        id_inputs.get("runtime_source_identity") != expected_runtime_identity
        or meta.get("runtime_source_identity") != expected_runtime_identity
    ):
        raise ContextBuildError("context tokenizer/scorer source identity changed")
    score_replayer = (
        _memoized_score_replayer(
            formal_score_replayer or _default_formal_score_replayer
        )
        if scientific
        else None
    )
    prepared_meta_keys = {
        "schema_version",
        "bundle_sha256",
        "prepared_bundle_sha256",
        "retrieval_provenance",
    }
    if scientific:
        prepared_meta_keys.add("prepared_bundle_file_sha256")
    if (
        not isinstance(stored_config, Mapping)
        or id_inputs.get("context_config_sha256")
        != canonical_sha256(stored_config)
        or not isinstance(prepared_meta, Mapping)
        or set(prepared_meta) != prepared_meta_keys
        or prepared_meta.get("schema_version") != PREPARED_BUNDLE_SCHEMA
        or prepared_meta.get("prepared_bundle_sha256")
        != id_inputs.get("prepared_bundle_sha256")
        or prepared_meta.get("bundle_sha256") != id_inputs.get("bundle_sha256")
        or not isinstance(prepared_meta.get("bundle_sha256"), str)
        or not SHA256_RE.fullmatch(str(prepared_meta.get("bundle_sha256")))
    ):
        raise ContextBuildError("context config/prepared-bundle lineage mismatch")
    expected_provenance = {
        "schema_version": CONTEXT_PROVENANCE_SCHEMA,
        "context_build_id": build_id,
        "dependencies": id_inputs.get("dependencies"),
        "retrieval_provenance": prepared_meta.get("retrieval_provenance"),
        "saw_model_predictions": False,
        "builder_code_sha256": id_inputs.get("builder_code_sha256"),
        "rendering_identity": current_rendering_identity,
        "runtime_source_identity": expected_runtime_identity,
    }
    if provenance != expected_provenance:
        raise ContextBuildError("context provenance cannot be replayed")
    frozen_prepared_bundle: dict[str, Any] | None = None
    if scientific:
        prepared_bundle_path = target / "prepared_bundle.json"
        if not prepared_bundle_path.is_file():
            raise ContextBuildError(
                "scientific context lacks its frozen prepared bundle"
            )
        value = _read_json(prepared_bundle_path)
        if not isinstance(value, Mapping):
            raise ContextBuildError("frozen prepared bundle must be an object")
        frozen_prepared_bundle = dict(value)
        expected_file_sha256 = _prepared_bundle_file_sha256(
            frozen_prepared_bundle
        )
        if (
            prepared_bundle_path.read_bytes()
            != _canonical_bytes(frozen_prepared_bundle) + b"\n"
            or expected_file_sha256
            != prepared_meta.get("prepared_bundle_file_sha256")
            or expected_file_sha256
            != id_inputs.get("prepared_bundle_file_sha256")
            or canonical_sha256(frozen_prepared_bundle)
            != id_inputs.get("prepared_bundle_sha256")
            or frozen_prepared_bundle.get("bundle_sha256")
            != id_inputs.get("bundle_sha256")
            or frozen_prepared_bundle.get("split") != split
        ):
            raise ContextBuildError(
                "frozen prepared bundle differs from context ID inputs"
            )
        _validate_prepared_bundle(
            frozen_prepared_bundle,
            formal=True,
            sealed=split == "test",
            workspace_root=workspace_root,
        )
        if (
            frozen_prepared_bundle.get("retrieval_provenance")
            != prepared_meta.get("retrieval_provenance")
        ):
            raise ContextBuildError(
                "frozen prepared retrieval provenance differs from metadata"
            )
        if id_inputs.get("retrieval_score_inputs") != frozen_prepared_bundle.get(
            "retrieval_provenance", {}
        ).get("score_matrix"):
            raise ContextBuildError(
                "frozen retrieval score inputs differ from context ID inputs"
            )
        _validate_formal_retrieval_provenance(
            prepared_meta.get("retrieval_provenance"),
            split=str(split),
            workspace_root=workspace_root,
        )
    resolved_source_targets: dict[str, Path] = {}
    if scientific and split != "test":
        source_refs = id_inputs.get("source_dependency_refs")
        if not isinstance(source_refs, Mapping) or set(source_refs) != {
            "data",
            "train_partition",
            "lexicon",
        }:
            raise ContextBuildError(
                "scientific train/dev context lacks portable source refs"
            )
        for kind in ("data", "train_partition", "lexicon"):
            stored_ref = _read_json(target / f"{kind}_ref.json")
            if stored_ref != source_refs[kind]:
                raise ContextBuildError(f"context {kind} ref differs from ID inputs")
            resolved_source_targets[kind] = _resolve_workspace_dependency(
                stored_ref,
                workspace_root,
                kind=("train-partition" if kind == "train_partition" else kind),
            )
            compact = {
                key: stored_ref[key]
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "artifact_id",
                    "payload_manifest_sha256",
                )
            }
            expected_dependency = (
                stored_ref if kind == "train_partition" else compact
            )
            if expected_dependency != id_inputs.get("dependencies", {}).get(kind):
                raise ContextBuildError(
                    f"context {kind} ref differs from provenance dependency"
                )
    elif scientific:
        frozen_policy = id_inputs.get("frozen_policy_ref")
        if (
            not isinstance(frozen_policy, Mapping)
            or _read_json(target / "frozen_policy_ref.json") != frozen_policy
        ):
            raise ContextBuildError(
                "scientific test context lacks its frozen policy lineage"
            )
        data_target, partition_target, lexicon_target = _sealed_source_targets(
            frozen_policy, workspace_root
        )
        resolved_source_targets = {
            "data": data_target,
            "train_partition": partition_target,
            "lexicon": lexicon_target,
        }
    records_path = target / f"context_manifest.{split}.jsonl"
    records = _read_jsonl(records_path)
    if meta.get("record_count") != len(records) or meta.get("records_sha256") != sha256_file(records_path):
        raise ContextBuildError("context record count/hash mismatch")
    for record in records:
        validate_context_record(record)
        if record.get("context_build_id") != build_id:
            raise ContextBuildError("context record build ID mismatch")
    ids = [record["query"]["id"] for record in records]
    if meta.get("ordered_query_ids_sha256") != canonical_sha256(ids):
        raise ContextBuildError("context ordered query frame mismatch")
    lexicon_catalog = _catalog(
        _read_jsonl(target / "catalogs" / "lexicon_pool.jsonl"), "lexicon"
    )
    demo_catalog = _catalog(
        _read_jsonl(target / "catalogs" / "demo_pool.train.jsonl"), "demo"
    )
    train_query_pool = _query_pool(
        _read_jsonl(target / "catalogs" / "query_pool.train.jsonl"),
        source_split="train",
    )
    split_query_pool = _query_pool(
        _read_jsonl(target / "catalogs" / f"query_pool.{split}.jsonl"),
        source_split=split,
    )
    _validate_query_pool_lineage(
        train_pool=train_query_pool,
        split_pool=split_query_pool,
        split=split,
        records=records,
        demo_catalog=demo_catalog,
    )
    if scientific:
        if frozen_prepared_bundle is None:  # pragma: no cover - guarded above
            raise ContextBuildError("scientific context lacks replay inputs")
        frozen_lexicon_catalog = _catalog(
            frozen_prepared_bundle.get("lexicon_catalog", []), "lexicon"
        )
        frozen_demo_catalog = _catalog(
            frozen_prepared_bundle.get("demo_catalog", []), "demo"
        )
        frozen_train_pool = _query_pool(
            frozen_prepared_bundle.get("train_query_pool"),
            source_split="train",
        )
        frozen_split_pool = _query_pool(
            frozen_prepared_bundle.get("query_pool"),
            source_split=str(split),
        )
        if (
            frozen_lexicon_catalog != lexicon_catalog
            or frozen_demo_catalog != demo_catalog
            or frozen_train_pool != train_query_pool
            or frozen_split_pool != split_query_pool
        ):
            raise ContextBuildError(
                "frozen prepared catalogs/query pools differ from context payload"
            )
        embedded_dependencies = {
            "data_dependency": _read_json(target / "data_ref.json"),
            "train_partition_dependency": _read_json(
                target / "train_partition_ref.json"
            ),
            "lexicon_dependency": _read_json(target / "lexicon_ref.json"),
        }
        prepared_dependencies = {
            key: frozen_prepared_bundle.get(key)
            for key in embedded_dependencies
        }
        if split == "test" and isinstance(
            prepared_dependencies["lexicon_dependency"], Mapping
        ):
            prepared_dependencies["lexicon_dependency"] = {
                key: prepared_dependencies["lexicon_dependency"][key]
                for key in (
                    "schema_version",
                    "artifact_kind",
                    "artifact_id",
                    "payload_manifest_sha256",
                )
            }
        if prepared_dependencies != embedded_dependencies:
            raise ContextBuildError(
                "frozen prepared dependencies differ from context source refs"
            )
        system_prompt, user_prompt = _resolved_prompts(stored_config)
        stored_budget = stored_config.get("budget")
        if not isinstance(stored_budget, Mapping):
            raise ContextBuildError("frozen context config lacks its budget")
        replayed_records = _replay_prepared_records(
            prepared_records=frozen_prepared_bundle.get("records", []),
            context_build_id=build_id,
            lexicon_catalog={row["lexicon_id"]: row for row in lexicon_catalog},
            demo_catalog={row["demo_id"]: row for row in demo_catalog},
            system_prompt=system_prompt,
            user_prompt_template=user_prompt,
            tokenizer=tokenizer,
            budget=stored_budget,
        )
        if replayed_records != records:
            raise ContextBuildError(
                "context records differ from exact frozen prepared-bundle replay"
            )
    if scientific:
        snapshot = _formal_source_snapshot(
            data_target=resolved_source_targets["data"],
            partition_target=resolved_source_targets["train_partition"],
            lexicon_target=resolved_source_targets["lexicon"],
            split=str(split),
            source_class_order=_source_class_order(stored_config),
            retrieval_provenance=prepared_meta["retrieval_provenance"],
            workspace_root=workspace_root,
        )
        _assert_source_snapshot_matches_bundle(
            snapshot,
            train_query_pool=train_query_pool,
            split_query_pool=split_query_pool,
            demo_catalog=demo_catalog,
            lexicon_catalog=lexicon_catalog,
            retrieval_provenance=prepared_meta["retrieval_provenance"],
        )
        if frozen_prepared_bundle is None or score_replayer is None:
            raise ContextBuildError("scientific context lacks score replay inputs")
        _replay_formal_retrieval_bundle(
            bundle=frozen_prepared_bundle,
            config=stored_config,
            snapshot=snapshot,
            score_replayer=score_replayer,
        )
    sources = meta.get("sources", {})
    expected_sources = {
        "train_query_pool": {
            "split": "train",
            "record_count": len(train_query_pool),
            "sha256": sha256_file(target / "catalogs" / "query_pool.train.jsonl"),
        },
        "split_query_pool": {
            "split": split,
            "record_count": len(split_query_pool),
            "sha256": sha256_file(target / "catalogs" / f"query_pool.{split}.jsonl"),
        },
    }
    for key, expected in expected_sources.items():
        if sources.get(key) != expected:
            raise ContextBuildError(f"context meta {key} lineage mismatch")
    expected_partition_label = (
        "fit" if meta.get("scientific_eligible") is True else "legacy-full-train"
    )
    if sources.get("demo_pool") != {
        "split": "train",
        "partition": expected_partition_label,
        "record_count": len(demo_catalog),
    } or sources.get("lexicon_pool") != {
        "train_only_verified": True,
        "partition": expected_partition_label,
        "record_count": len(lexicon_catalog),
    }:
        raise ContextBuildError("context meta catalog lineage mismatch")
    for condition in CONDITIONS:
        expected = [render_condition_item(record, condition) for record in records]
        actual = _read_json(target / "conditions" / "runner" / condition / f"{split}.json")
        if actual != expected:
            raise ContextBuildError(f"{condition} runner adapter cannot be replayed")
    if split != "test":
        expected_files = {
            "config.resolved.json",
            "prepared_bundle.meta.json",
            "catalogs/lexicon_pool.jsonl",
            "catalogs/demo_pool.train.jsonl",
            "catalogs/query_pool.train.jsonl",
            f"context_manifest.{split}.jsonl",
            f"context_manifest.{split}.meta.json",
            "provenance.json",
            "payload_manifest.json",
            *{
                f"conditions/runner/{condition}/{split}.json"
                for condition in CONDITIONS
            },
        }
        if split != "train":
            expected_files.add(f"catalogs/query_pool.{split}.jsonl")
        if meta.get("scientific_eligible") is True:
            expected_files.update(
                {
                    "prepared_bundle.json",
                    "data_ref.json",
                    "train_partition_ref.json",
                    "lexicon_ref.json",
                }
            )
        actual_files = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ContextBuildError("context file set is not canonical")
    if split == "test":
        if (
            meta.get("artifact_kind") != "test-context"
            or meta.get("scientific_eligible") is not True
        ):
            raise ContextBuildError("test context must be a scientific test-context artifact")
        frozen_policy = _read_json(target / "frozen_policy_ref.json")
        data_dependency = _read_json(target / "data_ref.json")
        train_partition_dependency = _read_json(
            target / "train_partition_ref.json"
        )
        lexicon_dependency = _read_json(target / "lexicon_ref.json")
        if (
            not isinstance(frozen_policy, Mapping)
            or frozen_policy.get("schema_version") != FROZEN_CONTEXT_POLICY_SCHEMA
            or id_inputs.get("frozen_policy_ref") != frozen_policy
            or frozen_policy.get("data_dependency") != data_dependency
            or frozen_policy.get("train_partition_dependency")
            != train_partition_dependency
            or frozen_policy.get("lexicon_dependency") != lexicon_dependency
        ):
            raise ContextBuildError("sealed context frozen-policy files/ID inputs disagree")
        frozen_target = _resolve_workspace_dependency(
            frozen_policy["frozen_dev_context_dependency"],
            workspace_root,
            kind="context",
        )
        frozen_meta = validate_context_target(
            frozen_target,
            workspace_root=workspace_root,
        )
        if frozen_meta.get("split") != "dev" or frozen_meta.get(
            "scientific_eligible"
        ) is not True:
            raise ContextBuildError("sealed context does not bind a scientific dev context")
        frozen_config = _read_json(frozen_target / "config.resolved.json")
        if _read_json(target / "config.resolved.json") != frozen_config:
            raise ContextBuildError("sealed context config differs from frozen dev")
        frozen_provenance = _read_json(frozen_target / "provenance.json")
        frozen_dependencies = frozen_provenance.get("dependencies")
        if not isinstance(frozen_dependencies, Mapping):
            raise ContextBuildError("frozen dev context dependency lineage is missing")
        compact_data = {
            key: data_dependency[key]
            for key in (
                "schema_version",
                "artifact_kind",
                "artifact_id",
                "payload_manifest_sha256",
            )
        }
        if (
            compact_data != frozen_dependencies.get("data")
            or train_partition_dependency
            != frozen_dependencies.get("train_partition")
            or lexicon_dependency != frozen_dependencies.get("lexicon")
        ):
            raise ContextBuildError(
                "sealed context data/partition/lexicon lineage differs from dev"
            )
        data_target = _resolve_workspace_dependency(
            data_dependency, workspace_root, kind="data"
        )
        partition_target = _resolve_workspace_dependency(
            train_partition_dependency,
            workspace_root,
            kind="train-partition",
        )
        frozen_inputs = frozen_meta.get("id_inputs")
        frozen_source_refs = (
            frozen_inputs.get("source_dependency_refs")
            if isinstance(frozen_inputs, Mapping)
            else None
        )
        if (
            not isinstance(frozen_source_refs, Mapping)
            or not isinstance(frozen_source_refs.get("lexicon"), Mapping)
            or not isinstance(
                frozen_source_refs.get("train_partition"), Mapping
            )
        ):
            raise ContextBuildError(
                "frozen dev context lacks partition/lexicon source lineage"
            )
        frozen_lexicon_ref = frozen_source_refs["lexicon"]
        lexicon_target = _resolve_workspace_dependency(
            frozen_lexicon_ref, workspace_root, kind="lexicon"
        )
        compact_frozen_lexicon = {
            key: frozen_lexicon_ref[key]
            for key in (
                "schema_version",
                "artifact_kind",
                "artifact_id",
                "payload_manifest_sha256",
            )
        }
        if compact_frozen_lexicon != lexicon_dependency:
            raise ContextBuildError("sealed lexicon dependency differs from frozen source")
        frozen_partition_ref = frozen_source_refs["train_partition"]
        if frozen_partition_ref != train_partition_dependency:
            raise ContextBuildError(
                "sealed train-partition dependency differs from frozen source"
            )
        raw_test = _read_json(data_target / "test.json")
        if not isinstance(raw_test, list) or split_query_pool != build_query_pool(
            raw_test, source_split="test"
        ):
            raise ContextBuildError("sealed context query frame differs from data target")
        if demo_catalog != _catalog(
            _read_jsonl(frozen_target / "catalogs/demo_pool.train.jsonl"), "demo"
        ) or lexicon_catalog != _catalog(
            _read_jsonl(frozen_target / "catalogs/lexicon_pool.jsonl"), "lexicon"
        ):
            raise ContextBuildError("sealed context train catalogs differ from frozen dev")
        prepared = _read_json(target / "prepared_bundle.meta.json")
        frozen_prepared = _read_json(frozen_target / "prepared_bundle.meta.json")
        test_retrieval = prepared.get("retrieval_provenance")
        dev_retrieval = frozen_prepared.get("retrieval_provenance")
        expected_policy = {
            "schema_version": FROZEN_CONTEXT_POLICY_SCHEMA,
            "frozen_dev_context_dependency": frozen_policy[
                "frozen_dev_context_dependency"
            ],
            "data_dependency": data_dependency,
            "train_partition_dependency": train_partition_dependency,
            "lexicon_dependency": lexicon_dependency,
            "dev_context_config_sha256": canonical_sha256(frozen_config),
            "retrieval_policy_sha256": canonical_sha256(
                _retrieval_policy_projection(dev_retrieval)
            ),
            "rendering_policy_sha256": canonical_sha256(frozen_config["rendering"]),
            "budget_policy_sha256": canonical_sha256(frozen_config["budget"]),
            "output_protocol_sha256": canonical_sha256(
                frozen_config["output_protocol"]
            ),
            "context_builder_code_sha256": sha256_file(__file__),
            "rendering_identity_sha256": canonical_sha256(
                frozen_inputs["rendering_identity"]
            ),
            "runtime_source_identity_sha256": canonical_sha256(
                frozen_inputs["runtime_source_identity"]
            ),
        }
        if (
            not isinstance(test_retrieval, Mapping)
            or not isinstance(dev_retrieval, Mapping)
            or _retrieval_policy_projection(test_retrieval)
            != _retrieval_policy_projection(dev_retrieval)
            or frozen_policy != expected_policy
        ):
            raise ContextBuildError("sealed context policy/code projection cannot be replayed")
        sealed_snapshot = _formal_source_snapshot(
            data_target=data_target,
            partition_target=partition_target,
            lexicon_target=lexicon_target,
            split="test",
            source_class_order=_source_class_order(frozen_config),
            retrieval_provenance=test_retrieval,
            workspace_root=workspace_root,
        )
        _assert_source_snapshot_matches_bundle(
            sealed_snapshot,
            train_query_pool=train_query_pool,
            split_query_pool=split_query_pool,
            demo_catalog=demo_catalog,
            lexicon_catalog=lexicon_catalog,
            retrieval_provenance=test_retrieval,
        )
        expected_files = {
            "config.resolved.json",
            "prepared_bundle.json",
            "prepared_bundle.meta.json",
            "catalogs/lexicon_pool.jsonl",
            "catalogs/demo_pool.train.jsonl",
            "catalogs/query_pool.train.jsonl",
            "catalogs/query_pool.test.jsonl",
            "context_manifest.test.jsonl",
            "context_manifest.test.meta.json",
            "provenance.json",
            "data_ref.json",
            "train_partition_ref.json",
            "lexicon_ref.json",
            "frozen_policy_ref.json",
            "payload_manifest.json",
            *{
                f"conditions/runner/{condition}/test.json" for condition in CONDITIONS
            },
        }
        actual_files = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ContextBuildError("sealed context file set is not canonical")
    return dict(meta)


def validate_context_target(
    target_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    require_directory_name: bool = True,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
) -> dict[str, Any]:
    """Validate a context, reconstructing scientific backends under lease.

    ``tokenizer`` remains accepted for engineering artifacts and compatibility
    with callers that validate both scopes.  It is deliberately ignored for a
    scientific target, so a caller-selected object cannot affect publication or
    replay.  The public formal score-replayer hook is rejected for the same
    reason; tests patch the private constructor/default instead.
    """

    target = Path(target_dir)
    metas = list(target.glob("context_manifest.*.meta.json")) if target.is_dir() else []
    if len(metas) != 1:
        raise ContextBuildError("context target must contain exactly one split meta")
    preview = _read_json(metas[0])
    scientific = isinstance(preview, Mapping) and preview.get(
        "scientific_eligible"
    ) is True
    if not scientific:
        if tokenizer is None:
            raise ContextBuildError(
                "engineering context validation requires its test-only tokenizer"
            )
        if formal_score_replayer is not None:
            raise ContextBuildError(
                "engineering context must not claim a formal score replayer"
            )
        return _validate_context_target_with_tokenizer(
            target,
            tokenizer=tokenizer,
            require_directory_name=require_directory_name,
            workspace_root=workspace_root,
            runtime_source_identity=None,
        )
    if formal_score_replayer is not None:
        raise ContextBuildError(
            "scientific context validation forbids caller-injected score replay"
        )
    config = _read_json(target / "config.resolved.json")
    prepared_meta = _read_json(target / "prepared_bundle.meta.json")
    retrieval_provenance = (
        prepared_meta.get("retrieval_provenance")
        if isinstance(prepared_meta, Mapping)
        else None
    )
    if not isinstance(config, Mapping) or not isinstance(
        retrieval_provenance, Mapping
    ):
        raise ContextBuildError("scientific context lacks runtime source inputs")
    runtime_identity, contract = _runtime_source_identity(
        config=config,
        retrieval_provenance=retrieval_provenance,
        workspace_root=workspace_root,
    )
    try:
        with verified_model_source_lease(
            contract, source_names=("tokenizer", "checkpoint")
        ) as paths:
            frozen_tokenizer = _construct_formal_tokenizer(paths.tokenizer_path)
            return _validate_context_target_with_tokenizer(
                target,
                tokenizer=frozen_tokenizer,
                require_directory_name=require_directory_name,
                workspace_root=workspace_root,
                formal_score_replayer=_default_formal_score_replayer,
                runtime_source_identity=runtime_identity,
            )
    except ModelRegistryError as exc:
        raise ContextBuildError(f"formal context source lease failed: {exc}") from exc


def validate_context_ref(
    ref_path: str | Path,
    *,
    tokenizer: Any | None = None,
    workspace_root: str | Path = REPOSITORY_ROOT,
    formal_score_replayer: FormalScoreReplayer | None = None,
) -> dict[str, Any]:
    locator = _read_json(ref_path)
    if locator.get("schema_version") != LOCATOR_SCHEMA or locator.get("artifact_kind") not in {"context", "test-context"}:
        raise ContextBuildError("invalid context locator")
    target = Path(str(locator.get("target_path", "")))
    meta = validate_context_target(
        target,
        tokenizer=tokenizer,
        workspace_root=workspace_root,
        formal_score_replayer=formal_score_replayer,
    )
    if target.name != locator.get("artifact_id") or locator.get("payload_manifest_sha256") != _verify_payload(target):
        raise ContextBuildError("context locator disagrees with target")
    return meta


__all__ = [
    "CONTEXT_META_SCHEMA",
    "ContextBuildError",
    "PREPARED_BUNDLE_SCHEMA",
    "build_prepared_context_artifact",
    "embedding_model_file_tree_sha256",
    "resolve_embedding_model_path",
    "seal_test_context_artifact",
    "validate_context_ref",
    "validate_context_target",
]
