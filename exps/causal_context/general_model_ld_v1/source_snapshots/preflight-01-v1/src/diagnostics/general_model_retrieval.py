"""Task-independent, auditable CPU BGE retrieval for fixed model experiments."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.context_selector import SELECTION_POLICY, select_demos
from model.stage1_registry import inventory_regular_file_tree
from rag.types import (
    RetrievalHit,
    canonical_json,
    content_sha256,
    sha256_text,
    stable_demo_id,
)
from utils.quadruple import canonicalize_quadruples


SCHEMA = "general-model-bge-retrieval/v1"
CACHE_SCHEMA = "general-model-bge-numpy-cache/v1"
GROUPS = ("non-hate", "Region", "Racism", "Sexism", "LGBTQ", "others")
DEFAULT_QUOTAS = dict(zip(GROUPS, (4, 1, 1, 2, 1, 1)))
QUERY_PREFIX = (
    "\u4e3a\u8fd9\u4e2a\u53e5\u5b50\u751f\u6210\u8868\u793a"
    "\u4ee5\u7528\u4e8e\u68c0\u7d22\u76f8\u5173\u6587\u7ae0\uff1a"
)


class RetrievalError(ValueError):
    pass


class CacheIntegrityError(RetrievalError):
    pass


def _digest(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    if "max_length" in policy or "torch_threads" in policy:
        raise RetrievalError("use max_embedding_tokens and torch_num_threads in the retrieval policy")
    order = list(policy.get("source_class_order", GROUPS))
    quotas = dict(policy.get("allocated_class_top_k", DEFAULT_QUOTAS))
    if not order or len(set(order)) != len(order) or set(order) - set(GROUPS):
        raise RetrievalError("source_class_order must contain unique supported groups")
    if set(quotas) != set(order) or any(type(value) is not int or value < 0 for value in quotas.values()):
        raise RetrievalError("class quotas must be nonnegative integers for exactly the source classes")
    top_k = policy.get("demo_top_k", sum(quotas.values()))
    if type(top_k) is not int or top_k <= 0 or top_k != sum(quotas.values()):
        raise RetrievalError("demo_top_k must equal the positive sum of class quotas")
    multiplier = policy.get("candidate_multiplier", 3)
    batch_size = policy.get("batch_size", 32)
    threads = policy.get("torch_num_threads", 4)
    if any(type(value) is not int or value < 1 for value in (multiplier, batch_size, threads)):
        raise RetrievalError("candidate_multiplier, batch_size and torch_num_threads must be positive integers")
    threshold = float(policy.get("similarity_threshold", 0.0))
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise RetrievalError("similarity_threshold must be finite and within [-1, 1]")
    if policy.get("max_embedding_tokens", 512) != 512 or policy.get("query_prefix", QUERY_PREFIX) != QUERY_PREFIX:
        raise RetrievalError("BGE token budget and query prefix are fixed by this protocol")
    return {
        "schema_version": SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "source_class_order": order,
        "allocated_class_top_k": quotas,
        "demo_top_k": top_k,
        "candidate_multiplier": multiplier,
        "similarity_threshold": threshold,
        "batch_size": batch_size,
        "torch_num_threads": threads,
        "max_embedding_tokens": 512,
        "query_prefix": QUERY_PREFIX,
        "embedding_device": "cpu",
        "embedding_dtype": "float32",
        "pooling": "last-hidden-state-cls-l2-normalized",
        "content_normalization": "crlf-to-lf/v1",
        "source_duplicate_policy": "same-content-same-gold-minimum-numeric-id-otherwise-exclude-cluster/v1",
        "gold_identity": "sorted-quadruples-and-group-sets/v1",
        "multiquad_policy": "include-all-groups-from-all-quadruples/v1",
        "query_overlap_policy": "exclude-source-id-or-normalized-content-hash/v1",
        "embedding_overflow_policy": "truncate-at-512-with-per-record-token-counts/v1",
    }


def _source_id(value: Any) -> str:
    if isinstance(value, bool) or not str(value).isdigit() or int(str(value)) < 1:
        raise RetrievalError("demo IDs must be positive decimal integers")
    return str(int(str(value)))


def _gold_identity(quadruples: Any) -> tuple[str, list[str]]:
    if not isinstance(quadruples, list) or not quadruples:
        raise RetrievalError("every demo must have a nonempty quadruples list")
    normalized = []
    classes: set[str] = set()
    for quad in canonicalize_quadruples(quadruples):
        groups = quad.targeted_group
        normalized.append({
            "target": quad.target, "argument": quad.argument,
            "targeted_group": sorted(groups), "hateful": quad.hateful,
        })
        classes.update(groups)
    return _digest(sorted(normalized, key=canonical_json)), [group for group in GROUPS if group in classes]


def prepare_demo_pool(
    demos: Sequence[Mapping[str, Any]], policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve duplicate content before embedding, retaining full quadruples."""
    resolved = resolve_policy(policy)
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids: set[str] = set()
    for source in demos:
        source_id = _source_id(source.get("id"))
        if source_id in ids:
            raise RetrievalError(f"duplicate demo source ID: {source_id}")
        ids.add(source_id)
        if not isinstance(source.get("content"), str) or not source["content"]:
            raise RetrievalError(f"demo {source_id} has no content")
        gold_hash, classes = _gold_identity(source.get("quadruples"))
        content = source["content"].replace("\r\n", "\n")
        content_hash = content_sha256(content)
        row = {
            "id": source_id, "content": content,
            "quadruples": copy.deepcopy(source["quadruples"]),
            "source_classes": classes, "content_sha256": content_hash,
            "gold_sha256": gold_hash,
            "demo_id": stable_demo_id(source_id, content_hash, gold_hash),
        }
        clusters[content_hash].append(row)
    retained = []
    duplicates = []
    conflicts = []
    for content_hash, rows in sorted(clusters.items()):
        rows.sort(key=lambda row: int(row["id"]))
        source_ids = [row["id"] for row in rows]
        gold_hashes = sorted({row["gold_sha256"] for row in rows})
        if len(gold_hashes) > 1:
            conflicts.append({"content_sha256": content_hash, "source_ids": source_ids, "gold_sha256s": gold_hashes})
            continue
        retained.append(rows[0])
        if len(rows) > 1:
            duplicates.append({"content_sha256": content_hash, "kept_id": source_ids[0], "removed_ids": source_ids[1:]})
    retained.sort(key=lambda row: int(row["id"]))
    if not retained:
        raise RetrievalError("no eligible demonstrations remain after content conflict filtering")
    class_counts = Counter(group for row in retained for group in row["source_classes"])
    audit = {
        "input_count": len(demos), "retained_count": len(retained),
        "retained_multiquad_count": sum(len(row["quadruples"]) > 1 for row in retained),
        "retained_multiclass_count": sum(len(row["source_classes"]) > 1 for row in retained),
        "class_counts": {group: class_counts[group] for group in resolved["source_class_order"]},
        "duplicate_clusters": duplicates, "conflicting_gold_clusters": conflicts,
        "same_gold_removed_count": sum(len(item["removed_ids"]) for item in duplicates),
        "conflicting_gold_excluded_count": sum(len(item["source_ids"]) for item in conflicts),
        "input_sha256": _digest(list(demos)), "pool_sha256": _digest(retained),
    }
    return retained, audit


def _queries(queries: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    rows = []
    ids = set()
    for query in queries:
        query_id = str(query.get("id", ""))
        if not query_id or query_id in ids:
            raise RetrievalError("query IDs must be nonempty and unique")
        ids.add(query_id)
        if not isinstance(query.get("content"), str) or not query["content"]:
            raise RetrievalError(f"query {query_id} has no content")
        content = query["content"].replace("\r\n", "\n")
        rows.append({"id": query_id, "content": content, "content_sha256": content_sha256(content)})
    if not rows:
        raise RetrievalError("at least one query is required")
    return rows


def select_from_scores(
    pool: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]],
    scores: Any, *, policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure selection over query-by-pool scores, shared by every task prompt."""
    resolved = resolve_policy(policy)
    query_rows = _queries(queries)
    if len(scores) != len(query_rows) or any(len(row) != len(pool) for row in scores):
        raise RetrievalError("score matrix dimensions must match queries and the prepared pool")
    by_id = {row["demo_id"]: row for row in pool}
    if len(by_id) != len(pool):
        raise RetrievalError("prepared pool demo IDs must be unique")
    selected_by_query = {}
    traces_by_query = {}
    for query_index, query in enumerate(query_rows):
        hits: dict[str, list[RetrievalHit]] = {group: [] for group in resolved["source_class_order"]}
        for source_index, row in enumerate(pool):
            score = float(scores[query_index][source_index])
            if not math.isfinite(score):
                raise RetrievalError("all retrieval scores must be finite")
            for group in row["source_classes"]:
                if group not in hits:
                    continue
                hits[group].append(RetrievalHit(
                    id=row["demo_id"], source_record_id=row["id"], content=row["content"],
                    output=canonical_json(row["quadruples"]), content_sha256=row["content_sha256"],
                    gold_sha256=row["gold_sha256"], score=score, rank=source_index,
                    source_class=group, method="bge-cls-l2-cosine",
                    provenance={"pool_index": source_index, "source_classes": list(row["source_classes"])},
                ))
        trace = select_demos(
            hits, source_class_order=resolved["source_class_order"],
            allocated_class_top_k=resolved["allocated_class_top_k"],
            similarity_threshold=resolved["similarity_threshold"],
            candidate_multiplier=resolved["candidate_multiplier"],
            query_source_record_id=query["id"], query_content_sha256=query["content_sha256"],
        )
        assignments = {item.demo_id: item for item in trace.quota_assignments}
        candidates = {item.demo_id: item for item in trace.candidates}
        selected = []
        for prompt_rank, demo_id in enumerate(trace.prompt_order):
            row = copy.deepcopy(dict(by_id[demo_id]))
            assignment = assignments[demo_id]
            row.update({
                "retrieval_score": candidates[demo_id].selection_score,
                "quota_class": assignment.assigned_quota_class,
                "quota_round": assignment.quota_round, "prompt_rank": prompt_rank,
            })
            selected.append(row)
        if len(selected) != resolved["demo_top_k"]:
            raise RetrievalError("selector did not return the configured number of demonstrations")
        selected_by_query[query["id"]] = selected
        traces_by_query[query["id"]] = {
            **trace.to_dict(), "query_id": query["id"], "query_content_sha256": query["content_sha256"],
            "pool_count": len(pool), "selected_source_ids": [row["id"] for row in selected],
            "prompt_rank_base": 0,
        }
    return {"selected_by_query": selected_by_query, "traces_by_query": traces_by_query}


class _BGEEncoder:
    def __init__(self, model_path: Path, policy: Mapping[str, Any]):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.policy = policy
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(policy["torch_num_threads"])
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.float32, local_files_only=True,
                trust_remote_code=False, use_safetensors=True,
            ).to("cpu").eval()
        except BaseException:
            torch.set_num_threads(self.previous_threads)
            raise

    def encode(self, texts: Sequence[str]) -> tuple[Any, list[int]]:
        import numpy as np
        import torch.nn.functional as functional

        blocks = []
        lengths = []
        started = time.monotonic()
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.policy["batch_size"]):
                batch = list(texts[start:start + self.policy["batch_size"]])
                raw = self.tokenizer(batch, padding=False, truncation=False, return_length=True)
                lengths.extend(int(value) for value in raw["length"])
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                output = self.model(**encoded)
                vectors = functional.normalize(output.last_hidden_state[:, 0].float(), p=2, dim=1)
                blocks.append(vectors.cpu().numpy().copy())
                completed = start + len(batch)
                if start == 0 or len(blocks) % 32 == 0 or completed == len(texts):
                    print(json.dumps({"build_stage": "BGE-CPU-encoding", "completed": completed,
                                      "total": len(texts), "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        return np.concatenate(blocks, axis=0).astype(np.float32, copy=False), lengths

    def close(self) -> None:
        del self.model
        self.torch.set_num_threads(self.previous_threads)


def _validate_vectors(vectors: Any, count: int) -> None:
    import numpy as np

    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 1 or vectors.dtype != np.float32:
        raise CacheIntegrityError("embedding array shape or dtype is invalid")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=1e-5):
        raise CacheIntegrityError("embedding array must contain finite L2-normalized rows")


def _read_cache(directory: Path, identity: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        expected_hash = metadata.pop("metadata_sha256")
        if expected_hash != _digest(metadata) or metadata["identity"] != identity:
            raise CacheIntegrityError("cache metadata identity or checksum does not match")
        if _file_hash(directory / "vectors.npy") != metadata["vectors_sha256"]:
            raise CacheIntegrityError("embedding cache file checksum mismatch")
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        _validate_vectors(vectors, len(identity["rows"]))
        lengths = metadata["untruncated_token_counts"]
        if len(lengths) != len(identity["rows"]) or any(type(value) is not int or value < 1 for value in lengths):
            raise CacheIntegrityError("cache token counts are invalid")
        if metadata["shape"] != list(vectors.shape):
            raise CacheIntegrityError("cache shape metadata does not match the array")
        metadata["metadata_sha256"] = expected_hash
        return vectors, metadata
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, CacheIntegrityError):
            raise
        raise CacheIntegrityError(f"invalid retrieval cache: {directory}") from exc


def _embedding_cache(
    rows: Sequence[Mapping[str, Any]], *, role: str, cache_root: Path,
    model_inventory: Mapping[str, Any], policy: Mapping[str, Any], encoder: Any,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    identity = {
        "schema_version": CACHE_SCHEMA, "role": role,
        "model_file_tree_sha256": model_inventory["file_tree_sha256"],
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "torch", "transformers")},
        "encoder_code_sha256": _file_hash(Path(__file__)),
        "policy": dict(policy),
        "rows": [{key: row[key] for key in ("id", "content_sha256", "gold_sha256") if key in row} for row in rows],
    }
    cache_id = "bge-" + _digest(identity)
    cache_root.mkdir(parents=True, exist_ok=True)
    directory = cache_root / cache_id
    with (cache_root / f"{cache_id}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if directory.exists():
            vectors, metadata = _read_cache(directory, identity)
            return vectors, {"cache_id": cache_id, "metadata": metadata}
        texts = [(QUERY_PREFIX if role == "queries" else "") + row["content"] for row in rows]
        vectors, lengths = encoder(texts)
        _validate_vectors(vectors, len(rows))
        if len(lengths) != len(rows) or any(type(value) is not int or value < 1 for value in lengths):
            raise CacheIntegrityError("encoder returned invalid token counts")
        staging = Path(tempfile.mkdtemp(prefix=f".{cache_id}-", dir=cache_root))
        try:
            np.save(staging / "vectors.npy", vectors, allow_pickle=False)
            metadata = {
                "identity": identity, "vectors_sha256": _file_hash(staging / "vectors.npy"),
                "shape": list(vectors.shape), "dtype": "float32",
                "untruncated_token_counts": lengths,
            }
            metadata["metadata_sha256"] = _digest(metadata)
            (staging / "metadata.json").write_text(canonical_json(metadata) + "\n", encoding="utf-8")
            os.rename(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return vectors, {"cache_id": cache_id, "metadata": metadata}


def build_retrieval(
    demos: list[dict[str, Any]], queries: list[dict[str, Any]], *, model_path: Path,
    cache_root: Path, policy: dict[str, Any], device: str = "cpu",
) -> dict[str, Any]:
    """Build shared D selections; the caller validates fit membership and publishes."""
    if device != "cpu":
        raise RetrievalError("this retrieval protocol requires CPU FP32")
    resolved = resolve_policy(policy)
    pool, pool_audit = prepare_demo_pool(demos, resolved)
    query_rows = _queries(queries)
    model_path = Path(model_path).resolve()
    inventory = inventory_regular_file_tree(model_path, workspace_root=model_path.parent, label="BGE encoder")
    instance = None

    def encode(texts: Sequence[str]) -> tuple[Any, list[int]]:
        nonlocal instance
        if instance is None:
            instance = _BGEEncoder(model_path, resolved)
        return instance.encode(texts)

    try:
        demo_vectors, demo_cache = _embedding_cache(
            pool, role="demos-all-quadruples", cache_root=Path(cache_root),
            model_inventory=inventory, policy=resolved, encoder=encode,
        )
        query_vectors, query_cache = _embedding_cache(
            query_rows, role="queries", cache_root=Path(cache_root),
            model_inventory=inventory, policy=resolved, encoder=encode,
        )
        after = inventory_regular_file_tree(model_path, workspace_root=model_path.parent, label="BGE encoder")
        if inventory != after:
            raise RetrievalError("BGE model/tokenizer tree changed during retrieval")
        if demo_vectors.shape[1] != query_vectors.shape[1]:
            raise CacheIntegrityError("query and demo embedding dimensions differ")
        result = select_from_scores(pool, query_rows, query_vectors @ demo_vectors.T, policy=resolved)
    finally:
        if instance is not None:
            instance.close()
    for query in query_rows:
        trace = result["traces_by_query"][query["id"]]
        trace["demo_cache_id"] = demo_cache["cache_id"]
        trace["query_cache_id"] = query_cache["cache_id"]
    truncation = {}
    for role, rows, cache in (("demos", pool, demo_cache), ("queries", query_rows, query_cache)):
        lengths = cache["metadata"]["untruncated_token_counts"]
        truncation[role] = {
            "row_count": len(rows), "truncated_count": sum(length > 512 for length in lengths),
            "records": [{"id": row["id"], "untruncated_tokens": length, "embedding_tokens": min(length, 512),
                         "truncated": length > 512} for row, length in zip(rows, lengths)],
        }
    result["summary"] = {
        "schema_version": SCHEMA, "policy": resolved, "pool": pool_audit,
        "query_count": len(query_rows), "model_inventory": inventory,
        "demo_cache": demo_cache, "query_cache": query_cache, "embedding_truncation": truncation,
        "selected_ids_sha256": _digest({key: [row["id"] for row in value] for key, value in result["selected_by_query"].items()}),
    }
    return result
