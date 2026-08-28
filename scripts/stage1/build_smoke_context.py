#!/usr/bin/env python3
"""Build a real-dev, train-only engineering context for Stage 1 P0 smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data.build_context_manifest import build_prepared_context_artifact  # noqa: E402
from data.retrieval_bundle import (  # noqa: E402
    build_lexicon_catalog,
    cosine_score_matrices,
    prepare_context_bundle_from_scores,
)
from data.training_artifacts import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    resolve_locator_ref,
    sha256_file,
    write_canonical_json,
)
from utils.quadruple import adapt_source_quad  # noqa: E402
from model.stage1_registry import (  # noqa: E402
    ModelRegistryError,
    ResolvedModelSourceContract,
    inventory_regular_file_tree,
    verified_model_source_lease,
)


class SmokeContextError(RuntimeError):
    pass


def _normalized_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record_id = raw.get("id")
    if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id <= 0:
        raise SmokeContextError("source record ID must be a positive JSON integer")
    content = raw.get("content")
    quads = raw.get("quadruples")
    if not isinstance(content, str) or not content or not isinstance(quads, list) or not quads:
        raise SmokeContextError("source record lacks content/quadruples")
    canonical = [adapt_source_quad(quad) for quad in quads]
    return {
        "id": str(record_id),
        "content": content,
        "quadruples": [
            {
                "target": quad.target,
                "argument": quad.argument,
                "targeted_group": list(quad.targeted_group),
                "hateful": quad.hateful,
            }
            for quad in canonical
        ],
    }


def _blocking_ordinals(audit_ref: Path) -> set[tuple[str, int]]:
    _, target = resolve_locator_ref(audit_ref, "data-audit")
    blocked: set[tuple[str, int]] = set()
    for issue in load_jsonl(target / "issues.jsonl"):
        for location in issue.get("locations", []):
            blocked.add((str(location["source_key"]), int(location["source_ordinal"])))
    return blocked


def _engineering_lexicon(
    train_records: Sequence[Mapping[str, Any]], *, max_terms: int
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in train_records:
        for quad in record["quadruples"]:
            target = quad["target"]
            if not isinstance(target, str) or not target.strip():
                continue
            for group in quad["targeted_group"]:
                if group != "non-hate":
                    counts[(target.strip(), group)] += 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    terms = [
        {
            "term": term,
            "category": group,
            "definition": f"仅由冻结训练标注抽取的 {group} 相关工程 smoke 词项。",
            "variants": [],
            "metadata": {
                "source": "train-annotation-frequency/engineering-smoke-v1",
                "train_count": count,
            },
        }
        for (term, group), count in ordered[:max_terms]
    ]
    if len(terms) < 10:
        raise SmokeContextError("too few deterministic train-derived smoke lexicon terms")
    return terms


def _tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise SmokeContextError(f"model tree contains symlink: {path}")
        if path.is_file():
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not rows:
        raise SmokeContextError(f"model tree is empty: {root}")
    return canonical_sha256(rows)


def _inventory(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return inventory_regular_file_tree(
            path,
            workspace_root=REPOSITORY_ROOT,
            label=label,
            inventory_policy="all-regular-files/v1",
        )
    except ModelRegistryError as exc:
        raise SmokeContextError(str(exc)) from exc


def build(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(args.source_train)
    if not isinstance(source, list) or len(source) != 6424:
        raise SmokeContextError("frozen std train source must contain 6424 records")
    blocked = _blocking_ordinals(args.audit_ref)
    train_records = [
        _normalized_record(record)
        for ordinal, record in enumerate(source[:5781])
        if ("std-train", ordinal) not in blocked
    ]
    dev_candidates = [
        _normalized_record(record)
        for ordinal, record in enumerate(source[5781:], start=5781)
        if ("std-train", ordinal) not in blocked
    ]
    if args.query_ids:
        by_id = {record["id"]: record for record in dev_candidates}
        if len(args.query_ids) != len(set(args.query_ids)):
            raise SmokeContextError("explicit smoke query IDs must be unique")
        missing = [query_id for query_id in args.query_ids if query_id not in by_id]
        if missing:
            raise SmokeContextError(f"explicit smoke query IDs are unavailable: {missing}")
        query_records = [by_id[query_id] for query_id in args.query_ids]
        selection_policy = "explicit-pre-outcome-infrastructure-eligible-ids/v1"
    else:
        query_records = dev_candidates[: args.query_count]
        selection_policy = "first-anomaly-free-dev-records/v1"
    if len(query_records) != (len(args.query_ids) if args.query_ids else args.query_count):
        raise SmokeContextError("not enough anomaly-free real dev records for smoke")
    terms = _engineering_lexicon(train_records, max_terms=args.lexicon_terms)

    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SmokeContextError("sentence-transformers and transformers are required") from exc
    lexicon_catalog = build_lexicon_catalog(terms)
    embedding_inventory = _inventory(
        args.embedding_model, label="engineering smoke embedding scorer"
    )
    embedding_contract = ResolvedModelSourceContract(
        workspace_root=REPOSITORY_ROOT,
        checkpoint_inventory=embedding_inventory,
        tokenizer_inventory=embedding_inventory,
        base_inventory=embedding_inventory,
    )
    try:
        with verified_model_source_lease(
            embedding_contract, source_names=("checkpoint",)
        ) as paths:
            embedding_model = SentenceTransformer(
                str(paths.checkpoint_path),
                device=args.device,
                local_files_only=True,
                trust_remote_code=False,
            )
            demo_scores, lexicon_scores = cosine_score_matrices(
                model=embedding_model,
                train_texts=[record["content"] for record in train_records],
                query_texts=[record["content"] for record in query_records],
                lexicon_texts=[row["rendered_block"] for row in lexicon_catalog],
                batch_size=args.batch_size,
            )
    except ModelRegistryError as exc:
        raise SmokeContextError(
            f"embedding scorer source lease failed: {exc}"
        ) from exc
    config = load_json(args.config)
    bundle = prepare_context_bundle_from_scores(
        train_records=train_records,
        query_records=query_records,
        lexicon_terms=terms,
        demo_scores=demo_scores,
        lexicon_scores=lexicon_scores,
        split="dev",
        retrieval_config=config["retrieval"],
        scorer_provenance={
            "backend": "sentence-transformers-local",
            "logical_model_path": args.embedding_model.relative_to(REPOSITORY_ROOT).as_posix(),
            "model_file_tree_sha256": _tree_hash(args.embedding_model),
            "device_class": "cuda",
            "engineering_smoke": True,
        },
    )
    bundle["engineering_fixture"] = {
        "schema_version": "stage1-real-dev-smoke-fixture/v1",
        "scientific_eligible": False,
        "source_train_sha256": sha256_file(args.source_train),
        "data_audit_id": load_json(args.audit_ref)["artifact_id"],
        "blocked_source_locations_excluded": len(blocked),
        "train_record_count": len(train_records),
        "dev_query_count": len(query_records),
        "ordered_dev_query_ids": [row["id"] for row in query_records],
        "query_selection_policy": selection_policy,
        "lexicon_policy": "train-annotation-frequency/engineering-smoke-v1",
        "lexicon_term_count": len(terms),
        "saw_test_records": False,
        "saw_model_predictions": False,
    }
    without_hash = dict(bundle)
    without_hash.pop("bundle_sha256", None)
    bundle["bundle_sha256"] = canonical_sha256(without_hash)
    write_canonical_json(args.prepared_bundle_output, bundle)

    tokenizer_inventory = _inventory(
        args.tokenizer, label="engineering smoke tokenizer"
    )
    tokenizer_contract = ResolvedModelSourceContract(
        workspace_root=REPOSITORY_ROOT,
        checkpoint_inventory=tokenizer_inventory,
        tokenizer_inventory=tokenizer_inventory,
        base_inventory=tokenizer_inventory,
    )
    try:
        with verified_model_source_lease(
            tokenizer_contract, source_names=("tokenizer",)
        ) as paths:
            tokenizer = AutoTokenizer.from_pretrained(
                str(paths.tokenizer_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            locator = build_prepared_context_artifact(
                prepared_bundle=bundle,
                config=config,
                tokenizer=tokenizer,
                write_ref=args.write_ref,
                formal=False,
                target_root=args.target_root,
            )
    except ModelRegistryError as exc:
        raise SmokeContextError(
            f"tokenizer source lease failed: {exc}"
        ) from exc
    return {
        **locator,
        "query_count": len(query_records),
        "train_record_count": len(train_records),
        "lexicon_term_count": len(terms),
        "ordered_query_ids": [row["id"] for row in query_records],
        "scientific_eligible": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--source-train", type=Path, default=REPOSITORY_ROOT / "data/full/std/train.json"
    )
    value.add_argument(
        "--audit-ref",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/data_audit_ref.json",
    )
    value.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "config/stage1/context_factorial.json"
    )
    value.add_argument(
        "--embedding-model", type=Path, default=REPOSITORY_ROOT / "models/base/bge-large-zh-v1.5"
    )
    value.add_argument(
        "--tokenizer", type=Path, default=REPOSITORY_ROOT / "models/base/Qwen3-8B"
    )
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--query-count", type=int, default=20)
    value.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Explicit pre-outcome infrastructure-eligible dev ID; repeat to freeze order.",
    )
    value.add_argument("--lexicon-terms", type=int, default=128)
    value.add_argument(
        "--prepared-bundle-output",
        type=Path,
        default=REPOSITORY_ROOT
        / "exps/causal_context/stage1_p0/smoke_inputs/prepared_bundle.dev.json",
    )
    value.add_argument(
        "--target-root",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/smoke_contexts",
    )
    value.add_argument(
        "--write-ref",
        type=Path,
        default=REPOSITORY_ROOT / "exps/causal_context/stage1_p0/refs/smoke_context_ref.json",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for path_name in ("source_train", "audit_ref", "config", "embedding_model", "tokenizer"):
        path = getattr(args, path_name).resolve()
        setattr(args, path_name, path)
    try:
        result = build(args)
    except (SmokeContextError, OSError, ValueError, RuntimeError) as exc:
        print(f"[stage1-smoke-context] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
