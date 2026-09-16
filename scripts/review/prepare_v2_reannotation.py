#!/usr/bin/env python3
"""Freeze unreviewed material inputs for the requested v2 AI second pass.

Reads an explicit private authoritative snapshot. Never writes human records or
exports original query answers, model results, or old AI labels into inputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore

EXP = ROOT / "exps/causal_context/general_model_evidence_applicability_v1"
OUT = EXP / "ai_reviews/v2-reannotation-20260911"

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Refusing to replace a frozen preparation artifact: " + str(path))
    if not path.exists():
        path.write_bytes(raw)

def jsonl(path, rows):
    write(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    store = EvidenceReviewStore(bundle_path=EXP / "bundle/evidence_bundle.json",
        session_path=args.snapshot, reviewer_id=snapshot["reviewer_id"],
        policy_path=EXP / "policies/group-scope-v2/policy_amendment.json")
    assert args.snapshot.read_bytes() == raw
    pending = [oid for oid in store.objects if not store._object_ready(snapshot, oid)]
    # This snapshot's entire remaining queue is untouched. If a later snapshot
    # differs, prepare a separate run with explicit per-task preservation rules.
    assert all(snapshot["objects"][oid]["status"] == "unreviewed" and
               snapshot["objects"][oid]["values"] is None for oid in pending)
    label_rows = []
    for oid in pending:
        obj = store.objects[oid]
        if obj["kind"] in {"query", "demo"}:
            label_rows.append({"id": oid, "kind": obj["kind"], "text": obj["source"]["text"]})
    for index in range(0, len(label_rows), 30):
        jsonl(OUT / "inputs" / f"labels-{index//30+1:02d}.jsonl", label_rows[index:index+30])
    definitions = [{"id": oid, "kind": "definition", "source": store.objects[oid]["source"]}
                   for oid in pending if store.objects[oid]["kind"] == "definition"]
    jsonl(OUT / "inputs/definitions.jsonl", definitions)
    resource_cases = []
    for key in store.order:
        case = store.cases[key]
        required = [store.objects[oid] for oid in case["object_ids"] if oid in pending
                    and store.objects[oid]["kind"] in {"relation", "hit"}]
        if not required:
            continue
        resources = {"query_id": key, "query": store.objects[case["query_object_id"]]["source"]["text"],
            "demos": {}, "definitions": {}, "required": []}
        for oid in case["object_ids"]:
            obj = store.objects[oid]; source = obj["source"]
            if obj["kind"] == "demo":
                resources["demos"][source["demo_id"]] = source["text"]
            if obj["kind"] == "definition":
                resources["definitions"][source["sense_id"]] = {"term": source["term"],
                    "text": source["text"], "categories": source["categories"]}
        for obj in required:
            resources["required"].append({"id": obj["id"], "kind": obj["kind"],
                "source": {k: v for k, v in obj["source"].items() if k in
                {"demo_id", "introduced_entry_ids", "source_kind", "source_id", "sense_id",
                 "raw_span", "raw_surface", "entry_id", "rendered_in", "provenance_status"}}})
        resource_cases.append(resources)
    for index, case in enumerate(resource_cases, 1):
        write(OUT / "inputs" / f"resources-{index:02d}-{case['query_id']}.json", encoded(case))
    manifest = {"schema_version": "evidence-v2-ai-reannotation-input/v1",
        "review_kind": "ai_note", "user_request": "按照目前v2规则再次标注未完成项，记录不确定项并沟通",
        "parent_session": {"sha256": sha(raw), "revision": snapshot["revision"]},
        "bundle_sha256": store.bundle_sha256,
        "policy": {k: store.policy[k] for k in ("version", "sha256")},
        "pending_ids": pending,
        "pending_versions": {oid: snapshot["objects"][oid]["version"] for oid in pending},
        "counts": dict(Counter(store.objects[oid]["kind"] for oid in pending)),
        "preserved_current_human_confirmations": len(store.objects) - len(pending),
        "human_confirmations_added": 0,
        "scope": "authorized discovery material objects only; no final case adjudication",
        "provenance_limits": {"existing_conversation_exposure": True,
            "blind_review_claimed": False, "model_revision": "not_exposed",
            "temperature": "not_exposed", "initial_label_inputs_exclude_answers_predictions": True,
            "labels_grouped_in_batches": True},
        "inputs": {p.name: sha(p.read_bytes()) for p in sorted((OUT / "inputs").iterdir())}}
    write(OUT / "input_manifest.json", encoded(manifest))
    print(json.dumps({"pending": len(pending), "counts": manifest["counts"],
                      "input_files": len(manifest["inputs"]), "output_dir": str(OUT)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
