#!/usr/bin/env python3
"""Freeze an explicitly accepted result and prepare a lossless authoritative writeback.

Building never writes a human session. Applying is a separate, version-checked
operation for the stopped authoritative writer (or an isolated test session).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from build_lex.annotated_lexicon_repair import (
    _session_lock, canonical_sha256, read_json, write_json,
    LexiconRepairConflict as ReviewConflict,
)
from tools.general_model_paired_review_ui.evidence_schema import empty_values, normalize_values
from tools.general_model_paired_review_ui.evidence_finalization import prepare_finalized_session
from tools.general_model_paired_review_ui.evidence_policy import _backup_exact, policy_ref
from tools.general_model_paired_review_ui.evidence_store import EvidenceReviewStore

RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"
BUNDLE = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/bundle/evidence_bundle.json"
POLICY = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/policies/group-scope-v2/policy_amendment.json"


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(run, ref):
    path = (run / ref["path"]).resolve()
    assert path.is_relative_to(run.resolve()) and sha(path.read_bytes()) == ref["sha256"], ref
    return read_json(path)


def build(run, authorization, bundle_path=BUNDLE, policy_path=POLICY):
    manifest = read_json(run / "input_manifest.json")
    bundle = read_json(bundle_path)
    policy = read_json(policy_path)
    assert sha(bundle_path.read_bytes()) == manifest["bundle_sha256"]
    assert authorization["acceptance_mode"] == "explicit_user_bulk_acceptance"
    assert authorization["authorization_text"].strip() and authorization["reviewer_id"]
    assert authorization["scope_ids"] == manifest["pending_ids"]
    for name, expected in manifest["inputs"].items():
        assert sha((run / "inputs" / name).read_bytes()) == expected, name
    refs, records = [], []
    doc = read_json(run / "sentence-completion-v1/current.json")
    assert doc["complete"] is True
    ref = doc["artifacts"]["current-sentences.json"]
    refs.append(ref)
    sentences = checked(run, ref)["records"]
    for kind in ("definitions", "hits", "relations"):
        pointer = read_json(run / "resource-reviews-v1" / kind / "current.json")
        ref = pointer["artifacts"]["reviews.json"]
        refs.append(ref)
        for original in checked(run, ref)["records"]:
            row = copy.deepcopy(original)
            row["kind"] = kind[:-1] if kind != "hits" else "hit"
            records.append((row, ref, original))
    records += [(r, refs[0], r) for r in sentences]
    policy_docs = [{"policy": policy_ref(policy), "path": str(policy_path.relative_to(ROOT)),
                    "text": policy["document_text"]}]
    supplements = []
    for path in sorted((run / "policies").glob("*.json")):
        supplement = read_json(path)
        supplements.append(supplement)
        document = supplement["document"]
        raw = (ROOT / document["path"]).read_bytes()
        assert sha(raw) == document["sha256"]
        effective = supplement.get("effective_policy", supplement.get("policy"))
        assert effective["sha256"] == sha(raw)
        policy_docs.append({"policy": effective, "path": document["path"], "text": raw.decode()})
    entries = []
    for row, ref, original in records:
        oid, kind = row["object_id"], row["kind"]
        obj = bundle["objects"][oid]
        assert kind == obj["kind"] and row["base_version"] == manifest["pending_versions"][oid]
        final = {"pre_acceptance_field_provenance": copy.deepcopy(row.get("field_provenance", {})),
                 "source_snapshot": ref, "task_policies": {},
                 "supporting_fields_provenance": "ai_adapter_or_retained_local_notes"}
        if kind in {"query", "demo"}:
            assert row["text"] == obj["source"]["text"]
            native = empty_values(kind)
            native.update({key: row["values"][key] for key in ("hate", "group")})
            # Local sentence review decided these three fields, not source-label
            # error classifications or stance/target subfields from the old UI.
            native.update(hate_reason="other", group_reason="other")
            native["note"] = ("本轮最终裁决：严重度 " + str(row["values"]["attack_severity"])
                              + "；0 对应 non-hate，非 0 对应 hate。完整依据与逐字段历史保存在最终采纳记录。"
                              + ("本轮未新增原标注对错裁决。" if kind == "demo" else ""))
            final["values"] = copy.deepcopy(row["values"])
            final["task_policies"] = copy.deepcopy(row["task_policies"])
            final["unreviewed_supporting_fields"] = ["stance", "target_types", "expression_types"]
            if kind == "demo":
                final["unreviewed_supporting_fields"] += ["hate_original_status", "group_original_status"]
        elif kind == "definition":
            native = copy.deepcopy(row["ai_draft"]["values"])
            native["definition_verdict"] = row["definition_verdict"]
            human = row["human_fields"].get("definition_verdict")
            if human:
                native["note"] = human["rationale_verbatim"]
                native["issues"] = (["valid_sense"] if row["definition_verdict"] == "reasonable" else
                                    ["overly_narrow_definition"] if row["definition_verdict"] == "too_narrow" else [])
            final["values"] = {"definition_verdict": row["definition_verdict"]}
            final["pre_acceptance_field_provenance"] = {"definition_verdict": row["verdict_provenance"]}
            final["adopted_definition"] = row["adopted_definition"] or row.get("instruction_applied_definition_ai", {}).get("definition_text")
            final["definition_wording_provenance"] = ("user_discussion" if row["adopted_definition"] else
                                                       "ai_instruction_application" if final["adopted_definition"] else None)
            final["definition_rewrite_instruction"] = row["rewrite_instruction"]
        else:
            native = copy.deepcopy(row["values"])
            final["values"] = {key: native[key] for key in (
                ("source_fit", "query_fit") if kind == "hit" else
                ("topic_hate", "topic_group", "rule_hate", "rule_group", "lexicon_risk"))}
            if kind == "relation":
                final["task_policies"] = {task: policy_ref(policy) for task in ("hate", "group")}
        normalized = normalize_values(kind, native, obj["source"], required=True,
                                      allow_policy_changed=True, final_label_only=kind in {"query", "demo"})
        if "group" in final["values"]:
            final["values"]["group"] = normalized["group"]
        entries.append({"object_id": oid, "kind": kind, "base_version": row["base_version"],
                        "source_sha256": canonical_sha256(obj["source"]), "source_version": obj["version"],
                        "source_record": copy.deepcopy(original), "native_values": normalized,
                        "final_annotation": final})
    assert len(entries) == 954 and {e["object_id"] for e in entries} == set(manifest["pending_ids"])
    artifact = {"schema_version": "evidence-final-result/v1", **copy.deepcopy(authorization),
                "bundle_sha256": manifest["bundle_sha256"], "parent_session": manifest["parent_session"],
                "sources": refs, "policy_documents": policy_docs, "policy_supplements": supplements,
                "lexicon_additions": [read_json(run / "resource-reviews-v1/lexicon-additions/female-circle-v1.json")],
                "scope_note": "954 material objects and the recorded lexicon supplement; separate case-level Gold/error/use assessments are not newly adjudicated.",
                "records": sorted(entries, key=lambda e: e["object_id"])}
    return artifact


def export_csv(artifact):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=["object_id", "kind", "hate", "group", "attack_severity",
        "definition_verdict", "adopted_definition", "source_fit", "query_fit", "topic_hate", "topic_group",
        "rule_hate", "rule_group", "lexicon_risk", "acceptance_mode", "pre_acceptance_field_provenance", "source_snapshot"])
    writer.writeheader()
    for entry in artifact["records"]:
        final = entry["final_annotation"]
        row = {"object_id": entry["object_id"], "kind": entry["kind"], **final["values"],
               "adopted_definition": final.get("adopted_definition"), "acceptance_mode": artifact["acceptance_mode"],
               "pre_acceptance_field_provenance": final["pre_acceptance_field_provenance"], "source_snapshot": final["source_snapshot"]}
        row = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v for k, v in row.items()})
    return ("\ufeff" + out.getvalue()).encode()


def apply(store, artifact_path, backup_path, expected_sha256):
    raw = artifact_path.read_bytes()
    artifact = json.loads(raw)
    with _session_lock(store.session_path):
        previous = store._read()
        current = store.session_path.read_bytes()
        if sha(raw) in previous.get("finalizations", {}):
            return {"already_applied": True, "revision": previous["revision"]}
        if sha(current) != expected_sha256 or artifact["parent_session"]["sha256"] != expected_sha256:
            raise ReviewConflict("权威记录已经变化；未执行写回。")
        result = prepare_finalized_session(store, previous, artifact, sha(raw))
        if backup_path.resolve() in {store.session_path, store.bundle_path, artifact_path.resolve()}:
            raise ValueError("Backup path overlaps source")
        _backup_exact(backup_path, current)
        write_json(store.session_path, result)
        return {"already_applied": False, "revision": result["revision"],
                "sha256": sha(store.session_path.read_bytes()), "final_artifact_sha256": sha(raw),
                "new_confirmations": len(artifact["records"]), "status": store._bootstrap(result)["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run", type=Path, default=RUN)
    b.add_argument("--authorization", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("apply")
    for name in ("bundle", "policy", "session", "artifact", "backup"):
        a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--reviewer-id", required=True)
    a.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.run, read_json(args.authorization))
        raw = encoded(result)
        args.output.mkdir(parents=True, exist_ok=True)
        for name, data in (("final-result.json", raw), ("final-result.csv", export_csv(result))):
            path = args.output / name
            if path.exists():
                assert path.read_bytes() == data, "Refusing to overwrite a different final artifact"
            else:
                path.write_bytes(data)
        print(json.dumps({"objects": len(result["records"]), "sha256": sha(raw)}, ensure_ascii=False))
    else:
        store = EvidenceReviewStore(bundle_path=args.bundle, policy_path=args.policy,
                                    session_path=args.session, reviewer_id=args.reviewer_id)
        print(json.dumps(apply(store, args.artifact, args.backup, args.expected_sha256), ensure_ascii=False))


if __name__ == "__main__":
    main()
