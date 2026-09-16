"""Read support for explicitly accepted, field-scoped offline final annotations.

The frozen bundle and pre-acceptance AI/human provenance are never rewritten.
Only the importer can create these receipts; the browser cannot submit one.
"""
from __future__ import annotations

import hashlib
import copy
from typing import Any

from build_lex.annotated_lexicon_repair import (
    LexiconRepairError as ReviewError, LexiconRepairConflict as ReviewConflict,
    canonical_sha256, _with_revision,
)
from tools.general_model_paired_review_ui.evidence_schema import normalize_values
from tools.general_model_paired_review_ui.evidence_policy import GROUP_KINDS

SCHEMA = "evidence-final-acceptance/v1"


def current_final(row: dict[str, Any]) -> dict[str, Any] | None:
    final = row.get("final_annotation")
    return final if (final and final["approved_version"] == row["version"]
                     and row["status"] == "confirmed") else None


def validate_finalizations(session: dict[str, Any], objects: dict[str, Any]) -> None:
    batches = session.get("finalizations", {})
    if not isinstance(batches, dict):
        raise ReviewError("最终采纳记录格式不正确。")
    for bid, batch in batches.items():
        if (batch.get("schema_version") != SCHEMA or batch.get("artifact_sha256") != bid
                or batch.get("bundle_sha256") != session["bundle_sha256"]
                or batch.get("reviewer_id") != session["reviewer_id"]
                or not batch.get("accepted_at") or not batch.get("authorization_text", "").strip()
                or batch.get("acceptance_mode") != "explicit_user_bulk_acceptance"
                or not batch.get("object_receipts")
                or not set(batch["object_receipts"]) <= set(objects)):
            raise ReviewError("最终采纳缺少明确授权、来源或对象范围。")
        known_policies = []
        for document in batch.get("policy_documents", []):
            if hashlib.sha256(document["text"].encode()).hexdigest() != document["policy"]["sha256"]:
                raise ReviewError("最终采纳的规则正文哈希不匹配。")
            known_policies.append(document["policy"])
        for oid, expected in batch["object_receipts"].items():
            row = session["objects"][oid]
            final = row.get("final_annotation", {})
            if (canonical_sha256(final) != expected or final.get("batch_id") != bid
                    or final.get("kind") != objects[oid]["kind"]
                    or type(final.get("approved_version")) is not int
                    or not 0 < final["approved_version"] <= row["version"]
                    or any(p not in known_policies for p in final.get("task_policies", {}).values())):
                raise ReviewError("最终采纳的字段、版本或规则来源发生变化。")
            if current_final(row):
                for field, value in final["values"].items():
                    if field != "attack_severity" and row["values"].get(field) != value:
                        raise ReviewError("当前正式字段与批量采纳结果不同。")
                if final["kind"] in {"query", "demo"}:
                    score = final["values"].get("attack_severity")
                    if (type(score) is not int or not 0 <= score <= 4
                            or final["values"]["hate"] != ("hate" if score else "non-hate")
                            or final["values"].get("group") is None):
                        raise ReviewError("最终句子标签缺失或与严重度映射不一致。")
    for row in session["objects"].values():
        if row.get("final_annotation", {}).get("batch_id") not in {None, *batches}:
            raise ReviewError("字段引用了未知的最终采纳批次。")


def prepare_finalized_session(store, previous: dict[str, Any], artifact: dict[str, Any],
                              artifact_sha256: str) -> dict[str, Any]:
    """Pure preparation. Caller must compare exact source bytes under the writer lock."""
    if artifact_sha256 in previous.get("finalizations", {}):
        store._validate_session(previous)
        return copy.deepcopy(previous)
    if (artifact.get("schema_version") != "evidence-final-result/v1"
            or artifact.get("bundle_sha256") != store.bundle_sha256
            or artifact.get("reviewer_id") != store.reviewer_id
            or artifact.get("parent_session", {}).get("revision") != previous["revision"]
            or artifact.get("acceptance_mode") != "explicit_user_bulk_acceptance"
            or not artifact.get("authorization_text", "").strip()):
        raise ReviewConflict("最终结果与授权、复核人或最新权威记录不一致。")
    entries = artifact["records"]
    selected = {entry["object_id"] for entry in entries}
    if (len(selected) != len(entries) or selected != set(artifact["scope_ids"])
            or not selected <= set(store.objects)):
        raise ReviewError("最终结果包含缺失、重复或未授权对象。")
    by_id = {entry["object_id"]: entry for entry in entries}
    for oid, entry in by_id.items():
        obj, row = store.objects[oid], previous["objects"][oid]
        if (row["version"] != entry["base_version"] or row["status"] == "confirmed"
                or canonical_sha256(obj["source"]) != entry["source_sha256"]
                or obj["version"] != entry["source_version"]
                or entry["kind"] != obj["kind"]):
            raise ReviewConflict("写回对象的版本、原文或原确认状态已变化：" + oid)
    session = copy.deepcopy(previous)
    batch = {key: copy.deepcopy(artifact[key]) for key in (
        "bundle_sha256", "reviewer_id", "accepted_at", "authorization_text",
        "acceptance_mode", "policy_documents", "sources", "lexicon_additions")}
    batch.update(schema_version=SCHEMA, artifact_sha256=artifact_sha256,
                 object_receipts={}, prior_session=copy.deepcopy(artifact["parent_session"]),
                 case_assessment_confirmations_added=0)
    session.setdefault("finalizations", {})[artifact_sha256] = batch
    order, done = [], set()
    def visit(oid):
        if oid in done or oid not in selected:
            return
        for dep in store.objects[oid].get("depends_on", []):
            visit(dep)
        done.add(oid)
        order.append(oid)
    for oid in sorted(selected):
        visit(oid)
    for oid in order:
        entry, obj, old = by_id[oid], store.objects[oid], previous["objects"][oid]
        deps = obj.get("depends_on", [])
        dependency_task = "hate" if obj["kind"] == "hit" else None
        if any(not store._object_ready(session, dep, dependency_task) for dep in deps):
            raise ReviewError("最终确认依赖尚未完成：" + oid)
        vals = normalize_values(obj["kind"], entry["native_values"], obj["source"],
                                required=True, allow_policy_changed=store.amended,
                                final_label_only=obj["kind"] in {"query", "demo"})
        final = copy.deepcopy(entry["final_annotation"])
        final.update(batch_id=artifact_sha256, approved_version=old["version"] + 1,
                     kind=obj["kind"], source_record_sha256=canonical_sha256(entry["source_record"]))
        row = session["objects"][oid]
        row.update(values=vals, version=old["version"] + 1, status="confirmed",
                   updated_at=artifact["accepted_at"], confirmed_at=artifact["accepted_at"],
                   reviewer_id=store.reviewer_id, review_kind="human_with_ai",
                   adoption="explicit_final_batch", final_annotation=final,
                   source_version=obj["version"], ai_version=obj["ai_draft"]["version"],
                   dependency_versions={dep: session["objects"][dep]["version"] for dep in deps},
                   dependency_non_group_versions={dep: store._semantic_version(session["objects"][dep], "hate") for dep in deps})
        ai = normalize_values(obj["kind"], obj["ai_draft"]["values"], obj["source"])
        row["field_diff"] = {f: {"ai": ai[f], "human": v} for f, v in vals.items() if ai[f] != v}
        if obj["kind"] in GROUP_KINDS:
            store._update_task_reviews(row, old, confirm=True, group_only=False)
            for task in ("hate", "group"):
                row["task_reviews"][task].update(policy=copy.deepcopy(final["task_policies"][task]),
                                                confirmed_at=artifact["accepted_at"])
        batch["object_receipts"][oid] = canonical_sha256(final)
        session["events"].append({"action": "accept_final_annotation", "at": artifact["accepted_at"],
                                  "reviewer_id": store.reviewer_id, "object_id": oid,
                                  "batch_id": artifact_sha256, "version": row["version"],
                                  "previous_record": copy.deepcopy(old)})
    session["events"].append({"action": "accept_final_batch", "at": artifact["accepted_at"],
                              "reviewer_id": store.reviewer_id, "batch_id": artifact_sha256,
                              "object_count": len(selected), "authorization_text": artifact["authorization_text"]})
    session["updated_at"] = artifact["accepted_at"]
    session = _with_revision(session)
    store._validate_session(session)
    if any(not store._object_ready(session, oid) for oid in selected):
        raise ReviewError("最终写回仍有未确认或过期依赖。")
    if (session["records"] != previous["records"]
            or any(row != previous["objects"][oid] for oid, row in session["objects"].items() if oid not in selected)):
        raise ReviewError("最终写回修改了本轮以外的裁决。")
    return session
