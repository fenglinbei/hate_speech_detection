#!/usr/bin/env python3
"""Validate and export a discussion batch; never change human sessions."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.general_model_paired_review_ui.evidence_schema import normalize_values

EXP = ROOT / "exps/causal_context/general_model_evidence_applicability_v1"
RUN = EXP / "ai_reviews/v2-reannotation-20260911"
GROUPS = {"R": "Racism", "D": "Region", "L": "LGBTQ", "S": "Sexism", "O": "others"}
ISSUES = {
    "role_group_scope": "社会角色或社群贬损的 hate 范围",
    "individual_scope": "个体辱骂的 hate 范围",
    "missing_context": "省略的命题或事件会改变 hate 判断",
    "missing_referent": "省略对象会改变完整 group 集合",
    "slang_reference": "缩写或隐语所指未确定",
    "stance_irony": "反讽、转述与实际认可未分清",
    "occupation_scope": "学历、职业的类比是否构成实际身份评价",
    "race_self_description": "自称种族歧视是否足以支持本句的种族类别",
    "background_identity": "身份只是来源说明，还是实际评价单位",
    "color_allusion": "颜色类比能否确认指向具体人群身份",
    "individual_group_scope": "纯个人辱骂是否单独计入 others，需对齐个案裁决与 v2 范围",
    "platform_referent": "平台名指内容、机构本身，还是转喻其成员",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    stem = f"labels-{args.batch:02d}"
    input_rows = {row["id"]: row for row in map(json.loads, (RUN / "inputs" / (stem + ".jsonl")).read_text().splitlines())}
    source_path = RUN / "annotations" / (stem + ".json")
    rows = json.loads(source_path.read_bytes())
    assert len(rows) == len(input_rows) and {row[0] for row in rows} == set(input_rows)
    # Current source answers are read only after the completed label draft exists.
    bundle = json.loads((EXP / "bundle/evidence_bundle.json").read_bytes())
    manifest = json.loads((RUN / "input_manifest.json").read_bytes())
    discussion_dir = RUN / "discussions" / f"batch-{args.batch:02d}"
    effective_policy = manifest["policy"]
    policy_context_path = RUN / "policy_context.json"
    policy_application_path = discussion_dir / "policy-application.json"
    policy_context = json.loads(policy_context_path.read_bytes()) if policy_context_path.exists() else None
    if policy_context and args.batch >= policy_context["effective_from_batch"]:
        assert policy_context["base_policy"] == manifest["policy"]
        effective_policy = policy_context["effective_policy"]
        for ref, base in ((policy_context["amendment"], RUN), (policy_context["document"], ROOT)):
            assert hashlib.sha256((base / ref["path"]).read_bytes()).hexdigest() == ref["sha256"]
        assert policy_application_path.exists(), "Apply the confirmed supplement before exporting this batch"
        application = json.loads(policy_application_path.read_bytes())
        assert application["base_policy"] == manifest["policy"]
        assert application["effective_policy"] == effective_policy
    task_policies = {"hate": manifest["policy"], "group": effective_policy}
    # Later explicit decisions can use the approved severity mapping policy,
    # while the retained AI label reference keeps its original task identity.
    mapping_policy = None
    if (RUN / "hate_mapping_context.json").exists():
        from export_severity_hate_mapping import load_mapping
        mapping_policy = load_mapping()[0]["policy"]
    discussion_paths = sorted(discussion_dir.glob("user-decisions-[0-9][0-9].json"))
    decisions = {}
    for discussion_path in discussion_paths:
        discussion = json.loads(discussion_path.read_bytes())
        assert discussion["policy"] == effective_policy
        assert discussion["bundle_sha256"] == manifest["bundle_sha256"]
        for decision in discussion["records"]:
            tasks = set(decision["scope"])
            assert tasks and tasks <= {"hate", "group"} and tasks == set(decision["values"])
            previous = decisions.setdefault(decision["object_id"], [])
            assert not any(tasks & set(d["scope"]) for d in previous), "Use an explicit amendment for repeated task decisions"
            previous.append(decision)
    assert set(decisions) <= set(input_rows)
    calibration_path = discussion_dir / "calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_bytes())
        assert calibration["policy"] == manifest["policy"]
        assert calibration["bundle_sha256"] == manifest["bundle_sha256"]
    records = []
    for oid, hate, group, hr, gr, stance, targets, expressions, quotes, note, uncertain in rows:
        source = input_rows[oid]
        value = {"hate": {"h": "hate", "n": "non-hate", "?": None}[hate],
                 "group": None if group == "?" else [] if group == "-" else [GROUPS[g] for g in group.split(",")],
                 "hate_reason": hr, "group_reason": gr, "stance": stance,
                 "target_types": targets.split(","), "expression_types": expressions.split(","),
                 "note": note, "evidence": []}
        for quote in quotes:
            start = source["text"].find(quote)
            assert start >= 0, (oid, quote)
            value["evidence"].append({"source": "text", "start": start, "end": start + len(quote), "text": quote})
        # Validate independently of original answers before deriving comparisons.
        value = normalize_values("query", value, {"text": source["text"]}, required=True)
        original = bundle["objects"][oid]["source"].get("original_answer")
        comparison = None
        if source["kind"] == "demo":
            assert original is not None
            comparison = {}
            for task in ("hate", "group"):
                if value[task] is None:
                    reason = value[task + "_reason"]
                    status = "policy_ambiguous" if reason in {"category_boundary", "individual_scope", "institution_scope"} else "context_insufficient"
                elif value[task] == original[task]:
                    status = "accepted"
                elif task == "group" and (value[task] == [] or "others" in value[task]):
                    status = "policy_changed"
                else:
                    status = "suspected_error"
                comparison[task + "_original_status"] = status
            value.update(comparison)
            value = normalize_values("demo", value, bundle["objects"][oid]["source"], required=True, allow_policy_changed=True)
        source_decisions = decisions.get(oid, [])
        record_task_policies = dict(task_policies)
        human_tasks = [t for t in ("hate", "group") if any(t in d["scope"] for d in source_decisions)]
        for d in source_decisions:
            assert d["text_sha256"] == hashlib.sha256(source["text"].encode()).hexdigest()
            assert d["base_version"] == manifest["pending_versions"][oid]
            assert all(value[task] == d["values"][task] for task in d["scope"]), oid
            declared = d.get("task_policies", {})
            assert set(declared) <= set(d["scope"])
            for task, policy in declared.items():
                allowed = [task_policies[task]] + ([mapping_policy] if task == "hate" and mapping_policy else [])
                assert policy in allowed, (oid, task, "Unrecognized decision policy")
                record_task_policies[task] = policy
        assert not any(u.split(":", 1)[0] in human_tasks for u in uncertain), oid
        decision = source_decisions[0] if len(source_decisions) == 1 else None
        if len(source_decisions) > 1:
            decision = {"object_id": oid, "values": {t: value[t] for t in human_tasks}, "scope": human_tasks,
                "rationale": "；".join(d["rationale"] for d in source_decisions),
                "rationale_provenance": "assembled_from_separate_verbatim_task_decisions", "source_decisions": source_decisions}
        records.append({"object_id": oid, "kind": source["kind"], "text": source["text"],
            "review_kind": "human_with_ai" if decision else "ai_note",
            "human_confirmed": False, "online_material_confirmed": False,
            "human_adjudicated_tasks": human_tasks,
            "human_decision": decision,
            "field_provenance": {"hate": "user_discussion" if "hate" in human_tasks else "ai_note",
                "group": "user_discussion" if "group" in human_tasks else "ai_note",
                "supporting_fields": "ai_note", "original_statuses": "ai_comparison_suggestion"},
            "policy": effective_policy, "task_policies": record_task_policies,
            "label_resolution": {task: "resolved" if value[task] is not None else "unresolved" for task in ("hate", "group")},
            "values": value, "uncertainties": [{"task": u.split(":", 1)[0], "code": u.split(":", 1)[1],
                "description": ISSUES.get(u.split(":", 1)[1], u)} for u in uncertain],
            "comparison_provenance": {"method": "suggestion_after_completed_label_draft",
                "requires_human_judgment": True} if comparison else None})
    output = RUN / "batches" / f"batch-{args.batch:02d}"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "evidence-ai-discussion-batch/v1", "batch": args.batch,
        "state": "awaiting_discussion" if any(r["uncertainties"] for r in records) else "uncertainties_resolved" if decisions else "drafted",
        "review_kind": "mixed_discussion" if decisions else "ai_note", "human_confirmations_added": 0,
        "human_label_decisions_recorded": len(decisions),
        "human_label_task_counts": {t: sum(t in r["human_adjudicated_tasks"] for r in records) for t in ("hate", "group")},
        "human_complete_label_decisions_recorded": sum(r["human_adjudicated_tasks"] == ["hate", "group"] for r in records),
        "human_confirmation_scope": "online full-material confirmations; discussion label decisions are counted separately",
        "bundle_sha256": manifest["bundle_sha256"], "base_policy": manifest["policy"],
        "policy": effective_policy, "task_policies": task_policies,
        "parent_session": manifest["parent_session"],
        "annotation_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "human_reference": "../../provenance/human-reference.json", "blind_review_claimed": False,
        "discussion_provenance": [{"path": "../../" + str(p.relative_to(RUN)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in discussion_paths],
        "calibration_provenance": {"path": "../../" + str(calibration_path.relative_to(RUN)),
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest()} if calibration_path.exists() else None,
        "policy_application_provenance": {"path": "../../" + str(policy_application_path.relative_to(RUN)),
            "sha256": hashlib.sha256(policy_application_path.read_bytes()).hexdigest()} if policy_application_path.exists() else None,
        "object_count": len(records), "uncertain_count": sum(bool(r["uncertainties"]) for r in records),
        "unresolved_label_count": sum(any(r["values"][task] is None for task in ("hate", "group")) for r in records),
        "records": records}
    (output / "ai_annotations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow(["object_id", "review_kind", "text", "hate", "group", "uncertainties", "note",
        "human_adjudicated_tasks", "user_rationale", "online_material_confirmed",
        "hate_policy_version", "hate_policy_sha256", "group_policy_version", "group_policy_sha256"])
    for row in records:
        v = row["values"]
        writer.writerow([row["object_id"], row["review_kind"], row["text"], v["hate"] or "null",
            json.dumps(v["group"], ensure_ascii=False), "；".join(u["description"] for u in row["uncertainties"]), v["note"],
            ",".join(row["human_adjudicated_tasks"]),
            row["human_decision"]["rationale"] if row["human_decision"] else "", False,
            row["task_policies"]["hate"]["version"], row["task_policies"]["hate"]["sha256"],
            row["task_policies"]["group"]["version"], row["task_policies"]["group"]["sha256"]])
    (output / "ai_annotations.csv").write_text(buffer.getvalue())
    title_state = "待讨论" if payload["uncertain_count"] else "疑点已处理" if decisions else "草稿已整理"
    policy_heading = "v2（含个人辱骂增补）" if effective_policy != manifest["policy"] else "v2"
    lines = [f"# {policy_heading}补标第 {args.batch} 批：{title_state}", "", f"本批 {len(records)} 项，其中 {payload['uncertain_count']} 项有待讨论疑点；{payload['unresolved_label_count']} 项仍有 null 标签。已登记 {len(decisions)} 项的用户标签字段裁决（hate {payload['human_label_task_counts']['hate']} 项、group {payload['human_label_task_counts']['group']} 项）。尚未导入线上，没有新增线上整项确认。", "",
        "已参考用户允许使用的既有人工裁决，来源身份见独立 reference 清单。原答案对照建议在本批标签写完后另行形成，不代表已经确认源数据错误。", "",
        "[完整 JSON](ai_annotations.json) · [CSV](ai_annotations.csv)", ""]
    if (RUN / "hate_mapping_context.json").exists() and args.batch >= 4:
        lines += ["本文件是标签参照层，保留显式裁决与 AI 标签草稿及其原政策。当前 hate 已改用严重度默认映射，请使用[合并讨论表](review.md)、[当前 JSON](review.json)或[当前 CSV](review.csv)。本层原答案对照状态不自动转为新映射下的源数据结论。", ""]
    if effective_policy != manifest["policy"]:
        lines += ["本批 group 采用用户确认的个人辱骂计入 others 增补规则；hate 参照层默认沿用父版，用户显式裁决另有政策声明时按逐条记录保留。", ""]
    if decisions:
        lines += ["## 本次用户裁决", "",
            "仅明确给出的标签字段及下列用户理由来自裁决；未明确给出的另一字段仍为 AI 草稿。辅助选项、证据高亮和原答案对照状态仍为 AI 整理，没有代替完整材料确认。", "",
            "| 对象 | hate | group | 用户理由／采纳范围 |", "| --- | --- | --- | --- |"]
        for record in records:
            decision = record["human_decision"]
            if not decision:
                continue
            oid = record["object_id"]
            v = decision["values"]
            reason = decision["rationale"].replace("|", "\\|").replace("\n", " ")
            group_text = json.dumps(v["group"], ensure_ascii=False) if "group" in v else "未明确裁决"
            lines += [f"| {oid} | {v.get('hate', '未明确裁决')} | {group_text} | {reason} |"]
        lines += [""]
        for index, source in enumerate(payload["discussion_provenance"], 1):
            lines += [f"[用户裁决记录 {index}]({source['path']})", ""]
    else:
        lines += ["现有人工裁决作为逐例参照，不把单例自动推广为新政策，也不改写原人工记录。", ""]
    if (discussion_dir / "followup.md").exists():
        lines += ["[补充核查与校准说明](../../discussions/" + f"batch-{args.batch:02d}" + "/followup.md)", ""]
    lines += ["## 本批待讨论项", ""]
    if not payload["uncertain_count"]:
        lines += ["本批原有疑点已处理。未被用户明确裁定的其他记录仍是 AI 草稿。", ""]
    for row in records:
        if not row["uncertainties"]:
            continue
        v = row["values"]
        lines += ["### " + row["object_id"], "", "> " + row["text"].replace("\n", "\n> "), "",
            "当前 AI 草稿：hate = " + str(v["hate"] or "未决") + "；group = " + ("未决" if v["group"] is None else json.dumps(v["group"], ensure_ascii=False)), "",
            "疑点：" + "；".join(u["description"] for u in row["uncertainties"]), "", v["note"], ""]
    (output / "discussion.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"batch": args.batch, "objects": len(records), "uncertain": payload["uncertain_count"], "schema_and_evidence": "passed", "output": str(output)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
