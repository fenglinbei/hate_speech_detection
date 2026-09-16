#!/usr/bin/env python3
"""Join local label and severity exports into one discussion view."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from export_severity_hate_mapping import build_mapping, load_mapping, load_group_priority, apply_group_priority

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True)
    batch = parser.parse_args().batch
    out = RUN / "batches" / f"batch-{batch:02d}"
    lp = out / "ai_annotations.json"
    sp = RUN / "severity" / f"batch-{batch:02d}" / "ai_annotations.json"
    labels, severity = (json.loads(p.read_bytes()) for p in (lp, sp))
    mapping_context, mapping_ref = load_mapping()
    group_applications, group_sources = load_group_priority()
    refs = {k: {"path": str(p.relative_to(RUN)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for k, p in (("labels", lp), ("severity", sp))}
    for index, source in enumerate(group_sources):
        refs[f"group_priority_{index}"] = source
    assert labels["batch"] == severity["label_batch"] == batch
    assert labels["parent_session"] == severity["parent_session"]
    assert labels["bundle_sha256"] == severity["bundle_sha256"]
    sr = {x["object_id"]: x for x in severity["records"]}
    assert len(sr) == len(severity["records"]) == len(labels["records"])
    assert set(sr) == {x["object_id"] for x in labels["records"]}
    records = []
    for l in labels["records"]:
        s = sr[l["object_id"]]
        assert l["text"] == s["text"]
        assert hashlib.sha256(l["text"].encode()).hexdigest() == s["text_sha256"]
        # Hate is derived from the active score. Its old independent doubts remain
        # in the label layer; only unresolved score semantics can block this view.
        pending = list(dict.fromkeys(x["task"] for x in l["uncertainties"] if x["task"] != "hate"))
        if s["assessment_status"] == "needs_discussion":
            pending.append("attack_severity")
        mapped = build_mapping(l, s, mapping_context, refs["labels"], refs["severity"])
        apply_group_priority(mapped, l, group_applications)
        records.append({"object_id": l["object_id"], "text": l["text"], "pending_tasks": pending,
            "values": mapped["values"], "field_provenance": mapped["field_provenance"],
            "task_policies": mapped["task_policies"], "hate_mapping": mapped,
            "label_annotation": l, "severity_annotation": s,
            "severity_label_reference_matches_current_values": s["prior_label_reference"]["values"] ==
                {t: mapped["values"][t] for t in ("hate", "group")},
            "severity_label_reference_matches_label_layer_values": s["prior_label_reference"]["values"] ==
                {t: l["values"][t] for t in ("hate", "group")}})
    pending = [x for x in records if x["pending_tasks"]]
    result = {"schema_version": "evidence-label-severity-discussion-view/v2", "batch": batch,
        "state": "awaiting_discussion" if pending else "discussion_resolved",
        "mapping_context": mapping_ref, "current_values_field": "records[].values",
        "human_hate_decisions_added_by_mapping": 0,
        "source_exports": refs, "object_count": len(records), "discussion_case_count": len(pending),
        "discussion_case_ids": [x["object_id"] for x in pending],
        "hate_group_uncertain_case_count": sum("group" in x["pending_tasks"] for x in records),
        "unmapped_hate_count": sum(x["values"]["hate"] is None for x in records),
        "severity_needs_discussion_count": severity["assessment_counts"].get("needs_discussion", 0),
        "severity_insufficient_evidence_count": severity["assessment_counts"].get("insufficient_evidence", 0),
        "human_label_decisions_recorded": labels["human_label_decisions_recorded"],
        "human_label_task_counts": labels.get("human_label_task_counts", {t: sum(t in x["human_adjudicated_tasks"] for x in labels["records"]) for t in ("hate", "group")}),
        "human_numeric_scores_recorded": severity["human_numeric_scores_recorded"],
        "online_material_confirmations_added": 0, "records": records}
    (out / "review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow(["object_id", "text", "hate", "group", "attack_severity", "pending_tasks", "severity_status",
        "label_review_kind", "severity_review_kind", "label_note", "severity_rationale", "severity_candidates",
        "severity_suggested_score", "hate_policy", "group_policy", "severity_policy",
        "severity_label_reference_matches_current_values", "online_material_confirmed",
        "hate_source", "group_source", "severity_source", "historical_hate", "historical_hate_policy"])
    for x in records:
        l, s = x["label_annotation"], x["severity_annotation"]
        writer.writerow([x["object_id"], x["text"], x["values"]["hate"] or "null", json.dumps(x["values"]["group"]),
            "null" if s["values"]["attack_severity"] is None else s["values"]["attack_severity"], ",".join(x["pending_tasks"]),
            s["assessment_status"], l["review_kind"], s["review_kind"], l["values"]["note"], s["values"]["rationale"],
            json.dumps(s["discussion"]["candidate_scores"]), s["discussion"]["suggested_score"],
            json.dumps(x["task_policies"]["hate"]), json.dumps(x["task_policies"]["group"]), json.dumps(s["task_policy"]),
            x["severity_label_reference_matches_current_values"], False,
            *[x["field_provenance"][t] for t in ("hate", "group", "attack_severity")],
            l["values"]["hate"] or "null", json.dumps(l["task_policies"]["hate"])])
    (out / "review.csv").write_text(buffer.getvalue())
    lines = [f"# 第 {batch} 批：标签与攻击严重度", "",
        f"本批 {len(records)} 条，{len(pending)} 条需要讨论。其中 group 有 {result['hate_group_uncertain_case_count']} 条疑点；严重度 {severity['assessment_counts'].get('scored',0)} 条已有数值，{result['severity_needs_discussion_count']} 条随语义或等级边界讨论，另 {result['severity_insufficient_evidence_count']} 条证据不足保留待定。", "",
        f"本批 {result['human_label_decisions_recorded']} 条有用户标签字段裁决（hate {result['human_label_task_counts']['hate']} 条、group {result['human_label_task_counts']['group']} 条），用户严重度数值 {result['human_numeric_scores_recorded']} 条；其余语义判断为 AI 草稿，hate 另记分数映射来源。没有新增线上整项材料确认。", "",
        "当前 hate 按用户暂行映射生成：0→non-hate，1–4→hate，null→null。JSON 的 records[].values 和 CSV 是当前结果；label_annotation 与 prior_label_reference 保留各阶段标签及来源。用户分数的映射也不新增一次直接 hate 裁决。", "",
        "[合并 JSON](review.json) · [合并 CSV](review.csv) · [标签记录](discussion.md) · " +
            f"[严重度记录](../../severity/batch-{batch:02d}/discussion.md)", "",
        "严重度使用已接受的 0–4 级试行规则；#309＝1、#2750＝3、#2180＝2、#2800＝3 是用户给出的本轮校准锚点。", ""]
    applied = [r["object_id"] for r in records if "group_priority" in r["hate_mapping"]]
    if applied:
        lines += ["本批 AI 草稿另按新合并优先级更新当前 group：" + "、".join(applied) +
                  "。原标签保留；规则应用不新增人工 group 裁决。[规则应用](../../group-priority-v1/README.md)", ""]
    confirmed = [x for x in records if x["label_annotation"]["human_adjudicated_tasks"] or x["severity_annotation"]["human_adjudicated_tasks"]]
    if confirmed:
        lines += ["## 已登记的用户字段", "", "仅下表标明的字段计作用户直接裁决；分数映射另记来源，其余语义判断保持 AI 来源。", "",
            "| 对象 | hate | group | 严重度 | 用户裁决字段 |", "| --- | --- | --- | --- | --- |"]
        for x in confirmed:
            l, s = x["label_annotation"], x["severity_annotation"]
            lt, st = l["human_adjudicated_tasks"], s["human_adjudicated_tasks"]
            h = str(x["values"]["hate"] or "待定") + ("（另有明确裁决）" if x["hate_mapping"]["compatible_explicit_human_hate"] else "（分数映射）")
            g = "待定" if l["values"]["group"] is None else "、".join(l["values"]["group"]) or "空"
            g += "" if "group" in lt else "（AI）"
            score = s["values"]["attack_severity"]
            score_text = ("待定" if score is None else str(score)) + ("" if "attack_severity" in st else "（AI）")
            lines += [f"| {x['object_id']} | {h} | {g} | {score_text} | {', '.join(lt + st)} |"]
        lines += [""]
    lines += ["## 待讨论项目", ""]
    for x in pending:
        l, s = x["label_annotation"], x["severity_annotation"]
        lv, sv, sd = {**l["values"], **x["values"]}, s["values"], s["discussion"]
        group = "待定" if lv["group"] is None else "、".join(lv["group"]) or "空"
        lines += ["### #" + x["object_id"].split(":")[1], "", "> " + x["text"].replace("\n", "\n> "), "",
            f"当前标签草稿：{lv['hate'] or '待定'} / {group}。严重度：{'待定' if sv['attack_severity'] is None else sv['attack_severity']}。", "",
            lv["note"] if l["uncertainties"] else "group 本轮没有待讨论疑点；hate 由活动严重度映射。", "",
            sv["rationale"], ""]
        if sd["candidate_scores"]:
            lines += [f"严重度候选：{sd['candidate_scores']}；倾向 {sd['suggested_score']}，倾向分未写入活动分值。", ""]
    lines += ["## 证据不足而暂不要求补猜", ""]
    for x in records:
        s = x["severity_annotation"]
        if s["assessment_status"] == "insufficient_evidence":
            lines += ["**#" + x["object_id"].split(":")[1] + "**", "", "> " + x["text"].replace("\n", "\n> "), "", s["values"]["rationale"], ""]
    (out / "review.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"batch": batch, "objects": len(records), "discussion_cases": len(pending),
        "discussion_case_ids": result["discussion_case_ids"], "output": str(out / "review.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
