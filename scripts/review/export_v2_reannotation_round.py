#!/usr/bin/env python3
"""Combine frozen source chunks for one local discussion round."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from export_severity_hate_mapping import checked, mapped_hate

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"


def read(path):
    return json.loads(path.read_bytes())


def reference(path):
    return {"path": str(path.relative_to(RUN)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    number = parser.parse_args().round
    out = RUN / f"discussions/round-{number:02d}"
    selection_path = out / "selection.json"
    selection = read(selection_path)
    assert selection["review_round"] == number
    manifest = read(checked(RUN, selection["initial_input_manifest"]))
    checked(RUN, selection["prior_calibration"])
    for key in ("preference_snapshot", "preference_application"):
        if key in selection:
            checked(RUN, selection[key])
    assert selection["parent_session"] == manifest["parent_session"]
    assert selection["bundle_sha256"] == manifest["bundle_sha256"]
    inputs = []
    for source in selection["source_inputs"]:
        inputs.extend(map(json.loads, checked(RUN, source).read_text().splitlines()))
    assert [row["id"] for row in inputs] == selection["selected_ids"]
    assert len(inputs) == selection["object_count"] == len({row["id"] for row in inputs})
    source_by_id = {row["id"]: row for row in inputs}
    assert set(source_by_id) <= set(manifest["pending_ids"])
    records, sources = [], []
    for batch in selection["source_batches"]:
        path = RUN / f"batches/batch-{batch:02d}/review.json"
        view = read(path)
        assert view["batch"] == batch
        for source in view["source_exports"].values():
            checked(RUN, source)
        for row in view["records"]:
            assert row["text"] == source_by_id[row["object_id"]]["text"]
            assert row["values"]["hate"] == mapped_hate(row["values"]["attack_severity"])
            assert row["severity_annotation"]["base_version"] == manifest["pending_versions"][row["object_id"]]
            assert not row["label_annotation"]["online_material_confirmed"]
            assert not row["severity_annotation"]["online_material_confirmed"]
            records.append({"source_batch": batch, **row})
        sources.append({"source_batch": batch, **reference(path)})
    assert [row["object_id"] for row in records] == selection["selected_ids"]
    pending = [row for row in records if row["pending_tasks"]]
    human_labels = {task: sum(task in row["label_annotation"]["human_adjudicated_tasks"] for row in records)
                    for task in ("hate", "group")}
    human_scores = sum(row["severity_annotation"]["human_numeric_score_confirmed"] for row in records)
    numeric = sum(row["values"]["attack_severity"] is not None for row in records)
    result = {
        "schema_version": "evidence-expanded-discussion-view/v1", "review_round": number,
        "state": "awaiting_discussion" if pending else "discussion_resolved",
        "selection": reference(selection_path), "source_exports": sources,
        "object_count": len(records), "source_batches": selection["source_batches"],
        "discussion_case_count": len(pending), "discussion_case_ids": [r["object_id"] for r in pending],
        "human_label_task_counts": human_labels, "human_numeric_scores_recorded": human_scores,
        "ai_numeric_score_draft_count": numeric - human_scores,
        "null_score_count": len(records) - numeric,
        "score_counts": dict(sorted(Counter(str(r["values"]["attack_severity"]) for r in records).items())),
        "current_values_field": "records[].values", "human_hate_decisions_added_by_mapping": 0,
        "online_material_confirmations_added": 0, "records": records,
    }
    for key in ("preference_snapshot", "preference_application"):
        if key in selection:
            result[key] = selection[key]
    write(out / "review.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["source_batch", "object_id", "text", "hate", "group", "attack_severity",
                     "pending_tasks", "hate_source", "group_source", "severity_source", "online_material_confirmed"])
    for r in records:
        value = r["values"]
        writer.writerow([r["source_batch"], r["object_id"], r["text"], value["hate"] or "null",
                         json.dumps(value["group"], ensure_ascii=False),
                         "null" if value["attack_severity"] is None else value["attack_severity"],
                         ",".join(r["pending_tasks"]), r["field_provenance"]["hate"],
                         r["field_provenance"]["group"], r["field_provenance"]["attack_severity"], False])
    write(out / "review.csv", buffer.getvalue())
    validation_file = "validation-after-decisions-01.json" if not pending and (human_scores or any(human_labels.values())) else "validation.json"
    title = "扩大后的补标讨论" if len(selection["source_batches"]) > 1 else "补标讨论"
    lines = [f"# 第 {number} 轮：{title}", "",
        f"共 {len(records)} 条，来自冻结输入分块 {', '.join(map(str, selection['source_batches']))}；当前 {len(pending)} 条待讨论。输入分块及原正文保持原身份。", "",
        f"本轮已登记用户直接 hate {human_labels['hate']} 条、group {human_labels['group']} 条、严重度数值 {human_scores} 条。另有 {numeric-human_scores} 条 AI 数值、{len(records)-numeric} 条待定分数。疑点处理不等于整批人工审核。", "",
        "当前 hate 继续按 0→non-hate、1–4→hate、null→null 派生。单独给出的分数不额外计为直接 hate 或 group 裁决。没有新增线上材料确认。", "",
        "[JSON](review.json) · [CSV](review.csv) · [范围与扩容依据](selection.json) · [词义查证与限制](sense-checks.json) · " + f"[校验]({validation_file})", "",
        "## 待讨论的完整原文", ""]
    if "preference_snapshot" in selection:
        lines[10:10] = [f"[本轮采用的裁决偏好快照]({RUN / selection['preference_snapshot']['path']}) · [相似案例复核]({out / 'preference-application.json'})", ""]
    for r in pending:
        value = r["values"]
        score = value["attack_severity"]
        group = "待定" if value["group"] is None else "、".join(value["group"]) or "[]"
        s = r["severity_annotation"]
        lines += [f"### #{r['object_id'].split(':')[1]}", "", "> " + r["text"].replace("\n", "\n> "), "",
                  f"当前严重度：{'待定' if score is None else score}；group：{group}。", "", s["values"]["rationale"], ""]
        if s["discussion"]["candidate_scores"]:
            lines += [f"候选分：{s['discussion']['candidate_scores']}；倾向 {s['discussion']['suggested_score']}。倾向未写入活动值。", ""]
    if not pending:
        lines += ["本轮疑点已处理。未获明确裁定的字段仍保留 AI 来源。", ""]
    lines += ["## 全部当前值", "", "| 对象 | 严重度 | hate | group | 待讨论字段 |",
              "| --- | --- | --- | --- | --- |"]
    for r in records:
        v = r["values"]
        g = "待定" if v["group"] is None else "、".join(v["group"]) or "[]"
        lines.append(f"| #{r['object_id'].split(':')[1]} | {v['attack_severity'] if v['attack_severity'] is not None else '待定'} | {v['hate'] or '待定'} | {g} | {', '.join(r['pending_tasks']) or '—'} |")
    write(out / "review.md", "\n".join(lines) + "\n")
    print(json.dumps({"round": number, "objects": len(records), "pending": len(pending),
                      "numeric_scores": numeric, "human_scores": human_scores}, ensure_ascii=False))


if __name__ == "__main__":
    main()
