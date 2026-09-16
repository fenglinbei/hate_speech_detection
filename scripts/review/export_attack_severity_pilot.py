#!/usr/bin/env python3
"""Export local severity drafts and explicit score decisions; never write human sessions."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"
STATUS = {"scored": "已试标", "needs_discussion": "边界待讨论", "insufficient_evidence": "证据不足"}
QUESTIONS = {
    "mild_or_explicit_insult": "1／2 级：日常人格贬评与明确侮辱",
    "insult_or_intrinsic_inferiority": "2／3 级：通用辱骂与天生低等类比",
    "violence_stance": "暴力字面表达是否获作者认可",
    "label_interpretation": "需要先明确的对象与作者立场",
    "dehumanization_strength": "动物性称谓：侮辱或严重非人化",
    "self_derogation": "自嘲与轻度贬损的范围",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def checked_ref(base, ref):
    path = base / ref["path"]
    if sha(path) != ref["sha256"]:
        raise ValueError("Frozen reference changed: " + str(path))
    return path


def write_output(path, text):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int)
    mode.add_argument("--batch", type=int, help="Attach severity to an ordinary label batch")
    args = parser.parse_args()
    number = args.pilot if args.pilot is not None else args.batch
    is_pilot = args.pilot is not None
    id_key = "pilot" if is_pilot else "label_batch"
    if number < 1:
        raise ValueError("Batch number must be positive")
    out = RUN / "severity" / (f"pilot-{number:02d}" if is_pilot else f"batch-{number:02d}")
    selection_path = out / "selection.json"
    selection = read_json(selection_path)
    assert selection[id_key] == number
    context = read_json(checked_ref(RUN, selection["context"]))
    policy = read_json(checked_ref(RUN, context["confirmation"]))
    checked_ref(ROOT, policy["document"])
    checked_ref(RUN, policy["accepted_proposal"])
    manifest = read_json(checked_ref(RUN, selection["initial_input_manifest"]))
    if "calibration_reference" in selection:
        checked_ref(RUN, selection["calibration_reference"])
    assert selection["severity_policy"] == context["policy"] == policy["effective_policy"]
    assert selection["parent_session"] == manifest["parent_session"]
    assert selection["bundle_sha256"] == manifest["bundle_sha256"]
    source_rows = list(map(json.loads, checked_ref(out, selection["inputs"]).read_text().splitlines()))
    inputs = {x["object_id"]: x for x in source_rows}
    assert len(inputs) == len(source_rows) == selection["object_count"]
    assert list(inputs) == selection["selected_ids"]
    assert set(inputs) <= set(manifest["pending_ids"])
    for x in source_rows:
        assert hashlib.sha256(x["text"].encode()).hexdigest() == x["text_sha256"]
        assert x["base_version"] == manifest["pending_versions"][x["object_id"]]
    decision_paths = sorted((out / "discussions").glob("user-decisions-[0-9][0-9].json"))
    decisions = {}
    for path in decision_paths:
        event = read_json(path)
        assert event["task_policy"] == context["policy"]
        assert event[id_key] == number and event["selection_sha256"] == sha(selection_path)
        assert event["online_material_confirmations_added"] == 0
        for decision in event["records"]:
            oid = decision["object_id"]
            assert oid in inputs and oid not in decisions, "Use an explicit amendment for repeated score decisions"
            assert decision["text_sha256"] == inputs[oid]["text_sha256"]
            assert decision["base_version"] == inputs[oid]["base_version"]
            assert decision["scope"] == ["attack_severity"]
            assert set(decision["values"]) == {"attack_severity"}
            v = decision["values"]["attack_severity"]
            assert v is None or type(v) is int and v in policy["score_values"]
            assert decision["online_material_confirmation_added"] is False
            decisions[oid] = decision
    annotations_path = RUN / "severity/annotations" / (f"pilot-{number:02d}.json" if is_pilot else f"labels-{number:02d}.json")
    rows = read_json(annotations_path)
    assert len(rows) == len(inputs) and {x["object_id"] for x in rows} == set(inputs)
    assert len({x["object_id"] for x in rows}) == len(rows)
    forms = policy["attack_forms"]
    records = []
    for row in rows:
        source = inputs[row["object_id"]]
        score, status = row["attack_severity"], row["assessment_status"]
        assert status in STATUS
        assert score is None or type(score) is int and score in policy["score_values"]
        assert row["rationale"].strip() and row["quotes"]
        for field in ("attack_forms", "candidate_forms"):
            assert len(row[field]) == len(set(row[field])) and set(row[field]) <= set(forms)
        candidates = row["candidate_scores"]
        assert len(candidates) == len(set(candidates))
        assert all(type(c) is int and c in policy["score_values"] for c in candidates)
        if status == "scored":
            assert score is not None
            assert not candidates and row["suggested_score"] is None and row["discussion_key"] is None
            assert not row["candidate_forms"]
            assert (score == 0) == (not row["attack_forms"])
        else:
            assert score is None
            if status == "needs_discussion":
                assert len(candidates) >= 2 and row["suggested_score"] in candidates
                assert row["discussion_key"] in QUESTIONS
            else:
                assert not candidates and row["suggested_score"] is None and row["discussion_key"] is None
        evidence = []
        for quote in row["quotes"]:
            assert quote and quote in source["text"], (row["object_id"], quote)
            start = source["text"].index(quote)
            evidence.append({"source": "text", "start": start, "end": start + len(quote), "text": quote})
        decision = decisions.get(row["object_id"])
        if decision:
            assert score == decision["values"]["attack_severity"]
            assert status != "needs_discussion"
        records.append({"object_id": row["object_id"], "kind": source["kind"],
            "text": source["text"], "text_sha256": source["text_sha256"], "base_version": source["base_version"],
            "review_kind": "human_with_ai" if decision else "ai_note", "human_confirmed": False,
            "human_adjudicated_tasks": ["attack_severity"] if decision else [], "human_score_decision": decision,
            "human_numeric_score_confirmed": decision is not None and score is not None,
            "online_material_confirmed": False, "task_policy": context["policy"],
            "prior_label_reference": source["prior_label_reference"],
            "field_provenance": {"attack_severity": "user_discussion" if decision else "ai_note", "attack_forms": "ai_note", "supporting_fields": "ai_note"},
            "values": {"attack_severity": score, "attack_forms": row["attack_forms"], "targets": row["targets"],
                "rationale": row["rationale"], "evidence": evidence},
            "assessment_status": status,
            "discussion": {"key": row["discussion_key"], "candidate_scores": candidates,
                "suggested_score": row["suggested_score"], "candidate_forms": row["candidate_forms"]}})
    counts = Counter(r["assessment_status"] for r in records)
    score_counts = Counter(str(r["values"]["attack_severity"]) for r in records if r["assessment_status"] == "scored")
    payload = {"schema_version": "evidence-severity-ai-pilot/v1" if is_pilot else "evidence-severity-ai-label-batch/v1", id_key: number,
        "state": "awaiting_discussion" if counts["needs_discussion"] else "uncertainties_resolved" if decisions else "ai_draft_ready",
        "review_kind": "mixed_discussion" if decisions else "ai_note", "task_policy": context["policy"],
        "selection": {"path": "selection.json", "sha256": sha(selection_path)},
        "annotation_source": {"path": "../annotations/" + annotations_path.name, "sha256": sha(annotations_path)},
        "discussion_provenance": [{"path": str(p.relative_to(out)), "sha256": sha(p)} for p in decision_paths],
        "parent_session": selection["parent_session"], "bundle_sha256": selection["bundle_sha256"],
        "object_count": len(records), "assessment_counts": dict(counts), "score_counts": dict(sorted(score_counts.items())),
        "null_score_count": sum(r["values"]["attack_severity"] is None for r in records),
        "human_numeric_scores_recorded": sum(x["human_numeric_score_confirmed"] for x in records),
        "human_score_decisions_recorded": len(decisions),
        "ai_numeric_score_draft_count": sum(x["assessment_status"] == "scored" and not x["human_score_decision"] for x in records),
        "online_material_confirmations_added": 0,
        "prior_labels_recomputed": False, "independent_annotator_agreement_claimed": False, "records": records}
    write_output(out / "ai_annotations.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["object_id", "text", "attack_severity", "assessment_status", "attack_forms", "targets", "rationale",
        "candidate_scores", "suggested_score", "candidate_forms", "hate_reference", "group_reference", "label_review_kind",
        "severity_review_kind", "severity_policy_version", "severity_policy_sha256", "text_sha256", "base_version",
        "hate_reference_policy_version", "hate_reference_policy_sha256", "group_reference_policy_version", "group_reference_policy_sha256",
        "human_numeric_score_confirmed", "online_material_confirmed", "user_score_rationale"])
    for x in records:
        v, d, ref = x["values"], x["discussion"], x["prior_label_reference"]
        writer.writerow([x["object_id"], x["text"], "null" if v["attack_severity"] is None else v["attack_severity"],
            x["assessment_status"], json.dumps(v["attack_forms"], ensure_ascii=False), json.dumps(v["targets"], ensure_ascii=False),
            v["rationale"], json.dumps(d["candidate_scores"]), "" if d["suggested_score"] is None else d["suggested_score"],
            json.dumps(d["candidate_forms"]), ref["values"]["hate"], json.dumps(ref["values"]["group"], ensure_ascii=False), ref["review_kind"],
            x["review_kind"], context["policy"]["version"], context["policy"]["sha256"], x["text_sha256"], x["base_version"],
            ref["task_policies"]["hate"]["version"], ref["task_policies"]["hate"]["sha256"],
            ref["task_policies"]["group"]["version"], ref["task_policies"]["group"]["sha256"], x["human_numeric_score_confirmed"], False,
            x["human_score_decision"]["rationale"] if x["human_score_decision"] else ""])
    write_output(out / "ai_annotations.csv", buffer.getvalue())
    label_human_count = sum(x["prior_label_reference"]["review_kind"] == "human_with_ai" for x in records)
    title = f"攻击严重度试标第 {number} 批" if is_pilot else f"标签第 {number} 批：攻击严重度"
    lines = [f"# {title}", "",
        f"共 {len(records)} 项：{counts['scored']} 项已有数值（其中 {payload['human_numeric_scores_recorded']} 项用户裁决、{payload['ai_numeric_score_draft_count']} 项 AI 草稿），{counts['needs_discussion']} 项边界待讨论，{counts['insufficient_evidence']} 项证据不足保留 null。没有新增线上材料确认。", "",
        f"已接受试行规则；数值逐条按明确回复记录。所参照 hate/group 中 {label_human_count} 项来自用户裁决，另外 {len(records)-label_human_count} 项为 AI 标签，保持各自来源、版本，未因强度赋分重算。", "",
        "[JSON](ai_annotations.json) · [CSV](ai_annotations.csv) · [选样依据](selection.json) · [规则接受记录](../../policies/attack-severity-v1.json)", "",
        "| 对象 | 严重度 | 状态 | 严重度来源 | 既有 hate/group 来源 |", "| --- | --- | --- | --- | --- |"]
    for x in records:
        score = x["values"]["attack_severity"]
        source = "用户裁决" if x["prior_label_reference"]["review_kind"] == "human_with_ai" else "AI 草稿"
        score_source = "用户裁决" if x["human_score_decision"] else "AI 草稿"
        lines.append(f"| {x['object_id']} | {'待定' if score is None else score} | {STATUS[x['assessment_status']]} | {score_source} | {source} |")
    if decisions:
        lines += ["", "## 已登记的用户严重度裁决", "",
            "仅分数与以下原回复来自用户；形式、对象与证据整理仍是 AI 辅助。", ""]
        for oid, decision in decisions.items():
            lines += [f"- {oid}：{decision['values']['attack_severity']}；{decision['rationale']}"]
        for path in decision_paths:
            lines += ["", f"[裁决来源]({path.relative_to(out)})"]
    lines += ["", "## 本批需要讨论的边界", ""]
    for key, heading in QUESTIONS.items():
        pending = [x for x in records if x["discussion"]["key"] == key]
        if not pending:
            continue
        lines += ["### " + heading, ""]
        for x in pending:
            d = x["discussion"]
            lines += [f"**#{x['object_id'].split(':')[1]}**：候选 {d['candidate_scores']}，倾向 {d['suggested_score']}；当前活动分值为 null。", "",
                "> " + x["text"].replace("\n", "\n> "), "", x["values"]["rationale"], ""]
    lines += ["## 证据不足的项目", ""]
    for x in records:
        if x["assessment_status"] == "insufficient_evidence":
            lines += [f"**#{x['object_id'].split(':')[1]}**", "", "> " + x["text"].replace("\n", "\n> "), "", x["values"]["rationale"], ""]
    lines += ["## 全部试标依据", ""]
    for x in records:
        v = x["values"]
        lines += [f"**#{x['object_id'].split(':')[1]}**：{'待定' if v['attack_severity'] is None else v['attack_severity']}；形式：" +
            ("、".join(forms[f] for f in v["attack_forms"]) or "无"), "", "> " + x["text"].replace("\n", "\n> "), "", v["rationale"], ""]
    write_output(out / "discussion.md", "\n".join(lines) + "\n")
    print(json.dumps({id_key: number, "objects": len(records), "assessment_counts": dict(counts),
        "score_counts": dict(sorted(score_counts.items())), "schema_and_evidence": "passed",
        "human_numeric_scores_recorded": payload["human_numeric_scores_recorded"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
