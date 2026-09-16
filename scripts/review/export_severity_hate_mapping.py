#!/usr/bin/env python3
"""Export the approved local severity-to-hate view without rewriting source records."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"


def read(path):
    return json.loads(path.read_bytes())


def ref(path, base=RUN):
    return {"path": str(path.relative_to(base)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(base, reference):
    path = base / reference["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
        raise ValueError("Changed mapping reference: " + str(path))
    return path


def load_mapping():
    context_path = RUN / "hate_mapping_context.json"
    context = read(context_path)
    confirmation = read(checked(RUN, context["confirmation"]))
    checked(ROOT, context["document"])
    assert context["policy"] == confirmation["effective_policy"]
    assert context["document"] == confirmation["document"]
    assert context["policy"]["sha256"] == context["document"]["sha256"]
    assert context["severity_policy"] == confirmation["severity_policy"]
    assert confirmation["mapping"] == {str(s): "hate" if s else "non-hate" for s in range(5)}
    assert confirmation["null_result"] is None
    assert not confirmation["online_policy_modified"]
    return context, ref(context_path)


def mapped_hate(score):
    if score is None:
        return None
    if type(score) is not int or score not in range(5):
        raise ValueError("Severity must be an integer 0–4 or null")
    return "hate" if score else "non-hate"


def load_group_priority(base=RUN):
    """Read explicitly scoped AI applications; never infer a blanket override."""
    context_path = base / "group_priority_context.json"
    if not context_path.exists():
        return {}, []
    context = read(context_path)
    acceptance = read(checked(base, context["confirmation"]))
    checked(ROOT, context["document"])
    assert context["policy"] == acceptance["effective_policy"]
    assert context["document"] == acceptance["document"]
    assert context["policy"]["sha256"] == context["document"]["sha256"]
    assert acceptance["status"] == "user_confirmed_scope"
    assert not acceptance["online_policy_modified"]
    checked(base, acceptance["confirmation_source"])
    checked(base, context["base_group_context"])
    records, sources = {}, [ref(context_path, base)]
    for path in sorted((base / context["application_directory"]).glob("batch-*.json")):
        application = read(path)
        assert application["review_kind"] == "ai_note"
        assert application["human_group_decisions_added"] == application["online_material_confirmations_added"] == 0
        assert checked(base, application["context"]) == context_path
        labels = read(checked(base, application["source_label_export"]))
        assert labels["batch"] == application["batch"]
        by_id = {row["object_id"]: row for row in labels["records"]}
        application_ref = ref(path, base)
        for record in application["records"]:
            oid = record["object_id"]
            assert oid not in records, "Do not combine repeated group-priority applications silently"
            label = by_id[oid]
            assert "group" not in label["human_adjudicated_tasks"], "Human group requires explicit reopening"
            assert not record["human_group_confirmation_added"]
            assert record["field_provenance"] == "rule_priority_from_ai_basis"
            assert record["others_basis"] == "individual_insult_only" and record["independent_others_identity"] is False
            previous = label["values"]["group"]
            assert record["previous_group"] == previous and {"Racism", "others"} <= set(previous)
            assert record["current_group"] == [g for g in previous if g != "others"]
            assert record["text_sha256"] == hashlib.sha256(label["text"].encode()).hexdigest()
            assert record["quotes"] and all(q and q in label["text"] for q in record["quotes"])
            records[oid] = {"application": record, "source": application_ref, "policy": context["policy"]}
        sources.append(application_ref)
    return records, sources


def apply_group_priority(mapped, label, applications):
    entry = applications.get(label["object_id"])
    if entry is None:
        return
    record = entry["application"]
    assert "group" not in label["human_adjudicated_tasks"]
    assert mapped["values"]["group"] == record["previous_group"]
    mapped["historical_group"] = {"value": label["values"]["group"],
                                  "policy": label["task_policies"]["group"],
                                  "field_provenance": label["field_provenance"]["group"]}
    mapped["values"]["group"] = list(record["current_group"])
    mapped["task_policies"]["group"] = entry["policy"]
    mapped["field_provenance"]["group"] = record["field_provenance"]
    mapped["source_records"]["group_priority"] = entry["source"]
    mapped["group_priority"] = entry
    mapped["human_group_decisions_added_by_priority"] = 0


def build_mapping(label, severity, context, label_ref, severity_ref):
    assert label["object_id"] == severity["object_id"]
    assert label["text"] == severity["text"]
    assert hashlib.sha256(label["text"].encode()).hexdigest() == severity["text_sha256"]
    assert severity["task_policy"] == context["severity_policy"]
    score = severity["values"]["attack_severity"]
    hate = mapped_hate(score)
    assert (score is not None) == (severity["assessment_status"] == "scored")
    previous = label["values"]["hate"]
    source = "unresolved_severity" if score is None else (
        "rule_mapping_from_user_score" if severity["human_numeric_score_confirmed"] else "rule_mapping_from_ai_score")
    direct_human = "hate" in label["human_adjudicated_tasks"]
    return {"object_id": label["object_id"], "text": label["text"],
        "text_sha256": severity["text_sha256"], "base_version": severity["base_version"],
        "values": {"hate": hate, "group": label["values"]["group"], "attack_severity": score},
        "task_policies": {"hate": context["policy"], "group": label["task_policies"]["group"],
            "attack_severity": severity["task_policy"]},
        "field_provenance": {"hate": source, "group": label["field_provenance"]["group"],
            "attack_severity": severity["field_provenance"]["attack_severity"]},
        "source_records": {"label": label_ref, "severity": severity_ref},
        "historical_hate": {"value": previous, "policy": label["task_policies"]["hate"],
            "field_provenance": label["field_provenance"]["hate"],
            "explicit_human_decision": label["human_decision"] if direct_human else None},
        "comparison_with_historical_hate": "unmapped" if hate is None else "agrees" if hate == previous else "changed",
        "compatible_explicit_human_hate": direct_human and hate is not None and hate == previous,
        "human_hate_decisions_added_by_mapping": 0, "online_material_confirmed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=int, action="append", default=[])
    parser.add_argument("--batch", type=int, action="append", default=[])
    args = parser.parse_args()
    if not (args.pilot or args.batch) or any(n < 1 for n in args.pilot + args.batch):
        parser.error("Specify at least one positive --pilot or --batch")
    context, context_ref = load_mapping()
    group_applications, group_sources = load_group_priority()
    manifest = read(RUN / "input_manifest.json")
    sources = [RUN / "severity" / f"{kind}-{n:02d}" / "ai_annotations.json"
        for kind, numbers in (("pilot", args.pilot), ("batch", args.batch)) for n in numbers]
    records, seen, label_exports = [], set(), {}
    for path in sources:
        payload = read(path)
        assert payload["parent_session"] == manifest["parent_session"]
        assert payload["bundle_sha256"] == manifest["bundle_sha256"]
        checked(path.parent, payload["selection"])
        checked(path.parent, payload["annotation_source"])
        for decision in payload["discussion_provenance"]:
            checked(path.parent, decision)
        for severity in payload["records"]:
            oid = severity["object_id"]
            assert oid not in seen, "Choose one current score per object; do not merge duplicates silently"
            assert oid in manifest["pending_ids"]
            assert severity["base_version"] == manifest["pending_versions"][oid]
            seen.add(oid)
            batch = severity["prior_label_reference"]["batch"]
            if batch not in label_exports:
                lp = RUN / "batches" / f"batch-{batch:02d}" / "ai_annotations.json"
                labels = read(lp)
                assert labels["parent_session"] == manifest["parent_session"]
                assert labels["bundle_sha256"] == manifest["bundle_sha256"]
                label_exports[batch] = ({x["object_id"]: x for x in labels["records"]}, ref(lp))
            labels, label_ref = label_exports[batch]
            mapped = build_mapping(labels[oid], severity, context, label_ref, ref(path))
            apply_group_priority(mapped, labels[oid], group_applications)
            records.append(mapped)
    changed = [x for x in records if x["comparison_with_historical_hate"] == "changed"]
    unresolved = [x for x in records if x["values"]["hate"] is None]
    payload = {"schema_version": "evidence-severity-hate-mapping-view/v1", "mapping_context": context_ref,
        "policy": context["policy"], "source_severity_exports": [ref(p) for p in sources],
        "source_label_exports": [r for _, r in label_exports.values()],
        "parent_session": manifest["parent_session"], "bundle_sha256": manifest["bundle_sha256"],
        "object_count": len(records), "hate_counts": dict(Counter(x["values"]["hate"] or "null" for x in records)),
        "changed_from_historical_hate_ids": [x["object_id"] for x in changed],
        "changed_from_explicit_human_hate_ids": [x["object_id"] for x in changed if x["historical_hate"]["explicit_human_decision"]],
        "unmapped_ids": [x["object_id"] for x in unresolved],
        "human_hate_decisions_added_by_mapping": 0, "online_material_confirmations_added": 0, "records": records}
    if group_sources:
        payload["group_priority_sources"] = group_sources
        payload["group_priority_applied_ids"] = [r["object_id"] for r in records if "group_priority" in r]
        payload["human_group_decisions_added_by_priority"] = 0
    out = RUN / "severity/hate-mapping-v1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ai_annotations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    buffer = io.StringIO(); writer = csv.writer(buffer)
    writer.writerow(["object_id", "text", "hate", "group", "attack_severity", "hate_source", "historical_hate",
        "historical_hate_source", "comparison", "hate_policy", "group_policy", "severity_policy", "online_material_confirmed"])
    for x in records:
        v, h = x["values"], x["historical_hate"]
        writer.writerow([x["object_id"], x["text"], v["hate"] or "null", json.dumps(v["group"], ensure_ascii=False),
            "null" if v["attack_severity"] is None else v["attack_severity"], x["field_provenance"]["hate"],
            h["value"] or "null", h["field_provenance"], x["comparison_with_historical_hate"],
            *[json.dumps(x["task_policies"][t], ensure_ascii=False) for t in ("hate", "group", "attack_severity")], False])
    (out / "ai_annotations.csv").write_text(buffer.getvalue())
    lines = ["# 当前严重度到 hate 的默认映射", "",
        "用户已接受暂行映射：0→non-hate，1–4→hate，null→null。仅活动分值参与映射；旧标签及裁决保留各自政策。", "",
        f"共 {len(records)} 条：{len(changed)} 条与现有标签层不同，{len(unresolved)} 条因分数待定无法映射。规则生成的 hate 不新增用户直接裁决。", "",
        "[JSON](ai_annotations.json) · [CSV](ai_annotations.csv)", "", "## 与现有标签层的差异", ""]
    for x in changed + unresolved:
        v, h = x["values"], x["historical_hate"]
        lines += [f"- {x['object_id']}：旧值 {h['value']}（{h['field_provenance']}）；分数 {v['attack_severity']} → 当前 {v['hate']}（{x['field_provenance']['hate']}）。"]
    lines += ["", "#4392 的 non-hate 草稿已由用户明确改为 hate，裁决前稿另存于第四批归档；此处与其最新明确裁决对照。其他差异只来自默认映射，没有重写旧裁决。", ""]
    if group_sources:
        lines += ["当前 group 另应用了已明确记录的种族／个人辱骂优先级：" +
                  "、".join(payload["group_priority_applied_ids"]) + "。仅作用于所列 AI 草稿；原 group 和政策保存在 historical_group。" +
                  "[应用与历史复核](../../group-priority-v1/README.md)", ""]
    (out / "README.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in payload.items() if k in ("object_count", "hate_counts", "changed_from_historical_hate_ids", "unmapped_ids")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
