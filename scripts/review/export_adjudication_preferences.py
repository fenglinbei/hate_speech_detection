#!/usr/bin/env python3
"""Build a traceable local preference table from explicit, field-scoped decisions.

Rule summaries and cross-case comparisons are curated AI notes, not adjudications.
This exporter never edits source decisions, frozen inputs, or review sessions.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/ai_reviews/v2-reannotation-20260911"
FIELDS = ("hate", "group", "attack_severity")


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def display(value):
    if value is None:
        return "待定"
    if isinstance(value, list):
        return "、".join(value) or "[]"
    return str(value)


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def resolve_amendments(events):
    """Only explicit, same-field amendments retire an earlier human event."""
    by_id = {e['event_id']: e for e in events}
    assert len(by_id) == len(events), 'Duplicate event ids'
    amendments = []
    for event in events:
        prior_ids = event.get('supersedes_event_ids', [])
        if not prior_ids:
            continue
        assert event.get('amendment_reason')
        assert len(prior_ids) == len(set(prior_ids))
        for prior_id in prior_ids:
            assert prior_id in by_id and prior_id != event['event_id'], prior_id
            prior = by_id[prior_id]
            assert (prior['object_id'], prior['field']) == (event['object_id'], event['field'])
            if prior.get('recorded_at_utc') and event.get('recorded_at_utc'):
                assert prior['recorded_at_utc'] <= event['recorded_at_utc']
            prior.setdefault('superseded_by', []).append(event['event_id'])
        amendments.append({'object_id': event['object_id'], 'field': event['field'],
                           'event_id': event['event_id'], 'supersedes_event_ids': prior_ids,
                           'reason': event['amendment_reason']})
    def visit(event_id, chain):
        assert event_id not in chain, 'Cyclic amendment history'
        for prior_id in by_id[event_id].get('supersedes_event_ids', []):
            visit(prior_id, chain | {event_id})
    for event_id in by_id:
        visit(event_id, set())
    return amendments


def build(run):
    pref = run / "preferences"
    sources = {}

    def source(path):
        relative = str(path.relative_to(run))
        sources[relative] = {"path": relative, "sha256": digest(path.read_bytes())}
        return sources[relative]

    catalog = read(pref / "rule-cards.json")
    source(pref / "rule-cards.json")
    consistency = read(pref / "consistency-notes.json")
    source(pref / "consistency-notes.json")
    workflow = read(pref / "maintenance.json")
    source(pref / "maintenance.json")
    manifest = read(run / "input_manifest.json")
    source(run / "input_manifest.json")
    texts = {}
    for path in sorted((run / "inputs").glob("labels-*.jsonl")):
        source(path)
        for row in map(json.loads, path.read_text().splitlines()):
            assert row["id"] not in texts, f"Duplicate input: {row['id']}"
            texts[row["id"]] = row

    events = []
    patterns = ("discussions/batch-*/user-decisions-[0-9][0-9].json",
                "severity/batch-*/discussions/user-decisions-[0-9][0-9].json",
                "severity/pilot-*/discussions/user-decisions-[0-9][0-9].json",
                "sentence-completion-v1/discussions/user-decisions-[0-9][0-9].json")
    paths = sorted({p for pattern in patterns for p in run.glob(pattern)})
    for path in paths:
        doc = read(path)
        ref = source(path)
        for index, record in enumerate(doc["records"]):
            oid = record["object_id"]
            assert oid in texts, f"Unauthorized case: {oid}"
            assert record["text_sha256"] == digest(texts[oid]["text"].encode()), oid
            assert set(record["scope"]) == set(record["values"]) <= set(FIELDS), oid
            assert record["review_kind"] == "human_with_ai", oid
            for field in record["scope"]:
                policy = record.get("task_policies", {}).get(field)
                policy = policy or doc.get("task_policies", {}).get(field)
                declared_task_policy = policy
                if not policy:
                    # Legacy label files declare the group policy at file scope.
                    # Their hate task remains the manifest policy, as in the label exporter.
                    policy = (doc.get("task_policy") if field == "attack_severity" else
                              manifest["policy"] if field == "hate" else doc.get("policy"))
                assert policy and policy.get("sha256"), (oid, field)
                event = {
                    "event_id": f"{ref['path']}#records/{index}/{field}",
                    "object_id": oid, "field": field, "value": record["values"][field],
                    "text_sha256": record["text_sha256"], "task_policy": policy,
                    "task_policy_basis": "explicit_task_declaration" if declared_task_policy else "legacy_exporter_task_context",
                    "source_file_policy": doc.get("policy", doc.get("task_policy")),
                    "recorded_at_utc": doc.get("recorded_at_utc"),
                    "rationale": record.get("rationale"),
                    "rationale_provenance": record.get("rationale_provenance", "not_recorded"),
                    "source": ref, "source_record_index": index,
                    "review_kind": record["review_kind"], "mode": record.get("mode"),
                }
                supersedes = record.get("supersedes_event_ids", {}).get(field, [])
                if supersedes:
                    assert record.get("amendment_reason"), (oid, field)
                    event["supersedes_event_ids"] = supersedes
                    event["amendment_reason"] = record["amendment_reason"]
                events.append(event)
    amendments = resolve_amendments(events)
    by_case = defaultdict(list)
    for event in events:
        by_case[event["object_id"]].append(event)
    for history in by_case.values():
        history.sort(key=lambda e: (e["recorded_at_utc"] or "", e["event_id"]))

    cards = catalog["rules"]
    assert len({card["id"] for card in cards}) == len(cards)
    covered = set()
    for card in cards:
        assert card["status"] in ("explicit_user_rule", "ai_case_synthesis")
        assert card["summary_provenance"] == "ai_note"
        if card["status"] == "explicit_user_rule":
            assert card.get("acceptance_source")
        if card.get("acceptance_source"):
            path = run / card["acceptance_source"]
            acceptance = read(path)
            assert acceptance.get("confirmation_text")
            document = acceptance.get("document")
            if document:
                assert digest((ROOT / document["path"]).read_bytes()) == document["sha256"]
            card["acceptance"] = {**source(path), "confirmation_text": acceptance["confirmation_text"]}
        for anchor in card["anchors"]:
            oid = anchor["object_id"]
            assert oid in by_case, f"Not a human-adjudicated anchor: {oid}"
            covered.add(oid)
            anchor["event_ids"] = []
            for field, value in anchor["fields"].items():
                matching = [e for e in by_case[oid] if e["field"] == field and e["value"] == value]
                assert matching, f"Anchor exceeds explicit user scope: {card['id']} {oid} {field}={value}"
                anchor["event_ids"].extend(e["event_id"] for e in matching)
                if all(e.get("superseded_by") for e in matching):
                    anchor.setdefault("historical_fields", []).append(field)

    notes = consistency["notes"]
    assert len({n["id"] for n in notes}) == len(notes)
    for note in notes:
        assert note["assessment_provenance"] == "ai_note"
        assert set(note["case_ids"]) <= set(by_case)
        assert set(note["rule_ids"]) <= {card["id"] for card in cards}
    cases, conflicts = [], []
    for oid in sorted(by_case, key=lambda x: (int(x.split(":")[1]), x)):
        history = by_case[oid]
        unique, ambiguous = {}, []
        for field in FIELDS:
            values = {json.dumps(e["value"], sort_keys=True, ensure_ascii=False): e["value"]
                      for e in history if e["field"] == field and not e.get("superseded_by")}
            if len(values) == 1:
                unique[field] = next(iter(values.values()))
            elif len(values) > 1:
                ambiguous.append(field)
                conflicts.append({"kind": "same_case_field_history_differs", "object_id": oid,
                                  "field": field, "values": list(values.values()),
                                  "action": "Compare policies and explicit amendment evidence; never choose the last value silently."})
        cases.append({"object_id": oid, "text": texts[oid]["text"],
                      "text_sha256": digest(texts[oid]["text"].encode()),
                      "unambiguous_human_fields": unique, "ambiguous_human_fields": ambiguous,
                      "history": history,
                      "rule_ids": [c["id"] for c in cards if any(a["object_id"] == oid for a in c["anchors"])],
                      "consistency_note_ids": [n["id"] for n in notes if oid in n["case_ids"]]})
    counts = dict(Counter(e["field"] for e in events))
    result = {"schema_version": "evidence-adjudication-preferences/v1",
              "scope": "Current-task local explicit decisions only; excludes online confirmations, original labels and AI guesses.",
              "summary_provenance": "ai_note", "maintenance": workflow,
              "rule_count": len(cards), "human_case_count": len(cases),
              "human_field_event_counts": counts,
              "uncategorized_case_ids": sorted(set(by_case) - covered),
              "sources": sorted(sources.values(), key=lambda s: s["path"]),
              "rules": cards, "cases": cases, "consistency_notes": notes,
              "automatic_conflict_candidates": conflicts,
              "explicit_amendments": amendments,
              "semantic_conflict_detection": "Manual comparison required; shared keywords or differing labels alone are not a conflict.",
              "human_decisions_added": 0, "online_material_confirmations_added": 0}
    version = digest(encoded(result))
    result["snapshot_id"] = version
    base = run / "preferences" / "snapshots" / version
    lines = ["# 裁决规则与偏好表", "",
             f"当前收录 {len(cases)} 个已裁决案例、{len(cards)} 条规则／归纳。明确字段记录：hate {counts.get('hate',0)}、group {counts.get('group',0)}、严重度 {counts.get('attack_severity',0)}。", "",
             "“已确认规则”有明确的规则级授权；“个案归纳”是 AI 对已确认案例的总结，遇到新边界仍需复核。表中的案例值只展示用户实际裁定的字段，未裁定字段留空。", "",
             "每轮先检索相近案例，再比较实际对象、攻击命题、作者立场、可见上下文和当时政策。疑似冲突会提示前后案例与具体字段；在用户说明前保留两边记录，不自动覆盖，也不把个案变成通则。", "",
             f"[完整 JSON]({base / 'preferences.json'}) · [案例 CSV]({base / 'cases.csv'}) · [维护说明]({run / 'preferences/README.md'})", "",
             "## 规则速查", "",
             "| 编号／情形 | 当前口径 | 依据 | 适用边界 |",
             "| --- | --- | --- | --- |"]
    for card in cards:
        label = "已确认规则" if card["status"] == "explicit_user_rule" else "个案归纳"
        anchors = "；".join(f"[#{a['object_id'].split(':')[1]}](#case-{a['object_id'].replace(':','-')})：" +
                            "，".join(f"{k}={display(v)}" for k,v in a["fields"].items()) for a in card["anchors"])
        if card.get("acceptance"):
            anchors = f"[规则确认]({run / card['acceptance']['path']})；" + anchors
        lines.append(f"| {card['id']} · {cell(card['situation'])}<br>{label} | {cell(card['guidance'])} | {anchors} | {cell(card['limits'])} |")
    lines += ["", "## 一致性复核记录", "",
              "以下判断属于 AI 复核记录；“可能差异”不代表已认定用户前后矛盾。", ""]
    for note in notes:
        lines += [f"- **{note['id']} · {note['status']}**：{note['observation']} {note['handling']}"]
    for amendment in amendments:
        lines += [f"- **明确修订**：{amendment['object_id']} 的 {amendment['field']} 已按本轮明确回复复核；旧值保留在下方字段历史，不作为未解释的裁决冲突。"]
    for conflict in conflicts:
        lines += [f"- **待复核**：{conflict['object_id']} 的 {conflict['field']} 有多个历史值；需核对政策与修订依据。"]
    if result["uncategorized_case_ids"]:
        lines += ["", "尚未纳入规则卡的案例：" + "、".join(result["uncategorized_case_ids"]) + "。记录已收录，等待人工整理适用边界。"]
    lines += ["", "## 可追溯的案例索引", "",
              "“未直接裁定”不等于空类别、0 级或 non-hate。历史 hate 不自动替换成当前严重度映射；例如 #1240 的政策变化见一致性记录。", ""]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["object_id", "text", "human_hate", "human_group", "human_attack_severity", "ambiguous_fields", "rule_ids", "history_json"])
    for case in cases:
        oid = case["object_id"]
        fields = case["unambiguous_human_fields"]
        lines += [f"<a id=\"case-{oid.replace(':','-')}\"></a>", "", f"### #{oid.split(':')[1]}", "",
                  "> " + case["text"].replace("\n", "\n> "), "",
                  "；".join(f"{field}：{display(fields[field]) if field in fields else '未直接裁定／存在多值'}" for field in FIELDS), ""]
        for event in case["history"]:
            state = "（历史值，已由明确复核取代）" if event.get("superseded_by") else ""
            lines += [f"- {event['field']}＝{display(event['value'])}{state}；理由：{event['rationale'] or '未提供'}（{event['rationale_provenance']}）。[字段记录]({run / event['source']['path']})"]
        lines += [""]
        writer.writerow([oid, case["text"], fields.get("hate", ""),
                         json.dumps(fields["group"], ensure_ascii=False) if "group" in fields else "",
                         fields.get("attack_severity", ""), ",".join(case["ambiguous_human_fields"]),
                         ",".join(case["rule_ids"]), json.dumps(case["history"], ensure_ascii=False)])
    return version, {"preferences.json": encoded(result), "preferences.md": ("\n".join(lines)+"\n").encode(),
                     "cases.csv": buf.getvalue().encode()}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--check", action="store_true", help="Verify current and immutable outputs without writing.")
    args = parser.parse_args()
    version, artifacts = build(args.run)
    pref = args.run / "preferences"
    for name, data in artifacts.items():
        frozen = pref / "snapshots" / version / name
        if args.check or frozen.exists():
            assert frozen.read_bytes() == data, f"Immutable snapshot differs: {frozen}"
        else:
            write(frozen, data)
        current = pref / name
        if args.check:
            assert current.read_bytes() == data, f"Refresh required: {current}"
        else:
            write(current, data)
    pointer = encoded({"snapshot_id": version, "artifacts": {name: {"path": f"preferences/snapshots/{version}/{name}", "sha256": digest(data)} for name, data in artifacts.items()}})
    if args.check:
        assert (pref / "current.json").read_bytes() == pointer
    else:
        write(pref / "current.json", pointer)
    print(json.dumps({"snapshot_id": version, "check_only": args.check, "output": str(pref / "preferences.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
