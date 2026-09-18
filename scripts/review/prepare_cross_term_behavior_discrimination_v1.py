"""Prepare/check a CPU-only review draft; no scoring or human-adoption operation."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLANS = ROOT / "docs/research/experiment-plans"
OUT = PLANS / "cross-term-behavior-discrimination-v1/draft-01"
PROTOCOL = PLANS / "task-applicability-scoring-v1/frozen-01"
MATERIALS = PLANS / "cross-term-materials-v1/frozen-01"
DEMOS = PLANS / "cross-term-demos-v1/frozen-01"
JOINT = PLANS / "cross-term-joint-v1/frozen-01"
PREVIOUS = ROOT / "reviews/cross-term-next-token-v1"
ARMS = ["none", "same_A", "same_B", "other_A", "other_B"]
RULES = {
    "A-01": "对具体行为的抱怨与另加的人身贬损分开判断；此示例以普通物件场景加面向行为实施者的口语辱骂呈现。",
    "A-02": "要求别人调整行为且未增加人身贬损，不自动构成人身攻击；此示例展示调整物品摆放的请求。",
    "B-01": "先识别对人的贬损性称呼，再绑定当前作者的使用／反对立场；此示例展示作者直接实施辱称的分支。",
    "B-02": "先识别对人的贬损性称呼，再绑定当前作者的使用／反对立场；此示例展示明确劝阻辱称且未另行攻击的分支。",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def info(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": digest(data)}


def require(value, message):
    if not value:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def span(text, excerpt):
    start = text.index(excerpt)
    return {"start": start, "end": start + len(excerpt), "text": excerpt}


def occurrences(text, term):
    result, start = [], 0
    while (start := text.find(term, start)) >= 0:
        result.append({"start": start, "end": start + len(term), "text": term})
        start += 1
    return result


def fit(value, information, source, target, source_excerpt, target_excerpt,
        rationale, limitation=None, steps=()):
    return {
        "value": value, "information": information,
        "source_spans": [span(source["raw_text"], source_excerpt)],
        "target_spans": [span(target["raw_text"], target_excerpt)],
        "rationale": rationale, "limitation": limitation,
        "decision_relevance": {
            "value": "task_relevant" if steps else "background", "steps": list(steps),
            "rationale": "指定信息涉及这些判题环节；不声称模型未知或已经使用。" if steps else
                         "此项具体信息没有目标对应位置；不排除其他任务示范或上下文作用。",
        },
    }


def base_relation(rid, kind, family, source, target, quality, term, created):
    shared = [term] if term in source["raw_text"] and term in target["raw_text"] else []
    return {
        "schema_version": "evidence-applicability-relation/v1",
        "task_protocol": "task-applicability-scoring/v1", "relation_id": rid,
        "family_id": family, "supersedes": None, "relation_kind": kind,
        "source": copy.deepcopy(source), "target": copy.deepcopy(target),
        "source_quality": copy.deepcopy(quality), "sense_fit": None,
        "semantic_reference_fit": None, "rule_fit": None,
        "lexical_overlap": {
            "comparison": "exact_unicode_literal", "focal_forms": [term],
            "value": "present" if shared else "absent", "shared_forms": shared,
            "source_spans": occurrences(source["raw_text"], term),
            "target_spans": occurrences(target["raw_text"], term),
            "rationale": "复用预先确定的研究词形；未共现不等于全文无共享词，也不等于语义完全无关。",
        },
        "presentation_refs": [], "introduced_lexicon_relation_ids": [],
        "provenance": {
            "authorship": "ai", "review_kind": "ai_note", "adoption": "none",
            "accepted_fields": [], "decision_ref": None, "recorded_at": created,
            "exposure": {"target_model_outputs_seen": True, "original_gold_seen": False,
                         "ai_notes_seen": True, "prior_related_results_seen": True},
        },
    }


def new_relations(qdata, ddata, old_relations, created):
    queries = {q["short_id"]: q for q in qdata["queries"]}
    demos = {d["material"]["material_id"]: d for d in ddata["demos"]}
    quality = {r["source"]["material_id"]: r["source_quality"] for r in old_relations}
    lexicons = {f: qdata["lexicons"][i] for i, f in enumerate("GJB")}
    records = []
    for letter in "GJB":
        lex = lexicons[letter]
        for suffix in RULES:
            source, target = lex["material"], demos["CTDD-X-" + suffix]["material"]
            rid = f"CTD2-LD-{source['material_id']}-X-{suffix}"
            r = base_relation(rid, "lexicon_to_demo", queries[letter + "1"]["family_id"],
                              source, target, quality[source["material_id"]], lex["term"], created)
            excerpt = "纸箱" if suffix.startswith("A") else "废物"
            partial = letter == "G" and suffix.startswith("B")
            r["sense_fit"] = fit(
                "partial" if partial else "none", source["raw_text"].split("：", 1)[1],
                source, target, "无用" if partial else source["raw_text"].split("：", 1)[1], excerpt,
                "指人‘废物’与该释义的‘无用／无价值’评价有可指出的部分语义联系。" if partial else
                f"示例中的‘{excerpt}’不承载这条词典所指定的{lex['term']}义项或所指。",
                "不是完整同义替换，也没有字面命中；只能迁移无用／低价值评价部分，不能由词典推断作者认可。此跨词义项联系待人审。" if partial else
                "仅排除该具体义项；示例中的普通名词、辱称判断或作者立场规则仍可能另有作用。",
                ["sense_interpretation"] if partial else [],
            )
            records.append(r)
    for short in [f + n for f in "GJB" for n in ("1", "3")]:
        q = queries[short]
        target = q["query"]
        term = lexicons[short[0]]["term"]
        for suffix in RULES:
            source = demos["CTDD-X-" + suffix]["material"]
            r = base_relation(f"CTD2-DQ-X-{suffix}-{short}", "demo_to_query", q["family_id"],
                              source, target, quality[source["material_id"]], term, created)
            source_word = "纸箱" if suffix.startswith("A") else "废物"
            partial = short[0] == "G" and suffix.startswith("B")
            r["semantic_reference_fit"] = fit(
                "partial" if partial else "none",
                "‘废物’用于表达对人的无用／低价值评价。" if suffix.startswith("B") else "‘纸箱’指普通物品。",
                source, target, source_word, term,
                "源中‘废物’和目标指人‘垃圾’共享无用／低价值评价这一部分；语义联系不因反对立场而消失。" if partial else
                f"源中的‘{source_word}’不能提供目标‘{term}’的具体{('地域' if short[0] == 'J' else '性污名' if short[0] == 'B' else '指人贬损')}义项／所指信息。",
                "不宣称完全同义，也不能从该联系推出作者认可辱称；认可或反对单独在规则维度判断。" if partial else
                "共同贬义或同为有／无不是具体词义对应；规则迁移另审。",
                ["sense_interpretation"] if partial else [],
            )
            if suffix == "A-01":
                value = "partial" if short == "G1" else "none"
                reason = {
                    "G1": "借款不还的行为指责与‘你这种人真是垃圾’的人身贬损可分开定位，对应源中通行抱怨与另加辱骂的区分。",
                    "J1": "目标围绕户口身份轻蔑和地域辱称，没有同样可分离的具体行为抱怨加口语辱骂结构；疑问语气本身不足以建立迁移。",
                    "B1": "目标实施性污名辱称，并反问听者为何珍视她；后半句不能据形式认作源中面向行为实施者的独立口语辱骂。",
                }.get(short, "目标在劝阻他人辱称，未另加源中那种作者实施的口语辱骂；不能把被反对的词归到作者名下。")
                limitation = "仅共享行为批评与额外对人贬损的区分；口语责问与人格辱称的具体识别不同。与旧G1←同词A-01的窄范围none并列保存，不修改旧值。" if value == "partial" else "none只针对这里命名的结构；不意味着示例整体无用或模型不会受它影响。"
                src_excerpt = source["raw_text"]
                tgt_excerpt = target["raw_text"]
                steps = ["attack_presence", "referent_binding"] if value == "partial" else []
            elif suffix == "A-02":
                value = "partial" if short.endswith("3") else "none"
                reason = "目标明确要求对方停止用辱称骂人；‘行为劝阻本身不自动构成攻击’可迁移，但源示例没有被反对的辱称，不能完成目标的立场绑定。" if value == "partial" else "目标没有要求对方调整行为而未增加人身贬损的对应请求；不能仅凭同为陈述／对话认定规则适用。"
                limitation = "只迁移劝阻／请求的性质，识别被反对辱称及作者立场仍需另判。旧同词A-02采用更窄的普通所指结构，旧值保留。" if value == "partial" else "不把所有无攻击示例合成一条规则，也不据查询与示例答案异同评级。"
                src_excerpt = "挪开点，别挡着别人"
                tgt_excerpt = {"G3": "你就说这事，别一口一个垃圾", "J3": "别动不动就拿京巴骂人", "B3": "别张口就叫人家公交车"}.get(short, target["raw_text"])
                steps = ["attack_presence", "author_stance"] if value == "partial" else []
            else:
                same_branch = short.endswith("1") == (suffix == "B-01")
                value = "direct" if same_branch else "partial"
                reason = "源和目标都把贬损性称呼作为当前作者对人的攻击；具体词义不同，但作者绑定的判断分支相同。" if same_branch and short.endswith("1") else "源和目标都明确劝阻他人的辱称且未另行发起攻击；需把被反对称呼与当前作者立场分开。" if same_branch else "可迁移‘识别辱称，再判断是否由当前作者认可’的两步区分；源与目标处在相反立场分支，示例结论不能直接迁移。"
                limitation = "规则direct不提供地域／性别等具体词义，不允许复制示例标签；目标是否另有攻击仍单独检查。" if same_branch else "只迁移辱称与作者立场的区分；源展示的认可／反对结论不适用于目标的相反立场。不是因为标签不同才判partial。"
                src_excerpt, tgt_excerpt = source["raw_text"], target["raw_text"]
                steps = ["author_stance", "attack_presence"]
            r["rule_fit"] = fit(value, RULES[suffix], source, target, src_excerpt, tgt_excerpt,
                                reason, limitation, steps)
            r["introduced_lexicon_relation_ids"] = [f"CTD2-LD-{lexicons[short[0]]['material']['material_id']}-X-{suffix}"]
            records.append(r)
    return records


def tokenizer():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT / "models/base/Qwen3-8B"),
                                         local_files_only=True, trust_remote_code=False)


def render(query, lexicon, demos, system, tok):
    ltext = "词条1：\n" + lexicon["raw_text"] if lexicon else "（无）"
    dtext = "\n\n".join(f"示例{i}：\n文本：{d['material']['raw_text']}\n答案：{d['human_review']['task_label']}" for i, d in enumerate(demos, 1)) or "（无）"
    user = f"【参考词典】\n{ltext}\n\n【参考示例】\n{dtext}\n\n【待判断文本】\n{query['raw_text']}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    return {"messages": messages, "chat_prompt": prompt, "prompt_sha256": digest(prompt.encode()),
            "input_ids": ids, "input_ids_sha256": digest(json.dumps(ids, separators=(",", ":")).encode()),
            "prompt_tokens": len(ids), "last_input_token_index": len(ids) - 1,
            "next_token_position": len(ids), "candidate_tokens": {"有": 18830, "无": 42192}}


def comparisons(conditions, queries):
    result = []
    lookup = {(c["query_short_id"], c["lexicon_arm"], c["demo_arm"]): c for c in conditions}
    for q in queries:
        short = q["short_id"]
        def term(l, d, coefficient):
            return {"condition_id": lookup[short, l, d]["condition_id"], "coefficient": coefficient}
        def add(name, terms, priority):
            result.append({"comparison_id": f"CTD2-{short}-{name}", "kind": name,
                           "query_id": q["query"]["material_id"], "query_short_id": short,
                           "family_id": q["family_id"], "terms": terms, "priority": priority,
                           "reference_direction_sign": 1 if q["human_review"]["task_label"] == "无" else -1,
                           "exposure": "contains_new_unscored_input" if any(
                               next(c for c in conditions if c["condition_id"] == t["condition_id"])["input_status"] == "new_unscored" for t in terms)
                               else "all_endpoints_previously_exposed", "human_adopted": False})
        for pack in "AB":
            x, s = "other_" + pack, "same_" + pack
            add("E_" + pack, [term("L", x, 1), term("absent", x, -1)], "primary")
            add("I_" + pack, [term("L", x, 1), term("absent", x, -1), term("L", "none", -1), term("absent", "none", 1)], "primary")
            add("K_" + pack, [term("L", x, 1), term("absent", x, -1), term("L", s, -1), term("absent", s, 1)], "primary")
            add("D_empty_" + pack, [term("absent", x, 1), term("absent", "none", -1)], "primary")
            add("D_L_" + pack, [term("L", x, 1), term("L", "none", -1)], "primary")
            add("X_minus_same_empty_" + pack, [term("absent", x, 1), term("absent", s, -1)], "secondary_bundle")
        for l in ("absent", "L"):
            add("X_B_minus_A_" + l, [term(l, "other_B", 1), term(l, "other_A", -1)], "secondary_bundle")
    return result


def tsv(path, rows, fields):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (list, dict)) else row[k] for k in fields})
    path.write_text(stream.getvalue(), encoding="utf-8")


def build():
    require(not OUT.exists(), f"Draft already exists: {OUT}; use check or a separate revision.")
    created = datetime.now(timezone.utc).isoformat()
    qdata, ddata, old_design = read(MATERIALS / "materials.json"), read(DEMOS / "materials.json"), read(JOINT / "design.json")
    old_relations = read(JOINT / "relations.json")["records"]
    lq = read(MATERIALS / "relations.json")["records"]
    old_inputs = {p["condition_id"]: p for p in map(json.loads, (PREVIOUS / "frozen-01/model-inputs.jsonl").read_text().splitlines())}
    old_conditions = {(c["query_short_id"], c["lexicon_arm"], c["demo_arm"]): c for c in old_design["conditions"]}
    demos = {d["material"]["material_id"]: d for d in ddata["demos"]}
    rels = new_relations(qdata, ddata, old_relations, created)
    all_relations = old_relations + lq + rels
    relation_by_pair = {(r["relation_kind"], r["source"]["material_id"], r["target"]["material_id"]): r for r in all_relations}
    system = (PROTOCOL / "model-task.txt").read_text()
    tok = tokenizer()
    conditions, inputs, bridges, appearances, bindings = [], [], [], [], []
    for q in qdata["queries"]:
        short = q["short_id"]
        lex = qdata["lexicons"]["GJB".index(short[0])]["material"]
        for d in ARMS:
            demo_ids = [] if d == "none" else [f"CTDD-{short[0] if d.startswith('same') else 'X'}-{d[-1]}-{i:02d}" for i in (1, 2)]
            for l in ("absent", "L"):
                old = old_conditions.get((short, l, d))
                cid = old["condition_id"] if old else f"CTD2-{short}-{l}-{d}"
                p = render(q["query"], lex if l == "L" else None, [demos[mid] for mid in demo_ids], system, tok)
                c = {"condition_id": cid, "query_id": q["query"]["material_id"], "query_short_id": short,
                     "family_id": q["family_id"], "lexicon_arm": l, "demo_arm": d,
                     "lexicon_slot_material_ids": [lex["material_id"]] if l == "L" else [],
                     "demo_ids": demo_ids, "demo_answers": [demos[mid]["human_review"]["task_label"] for mid in demo_ids],
                     "input_status": "historical_reuse" if old else "new_unscored",
                     "prompt_sha256": p["prompt_sha256"], "prompt_tokens": p["prompt_tokens"],
                     "human_design_adoption": None}
                if old:
                    prior = old_inputs[cid]
                    require(all(prior[k] == v for k, v in p.items()), f"Historical input changed: {cid}")
                    bridges.append({"condition_id": cid, "request_id": prior["request_id"],
                                    "prompt_sha256": prior["prompt_sha256"], "input_ids_sha256": prior["input_ids_sha256"],
                                    "input_ref": info(PREVIOUS / "frozen-01/model-inputs.jsonl"),
                                    "score_ref": info(PREVIOUS / f"run-01/scores/science/{prior['request_id']}.json"),
                                    "results_ref": info(PREVIOUS / "results-01/results.json"),
                                    "reuse_status": "historical_score_available_future_bridge_qualification_required"})
                else:
                    inputs.append({"request_id": f"CTD2-PREVIEW-{len(inputs)+1:03d}", "condition_id": cid, **p})
                conditions.append(c)
                candidate = [relation_by_pair["lexicon_to_query", lex["material_id"], q["query"]["material_id"]]]
                for did in demo_ids:
                    dq = relation_by_pair["demo_to_query", did, q["query"]["material_id"]]
                    ld = relation_by_pair["lexicon_to_demo", lex["material_id"], did]
                    candidate.extend([dq, ld])
                    bindings.append({"condition_id": cid, "demo_query_relation_id": dq["relation_id"],
                                     "candidate_lexicon_demo_relation_id": ld["relation_id"],
                                     "lexicon_present": l == "L", "prompt_sha256": p["prompt_sha256"]})
                visible = set(c["lexicon_slot_material_ids"] + demo_ids + [c["query_id"]])
                for r in candidate:
                    app = {"condition_id": cid, "prompt_sha256": p["prompt_sha256"],
                           "source_present": r["source"]["material_id"] in visible,
                           "target_present": r["target"]["material_id"] in visible}
                    appearances.append({"relation_id": r["relation_id"], **app})
                    if r in rels:
                        r["presentation_refs"].append(app)
    require(len(conditions) == 120 and len(inputs) == 36 and len(bridges) == 84, "36/84 inventory mismatch")
    require(len({c["prompt_sha256"] for c in conditions}) == 120, "Duplicate prompt")
    comp = comparisons(conditions, qdata["queries"])
    OUT.mkdir(parents=True)
    source_paths = [PROTOCOL / x for x in ["manifest.json", "model-task.txt", "scoring-spec.json", "relation-record.schema.json", "validate_contract.py"]]
    source_paths += [MATERIALS / x for x in ["manifest.json", "materials.json", "relations.json"]]
    source_paths += [DEMOS / x for x in ["manifest.json", "materials.json"]]
    source_paths += [JOINT / x for x in ["manifest.json", "design.json", "relations.json", "rule-calibration.md", "rule-review-audit.json"]]
    source_paths += [PREVIOUS / x for x in ["frozen-01/manifest.json", "frozen-01/model-inputs.jsonl", "run-01/qualification.json", "results-01/manifest.json", "results-01/results.json", "interpretation-01/INTERPRETATION.md"]]
    source_paths += [MATERIALS.parent / "feedback-01.json", DEMOS.parent / "feedback-01.json", JOINT.parent / "feedback-01.json",
                     PLANS / "evidence-applicability-priority-review-20260916.md"]
    for entry in read(PROTOCOL / "manifest.json")["tokenizer_sources"]:
        source_paths.append(ROOT / entry["path"])
    dump(OUT / "sources.json", {"sources": [info(p) for p in dict.fromkeys(source_paths)],
                                "old_frozen_files_modified": False})
    dump(OUT / "materials.json", {"status": "exact_reuse_with_prior_adoption_preserved", "queries": qdata["queries"],
                                  "lexicons": qdata["lexicons"], "demos": ddata["demos"],
                                  "current_exposure": "All queries and old contexts are outcome-exposed development materials; old embedded provenance retains its historical timestamp."})
    dump(OUT / "design.json", {"schema_version": "cross-term-behavior-discrimination-draft/v1", "created_at": created,
                               "status": "awaiting_human_review", "human_design_adoption": None,
                               "scientific_input_freeze": False, "gpu_run_started": False,
                               "new_query_or_demo_texts": 0, "new_inputs": 36, "historical_core_inputs": 84,
                               "core_inputs": 120, "historical_N_inputs_retained_externally": 36,
                               "total_unique_inputs_including_old_N": 156,
                               "conditions": conditions, "rule_fit_causal_stratification": False,
                               "new_family_confirmation": False})
    dump(OUT / "relations.json", {"status": "ai_proposals_awaiting_human_review", "human_adoption": False,
                                  "records": rels, "exposure_note": "Prior target-query outputs have been seen. The 36 new combinations have no model outputs; target_model_outputs_seen is conservatively true, not reset."})
    dump(OUT / "reused-relations.json", {"records": [{"relation_id": r["relation_id"],
                 "source_file": str((MATERIALS / 'relations.json' if r['relation_kind'] == 'lexicon_to_query' else JOINT / 'relations.json').relative_to(ROOT)),
                 "status": "prior_human_adoption_reused_without_edit"} for r in old_relations + lq]})
    dump(OUT / "relation-appearances.json", {"records": appearances, "context_bindings": bindings,
        "note": "New presence/link metadata is external to immutable old relation records. Candidate links do not imply L visibility or D-triggered retrieval."})
    dump(OUT / "historical-bridges.json", {"records": bridges, "new_GPU_bridge_performed": False,
                                          "old_N_inputs": [c["condition_id"] for c in old_design["conditions"] if c["lexicon_arm"] == "N"]})
    (OUT / "model-inputs.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n" for p in inputs), encoding="utf-8")
    dump(OUT / "analysis-plan.json", {"status": "proposal_before_new_36_outcomes_not_confirmation_registration",
         "scorer_must_not_read_this_file": True, "primary_margin": "z[42192]-z[18830]", "comparisons": comp,
         "references": [{"query_id": q["query"]["material_id"], "human_reference": q["human_review"],
                         "original_gold": None, "original_correct": None} for q in qdata["queries"]],
         "reporting": {"all_core_conditions": 120, "all_new_inputs": 36, "all_comparisons": 168,
                       "unit": "query within named construction; family equal weighting; three dependent exposed terms",
                       "independent_confirmation_families": 0, "pool_relation_values": False,
                       "drop_unfavorable_or_partial_rows": False, "reference_join": "after raw score seal and numerical qualification"},
         "numeric_error_bound": None, "continuous_minimum_important_effect": None,
         "semantic_equivalence_threshold": None,
         "bound_policy": "Attach each qualified physical score's bound, combine identical score IDs before summing abs(coefficient)*bound. New inputs require qualification and historical reuse requires a separately bound bridge; no old epsilon silently inherited.",
         "functional_criterion": "Repair: resolved wrong -> resolved correct for the same query/reference. Preservation under L and reduced adverse L effect are separate statements. Report reverse damage too.",
         "exposure": "Design follows completed first-round outcomes; new 36 are unscored. Comparisons with all old endpoints remain known development readouts.",
         "must_not_infer": ["classification repair proves semantic gating", "unresolved or nonsignificant difference proves equivalence", "fixed label counts exclude text-label matching", "other-term implies semantically unrelated", "A/B or same/other isolates a single semantic factor"]})
    dump(OUT / "review-decisions.json", {"status": "unreviewed", "reviewer": None,
         "scope_decisions": [{"review_id": key, "decision": None, "comment": None} for key in ["S1-design", "S2-semantic-links", "S3-rule-scopes", "S4-interpretation"]],
         "relation_decisions": [{"relation_id": r["relation_id"], "human_sense_fit": None,
                                 "human_semantic_reference_fit": None, "human_rule_fit": None,
                                 "human_lexical_overlap": None, "decision": None, "comment": None} for r in rels],
         "instruction": "AI proposals live separately in relations.json. Human answers must be recorded in a separate feedback/adoption version bound to this draft manifest; this blank template is not an adopted record."})
    tsv(OUT / "new-conditions.tsv", [c for c in conditions if c["input_status"] == "new_unscored"],
        ["condition_id", "query_short_id", "lexicon_arm", "demo_arm", "lexicon_slot_material_ids", "demo_ids", "demo_answers", "prompt_tokens", "prompt_sha256"])
    tsv(OUT / "condition-coverage.tsv", conditions, ["condition_id", "query_short_id", "lexicon_arm", "demo_arm", "input_status", "prompt_tokens", "prompt_sha256"])
    write_review_documents(qdata, ddata, rels, conditions, comp, inputs)
    # Documentation additions may precede the initial seal. check/seal is CPU only.
    print(json.dumps({"status": "draft_generated_pending_cpu_check", "path": str(OUT), "new_inputs": 36, "new_relations": 36}, ensure_ascii=False))


def write_review_documents(qdata, ddata, rels, conditions, comp, inputs):
    queries = {q["short_id"]: q for q in qdata["queries"]}
    demos = {d["material"]["material_id"]: d for d in ddata["demos"]}
    lines = ["# 第二轮行为辨别实验：人工审核入口", "",
        "这是基于第一轮结果提出的开发补充草案。复用原提示、主评分、12条查询、3条词典和已采纳示例正文及答案。新增36个输入组合与36条适用关系；本页所有新关系值都是AI建议，人工决定为空。", "",
        "请按S1–S4审核；可整批接受，也可用下表编号指出异议。旧材料正文、等级和旧关系无需重新确认。", "",
        "## S1：实验范围与新增条件", "",
        "把两个异词示例包X-A/X-B在所有12条查询上的‘空词典／真实词典’补齐。总共48个异词条件：12个历史空词典条件复用，36个新增。与72个同词／无示例锚组成120条件核心；旧N的36条件独立保留，合计156个不同输入。所有查询均保留，包括原本正确和没有明显变化的条目。", "",
        "| 查询 | 已采纳全文 | 参考／等级 | 本轮新增 |", "|---|---|---|---|"]
    for q in qdata["queries"]:
        n = sum(c["query_short_id"] == q["short_id"] and c["input_status"] == "new_unscored" for c in conditions)
        lines.append(f"| {q['short_id']} | {q['query']['raw_text']} | {q['human_review']['task_label']}／{q['human_review']['attack_severity']} | {n}个 |")
    lines += ["", "G1/G3/J1/J3/B1/B3：每条新增空+X-A、L+X-A、空+X-B、L+X-B。其余六条：每条新增L+X-A、L+X-B。完整ID在[new-conditions.tsv](new-conditions.tsv)。", "",
        "| 包／来源 | 已采纳示例全文 | 答案／等级 |", "|---|---|---|"]
    for suffix in RULES:
        d = demos["CTDD-X-" + suffix]
        lines.append(f"| X-{suffix} | {d['material']['raw_text']} | {d['human_review']['task_label']}／{d['human_review']['attack_severity']} |")
    lines += ["", "两个包始终先有后无，各1条；词典独立指定，不随示例重新检索。模型只看到原任务、词典正文、示例正文与答案、查询；看不到查询参考或关系审核。", "",
        "## S2：语义联系——异词不等于完全无关", "",
        "**重点建议：‘垃圾’↔指人‘废物’只按部分语义联系partial处理。** 可迁移的是无用／低价值评价，不是完整同义或作者认可。建议涉及下方4条D→query语义边，以及2条L→demo义项边。其他跨词项语义值建议none。若你认为具体词典义项不应据这部分联系迁移，请指出相应边及理由；草案未将这些值当作已采纳事实。", "",
        "| 编号 | 词典全文 | 示例 | sense_fit建议 | 理由与边界 |", "|---|---|---|---|---|"]
    ld = [r for r in rels if r["relation_kind"] == "lexicon_to_demo"]
    for i, r in enumerate(ld, 1):
        lines.append(f"| LD{i:02d} | {r['source']['raw_text']} | {r['target']['material_id']} | {r['sense_fit']['value']} | {r['sense_fit']['rationale']} {r['sense_fit']['limitation']} |")
    lines += ["", "X-A的纸箱与三条词典均无对应义项；X-B的废物不承载北京地域或女性性污名义项。词形维度全部为absent，只指垃圾／京巴／公交车这三个指定词形不共现。", "",
        "## S3：示例规则与查询的逐条关系", "",
        "每条规则先固定‘信息是什么’，再判断对应位置。下面的partial必须有明确可迁移部分；none只排除该命名规则。不同答案不自动意味着partial，同一答案也不意味着direct。", "",
        "| 规则编号／来源 | 本轮固定的具体规则 |", "|---|---|"]
    for suffix, rule in RULES.items():
        lines.append(f"| R-{suffix}／X-{suffix} | {rule} |")
    lines += ["", "| 编号 | 查询 | 来源 | 语义建议 | 规则建议 | 规则理由 |", "|---|---|---|---|---|---|"]
    dq = [r for r in rels if r["relation_kind"] == "demo_to_query"]
    for i, r in enumerate(dq, 1):
        lines.append(f"| DQ{i:02d} | {r['target']['material_id'][4:-3]} | {r['source']['material_id']} | {r['semantic_reference_fit']['value']} | {r['rule_fit']['value']} | {r['rule_fit']['rationale']} |")
    lines += ["", "**优先核对四类边界：**", "",
        "- G1←X-A-01：建议partial，迁移行为批评与额外人身贬损的区分，具体辱骂形式不同。旧G1←同词A-01采用较窄结构而为none，其范围问题已在旧AI观察中标记，旧值保留。", 
        "- J1/B1←X-A-01：建议none；不能仅凭反问形式，补造与源相同的行为抱怨加独立口语辱骂结构。",
        "- G3/J3/B3←X-A-02：建议partial，迁移‘劝阻行为本身不自动是攻击’，但它没有示范如何处理被反对的辱称。旧同词A-02的较窄范围评级保持原样。",
        "- X-B对同一立场分支建议direct，对相反立场分支建议partial；后者只迁移辱称识别与作者立场分开的判断步骤，不迁移示例结论。", "",
        "新关系与旧字段范围可能不同，因此本轮继续不按direct/none作二元因果分组，也不把同词和异词两包当成只改变词面一个因素。每条完整片段、限制和身份见[relation-review.md](relation-review.md)。", "",
        "## S4：预先解释范围", "",
        "- J2在L+异词包下变对，只能先称该组合修复；还要看L效应的交互，才能区分是否仅由D整体抬高分数。",
        "- 异词包也有效，说明这组材料下同词共现不是达到该结果的必要条件；不能直接证明通用规则是唯一原因。",
        "- 同词包更有效，说明具体内容组合有差别；词义、措辞、长度及按文本选择标签等解释仍需另行区分。",
        "- 反对辱称文本改善时，必须同时报告实施辱称及独立口语辱骂的有攻击文本有无损失，防止把普遍偏向无当规则修复。",
        "- 垃圾家族与废物示例的候选部分语义联系单独列出，不与京巴／公交车混称‘语义完全无关’。",
        "- 保留反向、未决和无明显变化结果。没有预设实用等效阈值，不把小差异或数值未决称为等效、无影响或统计不显著。",
        "- 本轮仍是三个已暴露且共享构造的开发词项；稳定误用也可成为后续研究对象，新家族确认和内部实验另立设计。", "",
        "完整公式与分岔条件见[INTERPRETATION-PLAN.md](INTERPRETATION-PLAN.md)。", "",
        "**回复方式：**可以直接回复‘S1–S4及36条关系按建议接受’，也可以只列异议，例如‘DQxx的rule_fit改为……，因为……’。本页交付不代表已审核、已冻结或已运行。"]
    (OUT / "REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    details = ["# 新关系逐条依据", "", "AI草案；全部adoption=none。正文与来源质量继承已有采纳，新关系不继承人审。旧目标查询输出已暴露，新36组合尚无输出。", ""]
    for group, rows in [("LD", ld), ("DQ", dq)]:
        for i, r in enumerate(rows, 1):
            details += [f"## {group}{i:02d} · {r['relation_id']}", "", f"来源：{r['source']['material_id']} — {r['source']['raw_text']}", "", f"目标：{r['target']['material_id']} — {r['target']['raw_text']}", ""]
            for key in ["sense_fit", "semantic_reference_fit", "rule_fit"]:
                f = r[key]
                if f:
                    details += [f"**{key}：{f['value']}**", "", f"信息／规则：{f['information']}", "", f"来源片段：{'；'.join(s['text'] for s in f['source_spans'])}", "", f"目标片段：{'；'.join(s['text'] for s in f['target_spans'])}", "", f"理由：{f['rationale']}", "", f"限制：{f['limitation'] or '无额外限制'}", ""]
            details += [f"词形共现：{r['lexical_overlap']['value']}；指定词形：{r['lexical_overlap']['focal_forms'][0]}。人工决定：未填写。", ""]
    (OUT / "relation-review.md").write_text("\n".join(details), encoding="utf-8")
    preview = ["# 36个新增输入的模型可见全文", "", "每项引用同一个已冻结的[system任务](../../task-applicability-scoring-v1/frozen-01/model-task.txt)，下方逐项显示完整user正文。实际system+user+assistant边界及token IDs逐条保存在model-inputs.jsonl。查询答案与审核关系不进入提示。", ""]
    for p in inputs:
        preview += [f"## {p['condition_id']}", "", "```text", p["messages"][1]["content"], "```", ""]
    (OUT / "PROMPTS.md").write_text("\n".join(preview), encoding="utf-8")


def check(seal=False):
    from jsonschema import Draft202012Validator, FormatChecker
    sources = read(OUT / "sources.json")["sources"]
    for entry in sources:
        require(info(ROOT / entry["path"]) == entry, f"Source changed: {entry['path']}")
    if not seal:
        manifest = read(OUT / "draft-manifest.json")
        for entry in manifest["artifacts"] + manifest["implementation"]:
            require(info(ROOT / entry["path"]) == entry, f"Draft changed: {entry['path']}")
    design, mats = read(OUT / "design.json"), read(OUT / "materials.json")
    conditions = design["conditions"]
    rels = read(OUT / "relations.json")["records"]
    old = read(JOINT / "relations.json")["records"] + read(MATERIALS / "relations.json")["records"]
    all_relations = {r["relation_id"]: r for r in old + rels}
    require(len(conditions) == len({c["condition_id"] for c in conditions}) == 120, "Condition inventory")
    require(Counter(c["input_status"] for c in conditions) == {"historical_reuse": 84, "new_unscored": 36}, "New/reused inventory")
    require(len(rels) == 36 and Counter(r["relation_kind"] for r in rels) == {"demo_to_query": 24, "lexicon_to_demo": 12}, "Relation inventory")
    require(len(all_relations) == 132, "Duplicate/missing relation identity")
    schema = Draft202012Validator(read(PROTOCOL / "relation-record.schema.json"), format_checker=FormatChecker())
    validate = runpy.run_path(str(PROTOCOL / "validate_contract.py"))["validate_relation"]
    for r in rels:
        validate(r, schema)
        require(r["provenance"]["adoption"] == "none" and not r["provenance"]["accepted_fields"], "Human adoption invented")
        require(r["source_quality"]["provenance"]["adoption"] != "none", "Source quality missing adoption")
        require(r["lexical_overlap"]["value"] == "absent", "Focal overlap changed")
    decisions = read(OUT / "review-decisions.json")
    require(all(x["decision"] is None and x["comment"] is None for x in decisions["scope_decisions"]), "Scope review falsely completed")
    require(all(all(v is None for k, v in x.items() if k != "relation_id") for x in decisions["relation_decisions"]), "Relation review falsely completed")
    require({x["relation_id"] for x in decisions["relation_decisions"]} == {r["relation_id"] for r in rels}, "Review queue coverage")
    qsource, dsource = read(MATERIALS / "materials.json"), read(DEMOS / "materials.json")
    require(mats["queries"] == qsource["queries"] and mats["lexicons"] == qsource["lexicons"] and mats["demos"] == dsource["demos"], "Adopted material/reference changed")
    qs = {q["short_id"]: q for q in mats["queries"]}
    ds = {d["material"]["material_id"]: d for d in mats["demos"]}
    ls = {l["material"]["material_id"]: l["material"] for l in mats["lexicons"]}
    new_inputs = {p["condition_id"]: p for p in map(json.loads, (OUT / "model-inputs.jsonl").read_text().splitlines())}
    old_inputs = {p["condition_id"]: p for p in map(json.loads, (PREVIOUS / "frozen-01/model-inputs.jsonl").read_text().splitlines())}
    require(len(new_inputs) == 36 and not set(new_inputs).intersection(old_inputs), "New input IDs")
    require(not {p["prompt_sha256"] for p in new_inputs.values()}.intersection(p["prompt_sha256"] for p in old_inputs.values()), "New prompt duplicates old one")
    tok = tokenizer()
    system = (PROTOCOL / "model-task.txt").read_text()
    checks = 0
    allowed = {"request_id", "condition_id", "messages", "chat_prompt", "prompt_sha256", "input_ids", "input_ids_sha256", "prompt_tokens", "last_input_token_index", "next_token_position", "candidate_tokens"}
    for c in conditions:
        p = (new_inputs if c["input_status"] == "new_unscored" else old_inputs)[c["condition_id"]]
        require(set(p) == allowed, "Model input metadata leak or missing field")
        lex = ls[c["lexicon_slot_material_ids"][0]] if c["lexicon_slot_material_ids"] else None
        reconstructed = render(qs[c["query_short_id"]]["query"], lex, [ds[d] for d in c["demo_ids"]], system, tok)
        require(all(p[k] == v for k, v in reconstructed.items()), "Visible input/token reconstruction differs")
        require(c["prompt_sha256"] == p["prompt_sha256"] and c["prompt_tokens"] == p["prompt_tokens"], "Design prompt binding differs")
        require(tok.apply_chat_template(p["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False) == p["input_ids"], "Direct chat template differs")
        for label, tid in p["candidate_tokens"].items():
            require(tok.encode(p["chat_prompt"] + label, add_special_tokens=False) == p["input_ids"] + [tid], "Unstable bare answer boundary")
            checks += 1
        if c["demo_ids"]:
            require(c["demo_answers"] == ["有", "无"], "Demo order/labels changed")
    by_id = {c["condition_id"]: c for c in conditions}
    apps = read(OUT / "relation-appearances.json")
    require(len(apps["records"]) == 504 and len(apps["context_bindings"]) == 192, "Presence coverage")
    for a in apps["records"]:
        c, r = by_id[a["condition_id"]], all_relations[a["relation_id"]]
        visible = set(c["lexicon_slot_material_ids"] + c["demo_ids"] + [c["query_id"]])
        require(a["source_present"] == (r["source"]["material_id"] in visible) and a["target_present"] == (r["target"]["material_id"] in visible), "Presence incorrect")
        require(a["prompt_sha256"] == c["prompt_sha256"], "Presence input hash")
    for r in rels:
        expected = [{k: v for k, v in a.items() if k != "relation_id"} for a in apps["records"] if a["relation_id"] == r["relation_id"]]
        require(r["presentation_refs"] == expected and expected, "New relationship presentation coverage")
    for b in apps["context_bindings"]:
        dq, ld = all_relations[b["demo_query_relation_id"]], all_relations[b["candidate_lexicon_demo_relation_id"]]
        require(dq["source"] == ld["target"], "L-D/D-Q composition mismatch")
        require(b["lexicon_present"] == (by_id[b["condition_id"]]["lexicon_arm"] == "L"), "Candidate L mistaken for presented L")
    analysis = read(OUT / "analysis-plan.json")
    require(analysis["comparisons"] == comparisons(conditions, mats["queries"]), "Formula or exposure metadata changed")
    require(len(analysis["comparisons"]) == 168, "Comparison count")
    for c in analysis["comparisons"]:
        require(sum(t["coefficient"] for t in c["terms"]) == 0, "Noncentered contrast")
        require(all(by_id[t["condition_id"]]["query_id"] == c["query_id"] for t in c["terms"]), "Cross-query contrast")
    bridges = read(OUT / "historical-bridges.json")
    require(len(bridges["records"]) == 84 and len(bridges["old_N_inputs"]) == 36, "Historical coverage")
    for b in bridges["records"]:
        require(info(ROOT / b["score_ref"]["path"]) == b["score_ref"], "Historical score receipt changed")
        require(b["prompt_sha256"] == old_inputs[b["condition_id"]]["prompt_sha256"], "Historical request mismatch")
    import re
    for path in OUT.glob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("https:", "http:", "#")):
                require((path.parent / target.split("#")[0]).is_file(), f"Broken link: {path.name}: {target}")
    result = {"status": "pass", "source_hashes": len(sources), "new_inputs": 36, "reused_inputs_reconstructed": 84,
              "bare_answer_boundaries": checks, "new_relations_schema_span_checks": len(rels),
              "new_DQ_semantic_counts": dict(Counter(r["semantic_reference_fit"]["value"] for r in rels if r["semantic_reference_fit"])),
              "new_DQ_rule_counts": dict(Counter(r["rule_fit"]["value"] for r in rels if r["rule_fit"])),
              "new_LD_sense_counts": dict(Counter(r["sense_fit"]["value"] for r in rels if r["sense_fit"])),
              "presence_records": 504, "context_bindings": 192, "comparisons": 168,
              "comparison_exposure": dict(Counter(c["exposure"] for c in analysis["comparisons"])),
              "new_prompt_token_range": [min(p["prompt_tokens"] for p in new_inputs.values()), max(p["prompt_tokens"] for p in new_inputs.values())],
              "human_decisions_added": 0, "model_weights_loaded": False, "GPU_forward_performed": False,
              "old_sources_unchanged": True, "numeric_qualification": False}
    if seal:
        require(not (OUT / "draft-manifest.json").exists(), "Draft already sealed; prepare a separate revision")
        dump(OUT / "cpu-check.json", result)
        dump(OUT / "draft-manifest.json", {"schema_version": "cross-term-behavior-review-draft-manifest/v1",
             "created_at": datetime.now(timezone.utc).isoformat(), "status": "reviewable_draft_not_human_freeze",
             "human_adoption": False, "GPU_execution": False,
             "artifacts": [info(p) for p in sorted(OUT.iterdir()) if p.is_file()],
             "implementation": [info(Path(__file__).resolve())], "sources": info(OUT / "sources.json")})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "seal", "check"])
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        check(seal=args.command == "seal")
