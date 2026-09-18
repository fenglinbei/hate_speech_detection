"""Record explicit review adoption and freeze the second behavioral input design.

CPU only. The original review draft and completed GPU run are read-only sources.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import re
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/experiment-plans/cross-term-behavior-discrimination-v1"
DRAFT = BASE / "draft-01"
FROZEN = BASE / "frozen-01"
PROTOCOL = BASE.parent / "task-applicability-scoring-v1/frozen-01"
PREPARER = ROOT / "scripts/review/prepare_cross_term_behavior_discrimination_v1.py"
FEEDBACK = BASE / "feedback-01.json"
EXPECTED_DRAFT_SHA = "51d5028484f3f5d2efb259157b9fb35c3fe9636deaa752de78fa109a23a6ade9"
USER_REPLY = "S1–S4及36条关系按建议接受"
UNCHANGED_FILES = [
    "model-inputs.jsonl", "materials.json", "new-conditions.tsv", "condition-coverage.tsv",
    "historical-bridges.json", "reused-relations.json", "sources.json", "PROMPTS.md",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def info(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": digest(data)}


def require(value, message):
    if not value:
        raise ValueError(message)


def verify(entry):
    require(info(ROOT / entry["path"]) == entry, f"Pinned source changed: {entry['path']}")


def write_new(path, value):
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def record_fields(relation):
    return (["/sense_fit", "/lexical_overlap"] if relation["relation_kind"] == "lexicon_to_demo"
            else ["/semantic_reference_fit", "/rule_fit", "/lexical_overlap"])


def sections():
    text = (DRAFT / "REVIEW.md").read_text()
    starts = list(re.finditer(r"^## (S[1-4])：(.+)$", text, flags=re.MULTILINE))
    require(len(starts) == 4, "Four reviewed sections missing")
    return [{"scope_id": m[1], "title": m[2],
             "reviewed_text": text[m.start():starts[i+1].start() if i+1 < len(starts) else len(text)],
             "adoption": "bulk", "decision": "accepted_as_proposed"}
            for i, m in enumerate(starts)]


def validate_draft(tokenize=False):
    manifest = read(DRAFT / "draft-manifest.json")
    require(info(DRAFT / "draft-manifest.json")["sha256"] == EXPECTED_DRAFT_SHA,
            "User acceptance must bind the exact delivered draft")
    for entry in manifest["artifacts"] + manifest["implementation"]:
        verify(entry)
    for entry in read(DRAFT / "sources.json")["sources"]:
        verify(entry)
    audit = read(BASE / "audits/draft-01-independent.json")
    require(audit["status"] == "pass" and audit["draft_manifest_sha256"] == EXPECTED_DRAFT_SHA,
            "Independent draft receipt mismatch")
    for r in read(DRAFT / "relations.json")["records"]:
        require(r["provenance"]["adoption"] == "none", "Original AI record changed")
    decisions = read(DRAFT / "review-decisions.json")
    require(all(r["decision"] is None for r in decisions["scope_decisions"] + decisions["relation_decisions"]),
            "Original blank human template changed")
    if tokenize:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            runpy.run_path(str(PREPARER))["check"]()
        receipt = json.loads(capture.getvalue())
        require(receipt["status"] == "pass" and receipt["bare_answer_boundaries"] == 240,
                "Original byte/token/boundary audit failed")
        return receipt
    return None


def transform_relations(old, feedback, id_map):
    output = copy.deepcopy(old)
    output["status"] = "human_bulk_adopted_scoped_relations_frozen"
    output["human_adoption"] = True
    output["feedback"] = info(FEEDBACK)
    output["source_draft"] = info(DRAFT / "relations.json")
    for r in output["records"]:
        parent_id = r["relation_id"]
        r["relation_id"] = id_map[parent_id]
        r["supersedes"] = parent_id
        r["introduced_lexicon_relation_ids"] = [id_map.get(x, x) for x in r["introduced_lexicon_relation_ids"]]
        r["provenance"] = {
            "authorship": r["provenance"]["authorship"], "review_kind": "human_with_ai",
            "adoption": "bulk", "accepted_fields": record_fields(r),
            "decision_ref": {k: v for k, v in info(FEEDBACK).items() if k != "bytes"},
            "recorded_at": feedback["recorded_at_utc"], "exposure": r["provenance"]["exposure"],
        }
    return output


def transform_design(feedback):
    d = copy.deepcopy(read(DRAFT / "design.json"))
    d.update(status="human_reviewed_material_relation_and_input_design_frozen",
             human_design_adoption=True, scientific_input_freeze=True, runtime_frozen=False,
             GPU_numerical_qualification=False, frozen_at_utc=feedback["recorded_at_utc"],
             feedback=info(FEEDBACK), source_draft_manifest=info(DRAFT / "draft-manifest.json"))
    for c in d["conditions"]:
        c["human_design_adoption"] = True
    return d


def transform_analysis():
    a = copy.deepcopy(read(DRAFT / "analysis-plan.json"))
    a["status"] = "registered_development_readouts_before_new_36_outcomes"
    a["feedback"] = info(FEEDBACK)
    a["source_review_plan"] = info(DRAFT / "analysis-plan.json")
    for c in a["comparisons"]:
        c["human_adopted"] = True
    return a


def transform_appearances(id_map):
    a = copy.deepcopy(read(DRAFT / "relation-appearances.json"))
    for row in a["records"]:
        row["relation_id"] = id_map.get(row["relation_id"], row["relation_id"])
    for row in a["context_bindings"]:
        for key in ["demo_query_relation_id", "candidate_lexicon_demo_relation_id"]:
            row[key] = id_map.get(row[key], row[key])
    return a


def create_freeze():
    require(not FROZEN.exists() and not FEEDBACK.exists(), "Adoption already exists; do not overwrite")
    draft_receipt = validate_draft(tokenize=True)
    selector = read(BASE / "current.json")
    require(selector["status"] == "draft_awaiting_human_review" and
            selector["draft_manifest"]["sha256"] == EXPECTED_DRAFT_SHA, "Unexpected review selector")
    for key in ["draft_manifest", "review_entry", "cpu_check", "independent_audit"]:
        verify(selector[key])
    history = BASE / "selector-history"
    history.mkdir(exist_ok=True)
    history_path = history / "draft-01.json"
    require(not history_path.exists(), "Selector history already exists")
    history_path.write_bytes((BASE / "current.json").read_bytes())

    recorded = datetime.now(timezone.utc).isoformat()
    old_relations = read(DRAFT / "relations.json")
    id_map = {r["relation_id"]: r["relation_id"] + "-review01" for r in old_relations["records"]}
    feedback = {
        "schema_version": "cross-term-behavior-review-feedback/v1",
        "decision_id": "CTD2-review-20260917-01", "recorded_at_utc": recorded,
        "source": "explicit_user_message_in_current_task", "user_message_verbatim": USER_REPLY,
        "authorship_of_proposals": "ai", "review_kind": "human_with_ai", "adoption": "bulk",
        "bound_draft_manifest": info(DRAFT / "draft-manifest.json"),
        "bound_review_entry": info(DRAFT / "REVIEW.md"),
        "bound_interpretation_plan": info(DRAFT / "INTERPRETATION-PLAN.md"),
        "bound_analysis_plan": info(DRAFT / "analysis-plan.json"),
        "scope": "S1-S4 and all 36 new relation proposals, with their exact information, spans, rationales and limitations; separate adoption/input-design freeze, preserving original draft and historical results.",
        "scope_decisions": sections(),
        "relation_decisions": [
            {"draft_relation_id": r["relation_id"], "adopted_relation_id": id_map[r["relation_id"]],
             "review_id": ("LD" if r["relation_kind"] == "lexicon_to_demo" else "DQ") +
                          f"{i + 1 if i < 12 else i - 11:02d}",
             "source_id": r["source"]["material_id"], "target_id": r["target"]["material_id"],
             "accepted_fields": record_fields(r),
             "values": {p[1:]: r[p[1:]] for p in record_fields(r)},
             "review_mode": "bulk_acceptance_not_individual_question_answer"}
            for i, r in enumerate(old_relations["records"])
        ],
        "input_conditions_accepted": [c["condition_id"] for c in read(DRAFT / "design.json")["conditions"]],
        "comparison_ids_accepted": [c["comparison_id"] for c in read(DRAFT / "analysis-plan.json")["comparisons"]],
        "new_material_text_or_query_reference_decisions": 0,
        "old_relation_revisions": 0, "old_AI_scope_observations_newly_adjudicated": 0,
        "new_GPU_schedule_or_device_decision": False,
        "execution_scope_note": "This reply adopts the reviewed design; it does not specify a new runtime, bridge qualification or GPU allocation. Prior session authorization is preserved separately.",
    }
    write_new(FEEDBACK, feedback)
    FROZEN.mkdir()
    for name in UNCHANGED_FILES:
        (FROZEN / name).write_bytes((DRAFT / name).read_bytes())
    adopted = transform_relations(old_relations, feedback, id_map)
    write_new(FROZEN / "relations.json", adopted)
    write_new(FROZEN / "design.json", transform_design(feedback))
    write_new(FROZEN / "analysis-plan.json", transform_analysis())
    write_new(FROZEN / "relation-appearances.json", transform_appearances(id_map))
    write_new(FROZEN / "change-log.json", {
        "source_draft_manifest": info(DRAFT / "draft-manifest.json"), "feedback": info(FEEDBACK),
        "relation_id_map": id_map, "relation_information_or_values_changed": False,
        "exact_byte_copies": UNCHANGED_FILES, "scientific_prompt_or_formula_changed": False,
        "changes": ["New relation IDs supersede AI draft IDs; field-scoped bulk provenance added.",
                    "New relation references in appearances and candidate links remapped consistently.",
                    "Design and comparison adoption states now accepted; runtime/numeric states remain pending."]})
    write_new(FROZEN / "review-completion.json", {
        "status": "complete_by_explicit_bulk_adoption", "feedback": info(FEEDBACK),
        "scopes": [{"scope_id": s["scope_id"], "decision": "accepted_as_proposed"} for s in feedback["scope_decisions"]],
        "relations": [{"review_id": d["review_id"], "draft_relation_id": d["draft_relation_id"],
                       "relation_id": d["adopted_relation_id"], "decision": "accepted_as_proposed",
                       "accepted_fields": d["accepted_fields"], "adoption": "bulk"}
                      for d in feedback["relation_decisions"]],
        "adopted_relation_count": 36, "adopted_dimension_fields": 96,
        "pending_new_relations": 0, "original_AI_authorship_preserved": True,
        "original_draft_human_fields_unchanged": True})
    write_documents(adopted)
    receipt = check(tokenize=False, pending_manifest=True)
    receipt["tokenizer_rechecked"] = True
    receipt["bare_answer_boundaries"] = draft_receipt["bare_answer_boundaries"]
    receipt["tokenizer_scope"] = "All 120 reviewed inputs; frozen new input bytes and all 84 historical bridges are identical."
    receipt_path = BASE / "audits/freeze-01-cpu.json"
    write_new(receipt_path, receipt)
    write_new(FROZEN / "manifest.json", {
        "schema_version": "cross-term-behavior-material-input-freeze/v1", "created_at_utc": recorded,
        "status": "immutable_human_adopted_relation_and_input_design",
        "artifacts": [info(p) for p in sorted(FROZEN.iterdir()) if p.is_file()],
        "sources": [info(FEEDBACK), info(DRAFT / "draft-manifest.json"), info(DRAFT / "sources.json"),
                    info(BASE / "audits/draft-01-independent.json"), info(receipt_path), info(history_path)],
        "implementation": [info(Path(__file__).resolve())],
        "human_adoption": True, "scientific_input_design_frozen": True,
        "runtime_frozen": False, "GPU_numerical_qualification": False, "new_model_forward_performed": False})
    new_selector = {
        "schema_version": "cross-term-behavior-review-selector/v1", "updated_at_utc": recorded,
        "status": "human_review_and_scientific_input_design_freeze_complete",
        "frozen_directory": str(FROZEN.relative_to(ROOT)), "manifest": info(FROZEN / "manifest.json"),
        "feedback": info(FEEDBACK), "review_completion": info(FROZEN / "review-completion.json"),
        "CPU_audit": info(receipt_path), "historical_draft_selector": info(history_path),
        "source_draft_manifest": info(DRAFT / "draft-manifest.json"),
        "adopted_scopes": ["S1", "S2", "S3", "S4"], "adopted_new_relations": 36,
        "reused_prior_relations": 96, "new_scientific_inputs": 36, "reused_core_inputs": 84,
        "core_inputs": 120, "historical_N_inputs_retained": 36, "registered_comparisons": 168,
        "scientific_input_design_frozen": True, "runtime_frozen": False,
        "GPU_numerical_qualification": False, "new_model_forward_performed": False,
        "next_stage": "separate execution binding, historical numerical bridge and new-input qualification"}
    temporary = BASE / ".current-adoption-01.tmp"
    write_new(temporary, new_selector)
    temporary.replace(BASE / "current.json")
    print(json.dumps({"status": "adoption_and_input_freeze_complete", "manifest": info(FROZEN / "manifest.json"),
                      "adopted_new_relations": 36, "GPU_forward_performed": False}, ensure_ascii=False, indent=2))


def write_documents(adopted):
    (FROZEN / "README.md").write_text("""# 第二轮行为辨别实验：审核与输入设计已冻结

用户明确确认：“S1–S4及36条关系按建议接受”。本包记录对精确草案的整批采纳，AI起草身份与逐项问答裁决保持区别。原草案及其空人工模板作为历史版本保存。

已完成S1–S4和36条新关系的审核，36个新增科学输入＋84个历史核心输入组成120条件核心；旧36个N输入完整保留，合计156个不同输入。全部模型提示、来源正文、答案和168项比较公式与审阅版本一致。

## 采纳范围

- S1：新增条件、全部12条查询的覆盖，以及历史锚点和N对照的保留。
- S2：12条L→异词示例义项关系、24条D→查询语义关系及指定词形比较。垃圾／废物的4条D语义边和2条L义项边按partial采纳，保留各自范围和限制。
- S3：24条D→查询规则关系，6 direct、10 partial、8 none；精确信息、片段、理由和限制一同绑定，不将关系值合并成总适用性分数。
- S4：分别报告分类修复、L的增量与交互、同词／异词整包差异，保留反向和未决结果及开发材料范围。没有新增内部机制主张。

原已采纳的84条关系、12条L→query和11条AI范围观察保持原身份，本次未追认修改。输入中L独立分配，关联L→demo不等于该条件实际呈现L。

## 当前文件

- [反馈记录](../feedback-01.json)：原话、草案哈希、S1–S4原审阅段落及36条字段级采纳值。
- [审核完成记录](review-completion.json)：36/36关系完成，96个适用维度字段整批采纳；新关系待审数为0。
- [已采纳关系表](relation-review.md)及[机器记录](relations.json)：新的采纳层，保留AI作者、历史暴露和来源质量。
- [科学输入设计](design.json)与[36条新模型输入](model-inputs.jsonl)：运行状态仍为未评分。
- [提示全文预览](PROMPTS.md)：字节与原审阅版一致，模型看不到查询参考或关系评级。
- [分析计划](analysis-plan.json)：168项注册开发比较；138项含未评分新端点，30项仅含已暴露旧端点。
- [原解释判据](../draft-01/INTERPRETATION-PLAN.md)：按S4采纳，保持原始文字；其中“提案／待审”是草案时点描述，当前状态以本采纳层为准。
- [资料呈现记录](relation-appearances.json)：504条呈现记录与192条L→D→query关联，新关系ID已统一映射。
- [CPU核查](../audits/freeze-01-cpu.json)：采纳、schema、源文件、输入字节、关系关联和公式校验。

关系理由中保留的“待人审”等草案措辞属于被确认的原始说明；最新采纳状态由provenance、feedback与review-completion明确记录，不改写历史措辞来抹除AI来源。

## 后续执行状态

科学输入与关系审核已完成。数值运行环境、设备、历史评分桥接和新输入数值资格需要单独绑定；本包没有进行模型forward，也不把旧误差界直接套用到新增输入。第一轮已完成run保持终态。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 .conda/stage1-p0/bin/python scripts/review/freeze_cross_term_behavior_discrimination_v1.py check --tokenize
```

修改本冻结包需要新版本。复用历史分数须保留原始物理评分身份，不能将84个旧端点计为新增独立重复或未见结果的确认材料。
""", encoding="utf-8")
    lines = ["# 已采纳的36条新关系", "", "对应用户对S1–S4及36条关系的整批确认。逐条原文、片段和完整限制见relations.json；正文与草案相同，只新增采纳身份及关系ID。", "",
             "| 审核编号 | 来源 | 目标 | 义项／语义 | 规则 | 指定词形 |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(adopted["records"]):
        label = ("LD" + f"{i+1:02d}") if i < 12 else ("DQ" + f"{i-11:02d}")
        meaning = r["sense_fit"] or r["semantic_reference_fit"]
        rule = r["rule_fit"]["value"] if r["rule_fit"] else "不适用字段"
        lines.append(f"| {label} | {r['source']['material_id']} | {r['target']['material_id']} | {meaning['value']} | {rule} | {r['lexical_overlap']['value']} |")
    lines += ["", "所有值都仅适用于记录中的具体信息、目标位置及限制；none不代表整份示例对模型绝无作用。原材料正文、等级和答案没有本次新增裁决。"]
    (FROZEN / "relation-review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(tokenize=False, pending_manifest=False):
    draft_receipt = validate_draft(tokenize=tokenize)
    if not pending_manifest:
        manifest = read(FROZEN / "manifest.json")
        for entry in manifest["artifacts"] + manifest["sources"] + manifest["implementation"]:
            verify(entry)
        actual = sorted(p.name for p in FROZEN.iterdir() if p.is_file() and p.name != "manifest.json")
        require(actual == sorted(Path(e["path"]).name for e in manifest["artifacts"]), "Frozen inventory changed")
    feedback = read(FEEDBACK)
    require(feedback["user_message_verbatim"] == USER_REPLY and feedback["adoption"] == "bulk",
            "Human reply or adoption mode changed")
    require(feedback["bound_draft_manifest"] == info(DRAFT / "draft-manifest.json"), "Feedback draft binding")
    require(feedback["scope_decisions"] == sections(), "S1-S4 adoption scope changed")
    old = read(DRAFT / "relations.json")
    id_map = {r["relation_id"]: r["relation_id"] + "-review01" for r in old["records"]}
    adopted = read(FROZEN / "relations.json")
    require(adopted == transform_relations(old, feedback, id_map), "Unapproved relation/provenance changes")
    decisions = {d["draft_relation_id"]: d for d in feedback["relation_decisions"]}
    require(set(decisions) == set(id_map), "Feedback relation coverage")
    from jsonschema import Draft202012Validator, FormatChecker
    validator = Draft202012Validator(read(PROTOCOL / "relation-record.schema.json"), format_checker=FormatChecker())
    validate = runpy.run_path(str(PROTOCOL / "validate_contract.py"))["validate_relation"]
    dimensions = 0
    for original, new in zip(old["records"], adopted["records"]):
        validate(new, validator)
        d = decisions[original["relation_id"]]
        require(d["adopted_relation_id"] == new["relation_id"] and d["accepted_fields"] == record_fields(original),
                "Accepted fields/IDs mismatch")
        require(d["values"] == {p[1:]: original[p[1:]] for p in d["accepted_fields"]}, "Adopted values changed")
        for key in ["source", "target", "source_quality", "sense_fit", "semantic_reference_fit", "rule_fit",
                    "lexical_overlap", "presentation_refs", "family_id"]:
            require(new[key] == original[key], f"Scientific relation field changed: {key}")
        require(new["provenance"]["exposure"] == original["provenance"]["exposure"], "Exposure reset")
        dimensions += len(d["accepted_fields"])
    require(dimensions == 96, "Adoption dimension count")
    for name in UNCHANGED_FILES:
        require((FROZEN / name).read_bytes() == (DRAFT / name).read_bytes(), f"Reviewed bytes changed: {name}")
    require(read(FROZEN / "design.json") == transform_design(feedback), "Unapproved design changes")
    require(read(FROZEN / "analysis-plan.json") == transform_analysis(), "Unapproved scoring/formula changes")
    require(read(FROZEN / "relation-appearances.json") == transform_appearances(id_map), "Presence/link ID mapping")
    require(feedback["input_conditions_accepted"] == [c["condition_id"] for c in read(FROZEN / "design.json")["conditions"]], "Input adoption coverage")
    require(feedback["comparison_ids_accepted"] == [c["comparison_id"] for c in read(FROZEN / "analysis-plan.json")["comparisons"]], "Comparison adoption coverage")
    completion = read(FROZEN / "review-completion.json")
    require(len(completion["relations"]) == 36 and completion["pending_new_relations"] == 0 and
            {d["relation_id"] for d in completion["relations"]} == set(id_map.values()), "Review completion mismatch")
    all_records = adopted["records"][:]
    for entry in read(FROZEN / "reused-relations.json")["records"]:
        old_r = next(r for r in read(ROOT / entry["source_file"])["records"] if r["relation_id"] == entry["relation_id"])
        all_records.append(old_r)
    by_relation = {r["relation_id"]: r for r in all_records}
    require(len(all_records) == len(by_relation) == 132, "New/old relation inventory")
    conditions = {c["condition_id"]: c for c in read(FROZEN / "design.json")["conditions"]}
    apps = read(FROZEN / "relation-appearances.json")
    for a in apps["records"]:
        r, c = by_relation[a["relation_id"]], conditions[a["condition_id"]]
        visible = set([c["query_id"]] + c["lexicon_slot_material_ids"] + c["demo_ids"])
        require(a["source_present"] == (r["source"]["material_id"] in visible) and
                a["target_present"] == (r["target"]["material_id"] in visible), "Visibility mismatch")
        require(a["prompt_sha256"] == c["prompt_sha256"], "Appearance prompt hash mismatch")
    for b in apps["context_bindings"]:
        require(by_relation[b["demo_query_relation_id"]]["source"] ==
                by_relation[b["candidate_lexicon_demo_relation_id"]]["target"], "L-D-Q endpoint mismatch")
        require(b["lexicon_present"] == (conditions[b["condition_id"]]["lexicon_arm"] == "L"), "Unpresented L treated as visible")
    for document in FROZEN.glob("*.md"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text()):
            if pending_manifest and (document.parent / target).resolve() == BASE / "audits/freeze-01-cpu.json":
                continue  # This successful check produces that receipt; final check requires it.
            require((document.parent / target).is_file(), f"Broken frozen-document link: {target}")
    if not pending_manifest:
        selector = read(BASE / "current.json")
        require(selector["manifest"] == info(FROZEN / "manifest.json") and
                selector["feedback"] == info(FEEDBACK), "Current adoption selector mismatch")
    return {"status": "pass", "adopted_scopes": ["S1", "S2", "S3", "S4"],
            "adopted_new_relations": 36, "field_scoped_dimensions": dimensions,
            "reused_prior_relations": 96, "pending_new_relations": 0,
            "DQ_rule_counts": dict(Counter(r["rule_fit"]["value"] for r in adopted["records"] if r["rule_fit"])),
            "DQ_semantic_counts": dict(Counter(r["semantic_reference_fit"]["value"] for r in adopted["records"] if r["semantic_reference_fit"])),
            "LD_sense_counts": dict(Counter(r["sense_fit"]["value"] for r in adopted["records"] if r["sense_fit"])),
            "new_prompt_bytes_unchanged": 36, "reused_core_inputs": 84,
            "unchanged_comparison_formulas": 168, "appearance_records": len(apps["records"]),
            "context_bindings": len(apps["context_bindings"]), "exact_byte_copy_files": len(UNCHANGED_FILES),
            "draft_human_fields_preserved": True, "prior_human_decisions_unchanged": True,
            "original_AI_authorship_preserved": True, "tokenizer_rechecked": tokenize,
            "bare_answer_boundaries": draft_receipt["bare_answer_boundaries"] if draft_receipt else 0,
            "scientific_input_design_frozen": True, "runtime_frozen": False,
            "GPU_numerical_qualification": False, "new_model_forward_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "check"])
    parser.add_argument("--tokenize", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        create_freeze()
    else:
        print(json.dumps(check(tokenize=args.tokenize), ensure_ascii=False, indent=2))
