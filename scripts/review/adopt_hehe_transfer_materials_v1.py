"""Persist the four named material approvals and validate eight CPU-only inputs.

This is a material adoption command, not a scientific GPU execution command.
It refuses existing outputs and never modifies old draft/scientific artifacts.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from diagnostics.case_attention_inputs_v1 import (
    build_input, canonical, digest, info, read, require, tokenizer, verify, write,
)

WORK = ROOT / "reviews/hehe-transfer-candidates-v1"
DRAFT = WORK / "draft-01"
OUT = WORK / "adopted-01"
PUBLIC = ROOT / "docs/research/experiment-plans/hehe-transfer-v1"
TASK = ROOT / "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt"
PARENT = ROOT / "reviews/hehe-sense-context-v1/prepared-01/materials.json"
IDS = ["T01", "T02", "T03", "T04"]
REFERENCES = dict(zip(IDS, ["无", "无", "有", "有"]))
EXPECTED_POSITIONS = dict(zip(IDS, [800, 814, 798, 816]))
USER_MESSAGE = "T01到04都可以纳入，T03可以按照原标签判断"


def text_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as f:
        f.write(text)


def main():
    require(not OUT.exists(), "Adopt into a new directory; do not overwrite")
    require(not (WORK / "feedback-01.json").exists(), "Feedback already recorded")
    require(not (PUBLIC / "current.json").exists(), "Material selector already exists")
    audit = read(DRAFT / "audit.json")
    for source in audit["artifacts"] + [audit["builder"]]:
        verify(dict(source, path=str(ROOT / source["path"])))
    shortlist = {r["item_id"]: r for r in read(DRAFT / "shortlist.json")["items"]}
    pool = {r["key"]: r for r in read(DRAFT / "candidate-pool.json")["records"]}
    parent = read(PARENT)
    definitions = parent["dictionaries"][:2]
    require([d["dictionary_id"] for d in definitions] == ["D01", "D02"], "Definition IDs")
    require(definitions == read(DRAFT / "shortlist.json")["definitions"], "Definition drift")
    raw_sources = {}
    for item_id in IDS:
        row = pool[shortlist[item_id]["key"]]
        path = ROOT / row["source_file"]
        raw_sources.setdefault(path, read(path))
        source = raw_sources[path][row["record_index_zero_based"]]
        require(source == row["source_record"] and source["content"] == shortlist[item_id]["text"], "Source drift")
    require(pool[shortlist["T03"]["key"]]["source_quadruples"][0]["hateful"] == "hate", "T03 source label")
    tok, system = tokenizer(), TASK.read_text(encoding="utf-8")
    rows, materials, references, positions = [], [], [], []
    for item_id in IDS:
        item, source = shortlist[item_id], pool[shortlist[item_id]["key"]]
        label_kind = "user_explicit_source_label_inheritance" if item_id == "T03" else "user_named_bulk_adoption_of_proposed_binary_reference"
        material = {
            "query_id": item_id, "source_id": source["source_id"], "corpus": source["corpus"],
            "text": item["text"], "text_sha256": source["text_sha256"],
            "role": "group_reference_candidate_with_inherited_label" if item_id == "T03" else item["role"],
            "source_file": source["source_file"], "record_index_zero_based": source["record_index_zero_based"],
            "source_quadruples": source["source_quadruples"], "decision": "accept",
            "selection_note_from_draft": item["reason"], "new_severity_adjudication": None,
            "historical_exposure": {
                "prior_material_hits_within_scanned_scope": source["prior_material_hits_within_scanned_scope"],
                "historical_corpus_copies": source["historical_corpus_copies"],
                "human_seen_predictions_history": source["human_seen_predictions_history"],
            },
        }
        materials.append(material)
        references.append({"query_id": item_id, "source_id": source["source_id"],
            "reference": REFERENCES[item_id], "hate": "hate" if REFERENCES[item_id] == "有" else "non-hate",
            "reference_basis": label_kind, "severity": None, "new_user_message": USER_MESSAGE,
            "reference_is_prompt_content": False,
            "interpretation_limit": "按用户要求继承原标签；不新增族群身份、严重度或纯词义裁决。" if item_id == "T03"
                else "采用展示的原文和二元参考；不新增严重度或资源适用性逐项裁决。"})
        for definition in definitions:
            lexicon = deepcopy(parent["base_lexicon"])
            next(e for e in lexicon if e["term"] == "嘿嘿")["senses"][0]["definition"] = definition["definition"]
            req = build_input({"query_id": item_id, "query_text": item["text"], "lexicon": lexicon, "demos": []},
                              ("L", False, "definition"), tok, system)
            req["request_id"] = f"htr-{item_id}-{definition['dictionary_id']}"
            req["dictionary_id"] = definition["dictionary_id"]
            expected = next(r for r in item["cpu_token_geometry_preview"] if r["dictionary_id"] == definition["dictionary_id"])
            require(req["prompt_sha256"] == expected["prompt_sha256"], "Adopted prompt differs from draft preview")
            focal = req["roles"]["query_focal"]
            require(focal == [EXPECTED_POSITIONS[item_id]], "Focal positions")
            pre = [focal[0] - 1]
            require(pre == expected["pre_positions"] and set(pre) <= set(req["roles"]["query_all"]), "Preceding control")
            require(req["token_text"][pre[0]] == ("你的" if item_id == "T03" else "，"), "Preceding text")
            req["patch_position_sets"] = {"focal": focal, "pre": pre}
            req["capture_positions"] = sorted(focal + pre)
            req["capture_prefix_length"] = focal[0] + 1
            prefix = tok.encode(req["prompt_text"][:req["token_offsets"][focal[0]][1]], add_special_tokens=False)
            require(prefix == req["input_ids"][:len(prefix)] and len(prefix) == req["capture_prefix_length"], "Exact focal prefix")
            rows.append(req)
            positions.append({"request_id": req["request_id"], "tokens": req["prompt_tokens"],
                "focal_positions": focal, "focal_text": [req["token_text"][i] for i in focal],
                "pre_positions": pre, "pre_text": [req["token_text"][i] for i in pre],
                "prefix_token_count": len(prefix), "prefix_exact": True})
    require(len(rows) == len({r["request_id"] for r in rows}) == 8, "Eight-input cross")
    for item_id in IDS:
        a, b = [r for r in rows if r["query_id"] == item_id]
        require(a["prompt_tokens"] == b["prompt_tokens"] and a["roles"] == b["roles"], "Paired geometry alignment")
        require(a["prompt_text"].replace(definitions[0]["definition"], definitions[1]["definition"]) == b["prompt_text"], "Only focal definition changes")
        span_a = next(s for s in a["spans"] if s["id"] == "lex-0419:definition")
        span_b = next(s for s in b["spans"] if s["id"] == "lex-0419:definition")
        require(span_a["token_positions"] == span_b["token_positions"], "Dictionary token alignment")
        changed = {i for i, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y}
        require(bool(changed) and changed <= set(span_a["token_positions"]), "Token changes outside focal definition")
    require("torch" not in sys.modules, "This command must remain tokenizer-only")

    OUT.mkdir(parents=True)
    feedback = {"schema": "hehe-transfer-feedback/v1", "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "user_message": USER_MESSAGE, "accepted_items": IDS, "T03_reference_policy": "use_original_hate_label_as_有",
                "scope": "Adopt the four displayed texts and their binary references; no old review record overwritten.",
                "source_draft": info(DRAFT / "REVIEW.md"), "source_shortlist": info(DRAFT / "shortlist.json")}
    write(WORK / "feedback-01.json", feedback)
    adoption = {"schema": "hehe-transfer-adoption/v1", "status": "all_four_materials_accepted",
                "user_message": USER_MESSAGE, "feedback": info(WORK / "feedback-01.json"),
                "accepted_items": IDS, "not_selected_items": ["B01", "B02"], "pending_material_items": [],
                "references": REFERENCES, "T03_label_inherited_not_new_semantic_adjudication": True,
                "position_policy": "Carry forward displayed equal-count immediately preceding query-token rule.",
                "position_policy_separate_user_quote": None,
                "COLD_materials_included_by_named_adoption": ["T01", "T02", "T04"],
                "new_severity_or_group_fields_adjudicated": False,
                "scope_completed_here": "material adoption and CPU prompt/token verification",
                "GPU_started": False}
    write(OUT / "adoption.json", adoption)
    write(OUT / "materials.json", {"queries": materials, "dictionaries": definitions,
                                    "base_lexicon": parent["base_lexicon"], "with_demos": False})
    write(OUT / "analysis-references.json", {"references": references, "worker_must_not_read": True})
    write(OUT / "positions.json", {"records": positions, "zero_based": True,
        "punctuation_controls": ["T01", "T02", "T04"], "control_expected_zero": False,
        "same_count_and_precedence_not_wordclass_or_norm_matching": True})
    design = {"schema": "hehe-transfer-material-design/v1", "inputs": 8, "queries": IDS,
              "dictionary_ids": ["D01", "D02"], "with_demos": False, "model": "Qwen3-8B",
              "task": info(TASK), "layer_numbers_zero_based": True,
              "carried_forward_mechanism_plan": {"focal_and_preceding_block_output_layer": 17,
                  "restorations_at_pre_answer": {"A": "layer26 attention after o_proj, before residual addition",
                                                "B": "layer28 MLP before residual addition", "AB": "both, sources from recipient-native run"},
                  "directions": "D01 into D02 and D02 into D01", "trajectory_layers": list(range(36)),
                  "no_layer_or_head_search": True, "fresh_same_run_sources_required": True},
              "analysis": {"raw_margin": "z(无)-z(有)", "reference_alignment": {"T01": 1, "T02": 1, "T03": -1, "T04": -1},
                  "groups": {"ordinary_replication": ["T01", "T02"], "inherited_label_group_reference": ["T03"],
                             "attack_outside_focal_laughter_boundary": ["T04"]},
                  "keep_all_four_cases_and_both_directions_regardless_of_predictions": True,
                  "small_denominator_ratios": "NA under inherited numerical gate; retain raw differences",
                  "case_count": 4, "conditions_or_directions_not_independent_samples": True},
              "execution_runtime_prepared_by_this_command": False, "GPU_started": False}
    write(OUT / "design.json", design)
    with (OUT / "model-inputs.jsonl").open("xb") as f:
        for req in rows:
            require(not {"reference", "hate", "gold", "human_decision", "review_status"} & set(req), "Reference leaked to scorer")
            f.write(canonical(req) + b"\n")
    text_file(OUT / "model-task.txt", system)
    prompt_document = ["# 已采用的八份完整模型输入", "", "T01–T04 × D01/D02；无示例。参考答案独立保存，不进入以下模型消息。", ""]
    for req in rows:
        text_file(OUT / "prompts" / (req["request_id"] + ".txt"), req["prompt_text"])
        prompt_document += ["## " + req["request_id"], "", "```text", req["prompt_text"], "```", ""]
    text_file(OUT / "ALL-PROMPTS.md", "\n".join(prompt_document) + "\n")
    ledger = ["request_id\tquery_id\tdictionary_id\ttokens\tfocal\tpre\tprompt_sha256"]
    for req in rows:
        ledger.append("\t".join(map(str, [req["request_id"], req["query_id"], req["dictionary_id"], req["prompt_tokens"],
                                         req["patch_position_sets"]["focal"][0], req["patch_position_sets"]["pre"][0], req["prompt_sha256"]])))
    text_file(OUT / "input-ledger.tsv", "\n".join(ledger) + "\n")
    sources = set(raw_sources) | {TASK, PARENT, Path(__file__).resolve(), ROOT / "src/diagnostics/case_attention_inputs_v1.py",
                                 WORK / "feedback-01.json"}
    sources.update(p for p in DRAFT.iterdir() if p.is_file())
    model_dir = ROOT / "models/base/Qwen3-8B"
    sources.update(p for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "chat_template.jinja")
                   if (p := model_dir / name).is_file())
    write(OUT / "source-ledger.json", {"sources": [info(p) for p in sorted(sources)], "model_weights_read": False})
    write(OUT / "cpu-input-checks.json", {"status": "pass", "source_texts_exact": 4,
        "adopted_prompt_hashes_match_draft_preview": 8, "single_token_answer_boundaries": 16,
        "paired_dictionary_geometry_checks": 4, "exact_focal_prefix_checks": 8,
        "preceding_query_position_checks": 8, "no_reference_fields_in_model_inputs": True,
        "all36_trajectory_layers_retained_in_plan": True, "torch_imported": False, "GPU_started": False,
        "scientific_runtime_qualification": "not performed by material adoption"})
    print(json.dumps({"status": "material_adoption_and_CPU_input_checks_pass", "directory": str(OUT),
                      "inputs": 8, "references": REFERENCES, "GPU_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
