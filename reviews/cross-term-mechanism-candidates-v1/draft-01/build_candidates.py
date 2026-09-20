"""Build a CPU-only candidate inventory from existing texts and sealed results.

This is a search/review artifact, not an execution package. It does not load a
tokenizer or model, allocate a GPU, adopt new materials, or change old selectors.
Existing scores guide development selection and are explicitly marked exposed.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TERMS = ["京巴", "垃圾", "公交车", "小丑", "花瓶", "寄生虫", "木头"]
SOURCE_FILES = set()


def read(rel):
    path = ROOT / rel
    SOURCE_FILES.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pin(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def write(name, obj):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def spans(text, term):
    return [{"start": i, "end": i + len(term), "text": term}
            for i in range(len(text)) if text.startswith(term, i)]


def strings(obj, prefix=""):
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from strings(value, prefix + "/" + str(key))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from strings(value, prefix + "/" + str(i))


def main():
    assert not (OUT / "candidate-pool.json").exists(), "Use a new version"
    data_ref = read("exps/causal_context/stage1_p0/refs/data_ref.json")
    partition_ref = read("exps/causal_context/stage1_p0/refs/train_partition_ref.json")
    partition_path = Path(partition_ref["target_path"]) / "partition.jsonl"
    SOURCE_FILES.add(partition_path)
    partition = {r["query_id"]: r["partition"]
                 for r in map(json.loads, partition_path.read_text().splitlines())}
    dev = read(str((Path(data_ref["target_path"]) / "dev.json").relative_to(ROOT)))
    dev_ids = {str(r["id"]) for r in dev}
    main_rows = read("data/full/std/train.json")
    cold_rows = read("data/cold/std/train.json")
    raw_main = {str(r["id"]): r for r in read("data/full/raw/train.json")}
    approved = read("docs/research/experiment-plans/cross-term-materials-v1/frozen-01/materials.json")
    lexicon = read("data/lexicon/annotated_lexicon_mechanism_frozen_v1.json")
    review_path = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/frozen-v1/material_reviews.jsonl"
    SOURCE_FILES.add(review_path)
    reviews = [json.loads(s) for s in review_path.read_text().splitlines() if s.strip()]
    eligible_main = [r for r in main_rows if str(r["id"]) in dev_ids
                     or partition.get(str(r["id"])) == "fit"]
    excluded = [str(r["id"]) for r in main_rows if str(r["id"]) not in dev_ids
                and partition.get(str(r["id"])) != "fit"]
    assert len(eligible_main) == 5808 and len(excluded) == 616
    pool = []
    for corpus, rel, rows in [("main", "data/full/std/train.json", main_rows),
                              ("cold", "data/cold/std/train.json", cold_rows)]:
        for index, row in enumerate(rows):
            rid = str(row["id"])
            if corpus == "main" and rid in excluded:
                continue
            matches = {t: spans(row["content"], t) for t in TERMS if t in row["content"]}
            if not matches:
                continue
            if corpus == "main":
                assert raw_main[rid]["content"] == row["content"]
            found_reviews = []
            for r in reviews:
                if (r.get("kind") not in ("query", "demo") or r.get("stale")
                        or r.get("source", {}).get("text") != row["content"]
                        or r.get("review", {}).get("status") != "confirmed"):
                    continue
                rv = r["review"]
                found_reviews.append({"record_id": r["record_id"], "kind": r["kind"],
                    "confirmed_at": rv.get("confirmed_at"), "version": rv.get("version"),
                    "hate": rv.get("values", {}).get("hate"),
                    "attack_severity": rv.get("final_annotation", {}).get("values", {}).get("attack_severity"),
                    "source": str(review_path.relative_to(ROOT)),
                    "unreviewed_supporting_fields": rv.get("final_annotation", {}).get("unreviewed_supporting_fields", [])})
            pool.append({"key": corpus + ":" + rid, "source_file": rel,
                "record_index_zero_based": index, "source_id": row["id"],
                "partition": "cold_train" if corpus == "cold" else "dev" if rid in dev_ids else "fit",
                "text": row["content"], "text_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                "literal_matches_not_semantic_decisions": matches,
                "source_labels": sorted({q["hateful"] for q in row["quadruples"]}),
                "source_quadruples": row["quadruples"], "existing_confirmed_reviews": found_reviews,
                "new_human_decision": None})
    index = {r["key"]: r for r in pool}

    natural_selections = [
        ("G01", "cold:cold_train_6794_1", "垃圾", "ordinary_no_attack", "无",
         "捡垃圾吃指实际废弃物；结尾肯定这些人与防疫人员的付出，没有作者攻击。"),
        ("G02", "main:7685", "垃圾", "pejorative_author_attack", "有",
         "用反问将他贬为垃圾；当前任务包含个人攻击。源non-hate不自动等于当前无。"),
        ("G03", "cold:cold_train_1338_1", "垃圾", "pejorative_rejected_or_criticized", "无",
         "将玩快手的都是垃圾与地域偏见类比并质疑；建议按反对概括贬损理解，引用范围和立场待审核。"),
        ("G04", "cold:cold_train_2914_1", "垃圾", "ordinary_with_other_attack", "有",
         "捡垃圾是职业/实物语境；作者另将安徽人概括为没文化、粗鲁，保留全文攻击。"),
        ("B01", "cold:cold_train_12278", "公交车", "ordinary_no_attack", "无",
         "赞叹运动员速度，以实际交通工具比较；出现族群词不等于攻击。"),
        ("B02", "main:6629", "公交车", "pejorative_author_attack", "有",
         "婚恋语境下以公交车污名化女性；继承已有demo人审有，其他缩写与贬损增加混杂。"),
        ("B04", "cold:cold_train_2265_1", "公交车", "ordinary_with_other_attack", "有",
         "公交车为实际车辆；变态、让人唾弃恶心另对人作攻击，待当前任务审核。"),
    ]
    shortlist = []
    for cid, key, term, role, proposed, why in natural_selections:
        row = index[key]
        prior_labels = {"有" if r["hate"] == "hate" else "无"
                        for r in row["existing_confirmed_reviews"] if r["hate"] in ("hate", "non-hate")}
        assert not prior_labels or prior_labels == {proposed}
        shortlist.append({"candidate_id": cid, "recommended_group": "primary" if cid.startswith("G") else "backup",
            "term": term, "key": key, "text": row["text"], "text_sha256": row["text_sha256"],
            "authorship": "source_corpus_exact_text", "source_file": row["source_file"],
            "source_index_zero_based": row["record_index_zero_based"], "partition": row["partition"],
            "role_proposal": role, "assistant_proposed_reference": proposed,
            "reference_status": "inherited_prior_human_review" if prior_labels else "awaiting_human_review",
            "existing_human_reference": proposed if prior_labels else None,
            "existing_human_reference_sources": row["existing_confirmed_reviews"],
            "source_labels": row["source_labels"], "rationale": why,
            "new_human_decision": None, "run_eligible": False})
    lookup = {q["short_id"]: q for q in approved["queries"]}
    for cid, old_id, role in [("J01", "J2", "ordinary_no_attack"),
                              ("J02", "J1", "pejorative_author_attack"),
                              ("J03", "J3", "pejorative_rejected_or_criticized"),
                              ("J04", "J4", "ordinary_with_other_attack"),
                              ("B03", "B3", "pejorative_rejected_or_criticized")]:
        q = lookup[old_id]; human = q["human_review"]
        assert human["text_adopted"] and q["authorship"] == "ai"
        shortlist.append({"candidate_id": cid, "recommended_group": "primary" if cid.startswith("J") else "backup",
            "term": "京巴" if cid.startswith("J") else "公交车", "key": q["query"]["material_id"],
            "text": q["query"]["raw_text"], "text_sha256": q["query"]["text_sha256"],
            "authorship": "previously_human_adopted_AI_construction", "source_file":
            "docs/research/experiment-plans/cross-term-materials-v1/frozen-01/materials.json",
            "role_proposal": role, "existing_human_reference": human["task_label"],
            "existing_human_severity": human["attack_severity"],
            "existing_human_reference_sources": [human["decision_ref"]],
            "assistant_proposed_reference": human["task_label"],
            "reference_status": "inherited_prior_human_review", "new_human_decision": None,
            "independent_confirmation": False, "run_eligible": False})
    shortlist.sort(key=lambda r: ({"J": 0, "G": 1, "B": 2}[r["candidate_id"][0]], r["candidate_id"]))

    # Exposure checks use text inventories, not new corpus prediction tables.
    historical = set()
    for base in (ROOT / "reviews", ROOT / "docs/research/experiment-plans",
                 ROOT / "exps/causal_context/general_model_evidence_applicability_v1/reviews"):
        for name in ("materials.json", "original-materials.json", "inherited-demo-materials.json"):
            historical.update(p for p in base.rglob(name) if OUT not in p.parents)
    exposure = {r["candidate_id"]: [] for r in shortlist}
    for path in sorted(historical):
        obj = read(str(path.relative_to(ROOT)))
        vals = list(strings(obj))
        for r in shortlist:
            hits = [jpath for jpath, value in vals if r["text"] in value]
            if hits:
                exposure[r["candidate_id"]].append({"file": str(path.relative_to(ROOT)), "json_paths": hits})
    for r in shortlist:
        r["prior_material_hits_within_scanned_scope"] = exposure[r["candidate_id"]]
        r["prior_prediction_exposure"] = "known_exposed_development" if r["authorship"].startswith("previously") else "unknown_beyond_recorded_prior_material_and_review_hits"

    # Only these already completed artificial-family scores are newly inspected.
    score_path = ROOT / "reviews/cross-model-applicability-execution-v1/comparison-01/legacy-scores.tsv"
    SOURCE_FILES.add(score_path)
    scores = list(csv.DictReader(score_path.open(encoding="utf-8"), delimiter="\t"))
    history = []
    for letter, term in [("J", "京巴"), ("G", "垃圾"), ("B", "公交车")]:
        for number in range(1, 5):
            sid = letter + str(number); q = lookup[sid]
            conditions = []
            for cond in ["absent-none", "L-none", "L-same_A", "L-same_B"]:
                row = next(r for r in scores if r["condition_id"] == "CTDJ-" + sid + "-" + cond)
                conditions.append({"condition_id": row["condition_id"], "condition": cond,
                    "reference": row["adopted_reference"], "models": {
                        model: {"m": row[model + "_m"], "prediction": row[model + "_raw_prediction"],
                                "error_bound": row[model + "_margin_error_bound"],
                                "origin": row[model + "_origin"], "score_id": row[model + "_physical_score_id"]}
                        for model in ("qwen3-8b", "qwen3-14b", "glm4-9b-chat")}})
            history.append({"term": term, "material_id": q["query"]["material_id"],
                "text": q["query"]["raw_text"], "human_reference": q["human_review"]["task_label"],
                "authorship": "human_adopted_AI_construction", "scores_are_prior_not_new": True,
                "conditions": conditions})

    ordinary = {"京巴": "京巴：一种小型家犬，可作为宠物饲养。",
                "垃圾": "垃圾：指被丢弃或失去使用价值、需要收集清理的废弃物。",
                "公交车": "公交车：按规定路线和站点运行、供公众乘坐的公共交通车辆。"}
    definitions = []
    for term in ("京巴", "垃圾", "公交车"):
        old = next(r for r in approved["lexicons"] if r["term"] == term)
        base = next(r for r in lexicon["terms"] if r["term"] == term)
        definitions.append({"term": term, "base_frozen_dictionary_entry": base,
            "proposed_pejorative": {"text": old["material"]["raw_text"], "status": "previously_human_adopted",
                                     "source": old["human_review"]["decision_ref"],
                                     "is_verbatim_base_dictionary_entry": False},
            "proposed_ordinary": {"text": ordinary[term], "authorship": "assistant_new_draft",
                                  "status": "awaiting_human_review", "new_human_decision": None}})
    counts = [{"term": t,
               "main_fit_dev_literal_rows": sum(t in r["content"] for r in eligible_main),
               "cold_train_literal_rows": sum(t in r["content"] for r in cold_rows),
               "main_original_train_literal_rows_before_partition_filter": sum(t in r["content"] for r in main_rows)}
              for t in TERMS]
    for rel in ["docs/research/experiment-plans/internal-reference-utilization-roadmap-20260918.md",
                "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt",
                "data/cold/README.md", "src/data/cold_adapter.py"]:
        SOURCE_FILES.add(ROOT / rel)
    evidence = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Local existing reviewed families plus main fit/dev and COLD train; no new external corpus download or model run",
        "term_list": TERMS, "counts": counts,
        "source_train_rows_initially_scanned": len(main_rows) + len(cold_rows),
        "eligible_corpus_rows": len(eligible_main) + len(cold_rows),
        "calibration_excluded_from_candidate_pool_count": len(excluded),
        "calibration_filter_timing": "Preliminary literal counts/content inspection used original main train; partition filter then excluded all616 calibration rows from candidate pool. Not a claim those bytes were never read.",
        "calibration_ids_with_screened_terms_before_filter": [str(r["id"]) for r in main_rows if str(r["id"]) in excluded and any(t in r["content"] for t in TERMS)],
        "test_and_COLD_val_test_used_for_new_candidate_search": False,
        "prior_material_files_scanned": len(historical), "review_snapshot_rows": len(reviews),
        "source_annotation_caveat": "Source hate/non-hate differs from current author-attack task. COLD raw CSV is absent; labels are local adapter outputs.",
        "COLD_raw_directory_present": (ROOT / "data/cold/raw").is_dir(),
        "selection_uses_known_historical_behavior": True,
        "new_predictions_or_scores_generated": False, "new_corpus_candidate_prediction_tables_read": False,
        "model_loaded": False, "GPU_started": False,
        "source_pins": [pin(p) for p in sorted(SOURCE_FILES)],
        "limitations": ["Development candidate selection, not independent confirmation.",
            "Literal substring hits do not establish a word occurrence or a valid sense.",
            "Prior material scan is bounded; absence of a hit does not prove no historical exposure.",
            "J and B reviewed controls are AI-authored; natural texts are identified separately.",
            "Proposed ordinary definitions and new labels remain unadopted."]}
    write("candidate-pool.json", {"status": "unadopted_candidate_search", "counts": counts, "records": pool})
    write("shortlist.json", {"status": "awaiting_candidate_selection_and_new_material_review",
        "recommended_primary_terms": ["京巴", "垃圾"], "backup_term": "公交车",
        "items": shortlist, "definitions": definitions, "GPU_execution_authorized_by_this_artifact": False})
    write("historical-behavior.json", {"source": pin(score_path), "scope": "48 existing conditions in12 reviewed artificial queries; not new results", "cases": history})
    write("search-evidence.json", evidence)
    audit = {"status": "pass", "pool_records": len(pool),
        "unique_exact_pool_texts": len({r["text"] for r in pool}),
        "shortlist": len(shortlist), "primary": sum(r["recommended_group"] == "primary" for r in shortlist),
        "natural_selected_texts_equal_sources": all(r["text"] == index[r["key"]]["text"] for r in shortlist if r["authorship"] == "source_corpus_exact_text"),
        "reviewed_constructed_texts_and_references_preserved": 5,
        "no_calibration_candidates": all(r["partition"] in ("fit", "dev", "cold_train") for r in pool),
        "no_new_human_decisions_or_adoptions": all(r["new_human_decision"] is None and not r["run_eligible"] for r in shortlist),
        "historical_conditions_copied": sum(len(r["conditions"]) for r in history),
        "no_new_model_run": True,
        "artifacts": [pin(OUT / n) for n in ["candidate-pool.json", "shortlist.json", "historical-behavior.json", "search-evidence.json"]],
        "builder": pin(Path(__file__).resolve())}
    assert audit["natural_selected_texts_equal_sources"] and audit["no_calibration_candidates"]
    write("audit.json", audit)
    print(json.dumps({"status": "pass", "pool": len(pool), "counts": counts,
        "shortlist": len(shortlist), "primary": audit["primary"],
        "exposure_hits": {r["candidate_id"]: len(r["prior_material_hits_within_scanned_scope"]) for r in shortlist},
        "GPU_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
