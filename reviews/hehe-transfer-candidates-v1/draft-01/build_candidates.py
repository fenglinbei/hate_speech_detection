"""Read-only corpus search plus a CPU-only, unadopted candidate-review snapshot.

No model weights, predictions, GPU controller, scientific selector or old source
are read/modified. This script writes only new files beside itself, exclusively.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from diagnostics.case_attention_inputs_v1 import tokenizer, build_input


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": digest(data)}


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def strings(obj, prefix=""):
    if isinstance(obj, str):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from strings(v, prefix + "/" + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from strings(v, prefix + "/" + str(i))


def main():
    assert not (OUT / "candidate-pool.json").exists(), "Use a new draft directory"
    files = [ROOT / f"data/full/std/{s}.json" for s in ("train", "test")]
    files += [ROOT / f"data/cold/std/{s}.json" for s in ("train", "val", "test")]
    pool, counts = [], []
    for path in files:
        corpus = "main" if "/full/" in str(path) else "cold"
        records = read(path)
        found = []
        for index, row in enumerate(records):
            if "嘿嘿" not in row["content"]:
                continue
            rid = str(row["id"])
            key = f"{corpus}:{rid}"
            item = {
                "key": key, "corpus": corpus, "source_id": row["id"],
                "source_file": str(path.relative_to(ROOT)), "record_index_zero_based": index,
                "text": row["content"], "text_sha256": digest(row["content"].encode()),
                "source_quadruples": row["quadruples"],
                "source_record": row,
                "source_label_is_current_task_human_approval": False,
                "already_used_Q01_Q02": key in ("main:3169", "main:3660"),
                "new_user_decision": None,
            }
            pool.append(item)
            found.append(key)
        counts.append({"file": str(path.relative_to(ROOT)), "records": len(records),
                       "literal_match_count": len(found), "keys": found})
    assert len(pool) == len({r["key"] for r in pool}) == 36
    bykey = {r["key"]: r for r in pool}
    assert sum(r["corpus"] == "main" for r in pool) == 7
    assert sum(r["corpus"] == "cold" for r in pool) == 29

    # Check source equivalence without inventing a second independent main pool.
    raw_paths = [ROOT / f"data/full/raw/{s}.json" for s in ("train", "test")]
    raw_records = {str(r["id"]): (p, i, r) for p in raw_paths for i, r in enumerate(read(p))}
    for row in pool:
        if row["corpus"] == "main":
            p, i, raw = raw_records[str(row["source_id"])]
            assert raw["content"] == row["text"]
            row["main_raw_provenance"] = {"file": str(p.relative_to(ROOT)), "index_zero_based": i,
                                           "original_sen_hate": raw.get("sen_hate")}

    review_path = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/reviews/analysis-freeze-20260912/frozen-v1/material_reviews.jsonl"
    references = [json.loads(s) for s in review_path.read_text().splitlines() if s.strip()]
    for row in pool:
        row["existing_confirmed_query_demo_records"] = [
            {"record_id": r["record_id"], "status": r["review"]["status"],
             "hate": r["review"].get("values", {}).get("hate")}
            for r in references if r.get("kind") in ("query", "demo")
            and r.get("source", {}).get("text") == row["text"]
            and r["review"].get("status") == "confirmed" and not r.get("stale")]

    # Scan only prior input/material inventories, not prediction/result tables.
    historical = set()
    review_root = ROOT / "exps/causal_context/general_model_evidence_applicability_v1/reviews"
    for base in (ROOT / "reviews", ROOT / "docs/research/experiment-plans", review_root):
        for pattern in ("materials.json", "original-materials.json", "inherited-demo-materials.json"):
            historical.update(p for p in base.rglob(pattern) if OUT not in p.parents)
    cards = ROOT / "exps/causal_context/general_model_ld_nolabel_paired_cases_v1/results/paired-cases-02/cases/card_data"
    historical.update(cards.glob("*.json"))
    exposure = {r["key"]: [] for r in pool}
    for path in sorted(historical):
        obj = read(path)
        if path.parent == cards:
            obj = {"query": obj["query"], "demonstrations": obj["demonstrations"]}
            # Projection fields are not used in selection; extract text only.
            obj = {"query": obj["query"].get("content"),
                   "demonstrations": [d.get("content") for d in obj["demonstrations"]]}
        values = list(strings(obj))
        for row in pool:
            hits = [jpath for jpath, value in values if row["text"] in value]
            if hits:
                exposure[row["key"]].append({"file": str(path.relative_to(ROOT)), "json_paths": hits})
    mirrors = [ROOT / f"data/v1/raw/{s}.json" for s in ("train", "test", "test2")]
    mirrors += [p for b in ("data/mav", "data/rag") for p in sorted((ROOT / b).glob("*.json"))]
    mirror_index = {}
    for path in mirrors:
        obj = read(path)
        if not isinstance(obj, list):
            continue
        for r in obj:
            if isinstance(r, dict) and isinstance(r.get("content"), str) and "嘿嘿" in r["content"]:
                mirror_index.setdefault(r["content"], []).append({"file": str(path.relative_to(ROOT)), "id": r.get("id")})
    for row in pool:
        row["prior_material_hits_within_scanned_scope"] = exposure[row["key"]]
        row["historical_corpus_copies"] = mirror_index.get(row["text"], [])
        row["human_seen_predictions_history"] = "unknown; not reconstructed"

    recommendations = [
        ("T01", "cold:cold_train_2640", "ordinary_replication", "无",
         "自述地域身份与自豪，嘿嘿是笑声；未见对人的贬损。来自COLD。"),
        ("T02", "cold:cold_train_3562", "ordinary_replication", "无",
         "对举例表达高兴，嘿嘿是笑声；未见对人的贬损。来自COLD。"),
        ("T03", "main:7050", "group_reference_candidate_pending_semantic_review", None,
         "语法上较像人群指代，最接近旧Q02；但原文不明说对象或贬损性质，不能把数据标签当成缺失上文。"),
        ("T04", "cold:cold_train_9078", "boundary_attack_outside_focal_laughter", "有",
         "嘿嘿可作讥笑；瞧不起某个种族、把中国拖入痛苦深渊是独立的群体贬损。检验普通义是否错误覆盖全文攻击证据。"),
        ("B01", "main:1301", "ordinary_backup_missing_preceding_query_control", "无",
         "普通聊天非常清楚，但嘿嘿在查询开头，没有查询正文内的前置token；不能原样套用已有位置对照。"),
        ("B02", "main:5871", "boundary_backup_with_other_lexicon_hit", "有",
         "小仙女整破防了带有轻蔑挖苦；嘿嘿是笑声而非人群指代。另命中固定词典中的仙女，且连续三个嘿与旧非重叠匹配规则存在焦点范围差别。"),
    ]
    material_path = ROOT / "reviews/hehe-sense-context-v1/prepared-01/materials.json"
    task_path = ROOT / "docs/research/experiment-plans/task-applicability-scoring-v1/frozen-01/model-task.txt"
    materials, task, tok = read(material_path), task_path.read_text(), tokenizer()
    shortlist = []
    for item_id, key, role, proposed, reason in recommendations:
        record = bykey[key]
        assert not record["already_used_Q01_Q02"]
        assert not record["existing_confirmed_query_demo_records"]
        assert not record["prior_material_hits_within_scanned_scope"]
        pair = []
        for definition in materials["dictionaries"][:2]:
            lexicon = deepcopy(materials["base_lexicon"])
            next(e for e in lexicon if e["term"] == "嘿嘿")["senses"][0]["definition"] = definition["definition"]
            case = {"query_id": str(record["source_id"]), "query_text": record["text"], "lexicon": lexicon, "demos": []}
            req = build_input(case, ("L", False, "definition"), tok, task)
            focal = req["roles"]["query_focal"]
            pre = list(range(min(focal) - len(focal), min(focal)))
            valid_pre = set(pre) <= set(req["roles"]["query_all"])
            pair.append({"dictionary_id": definition["dictionary_id"],
                         "prompt_sha256": req["prompt_sha256"], "tokens": req["prompt_tokens"],
                         "focal_positions": focal, "focal_text": [req["token_text"][i] for i in focal],
                         "pre_inside_query": valid_pre, "pre_positions": pre if valid_pre else None,
                         "pre_text": [req["token_text"][i] for i in pre] if valid_pre else None})
        assert pair[0]["tokens"] == pair[1]["tokens"]
        assert pair[0]["focal_positions"] == pair[1]["focal_positions"]
        assert pair[0]["pre_positions"] == pair[1]["pre_positions"]
        shortlist.append({"item_id": item_id, "key": key, "role": role, "text": record["text"],
                          "assistant_proposed_reference": proposed, "reason": reason,
                          "new_user_decision": None, "run_eligible": False,
                          "cpu_token_geometry_preview": pair})
        record["shortlist_item"] = item_id
    assert "torch" not in sys.modules

    sources = set(files + raw_paths + mirrors + list(historical))
    sources.update([review_path, material_path, task_path, ROOT / "data/cold/README.md",
                    ROOT / "src/data/cold_adapter.py", ROOT / "src/diagnostics/case_attention_inputs_v1.py"])
    sources.update((ROOT / "models/base/Qwen3-8B").glob("*token*json"))
    sources.add(ROOT / "models/base/Qwen3-8B/chat_template.jinja")
    sources = {p for p in sources if p.is_file()}
    originals = {r["text"] for r in pool if r["already_used_Q01_Q02"]}
    assert not any(r["text"] in originals for r in shortlist)
    write("candidate-pool.json", {"status": "unadopted_candidate_search", "match": "literal substring 嘿嘿",
                                  "counts": counts, "records": pool})
    write("shortlist.json", {"status": "awaiting_content_and_reference_review", "run_authorized": False,
                             "items": shortlist, "definitions": materials["dictionaries"][:2],
                             "with_demonstrations": False, "model_outputs_read_or_generated": False})
    write("search-evidence.json", {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Existing local main and COLD standardized corpora; not an internet search or exhaustive linguistic search.",
        "counts": counts, "unique_exact_texts": len({r["text"] for r in pool}),
        "review_snapshot_rows": len(references), "prior_material_files_scanned": len(historical),
        "scanned_material_sources": [str(p.relative_to(ROOT)) for p in sorted(historical)],
        "COLD_raw_directory_present": (ROOT / "data/cold/raw").is_dir(),
        "COLD_caveat": "Local normalized files available; raw CSV not present. Adapter maps coarse offensive/hate labels to project format; current-task labels require new human review.",
        "source_pins": [pin(p) for p in sorted(sources)],
        "old_prediction_inspection": False, "model_weights_loaded": False, "GPU_started": False,
        "scope_limits": ["No claim of never-seen model training data or investigator-blind historical exposure.",
                         "Matching labels are source annotations, not approved references for the current author-attack task.",
                         "Exact source text preserved; no rewrite of Q01/Q02. Semantic independence not statistically established.",
                         "Tokenizer-only geometry preview is not a frozen experiment protocol."],
    })
    write("audit.json", {"status": "pass", "searched_records": sum(c["records"] for c in counts),
                         "main_matches": 7, "cold_matches": 29, "old_main_cases_excluded": 2,
                         "shortlist_and_backups": len(shortlist), "exact_raw_main_text_matches": 7,
                         "all_shortlist_texts_equal_sources": all(bykey[r["key"]]["text"] == r["text"] for r in shortlist),
                         "all_shortlist_new_human_decisions_null": all(r["new_user_decision"] is None for r in shortlist),
                         "D01_D02_geometry_aligned_pairs": len(shortlist), "torch_imported": False,
                         "GPU_started": False, "new_experiment_frozen": False,
                         "artifacts": [pin(OUT / n) for n in ("candidate-pool.json", "shortlist.json", "search-evidence.json")],
                         "builder": pin(Path(__file__).resolve())})
    print(json.dumps({"status": "pass", "directory": str(OUT), "pool": len(pool),
                      "shortlist": len(shortlist), "GPU_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
