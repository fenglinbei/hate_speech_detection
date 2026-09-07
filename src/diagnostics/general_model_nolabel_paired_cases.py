"""CPU-only, posthoc paired analysis and deterministic mechanism-case preparation.

Only explicitly hashed inputs are read. No model, tokenizer, old plan loader, or
experiment execution module is imported. This module never changes a source run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import unicodedata

import numpy as np

from diagnostics.general_model_numeric_analysis import (
    GROUP_LABELS, SCORE_MODES, _gold_ordinal, _margins, _validated_candidates,
    logsumexp,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "general-model-nolabel-paired-cases/v1"
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "CLnewNoCat", "CLDnewNoCat")
TASKS = ("hate", "group")
CORE = ("C0", "CLnewNoCat", "CD", "CLDnewNoCat")
BUCKETS = ("H_rescue", "H_residual", "H_removal_harm", "G_category_support",
           "H_joint_only", "G_joint_only", "Stable_correct", "Stable_wrong")
PAIRS = {
    "remove_with_D": ("CLDnew", "CLDnewNoCat"),
    "remove_without_D": ("CLnew", "CLnewNoCat"),
    "S_given_D": ("CD", "CLDnewNoCat"),
    "S_vs_0": ("C0", "CLnewNoCat"),
    "D_vs_0": ("C0", "CD"),
    "D_given_S": ("CLnewNoCat", "CLDnewNoCat"),
}
EFFECTS = {
    "E_remove_with_D": {"CLDnewNoCat": 1, "CLDnew": -1},
    "E_remove_without_D": {"CLnewNoCat": 1, "CLnew": -1},
    "E_S_given_D": {"CLDnewNoCat": 1, "CD": -1},
    "I_S_D": {"CLDnewNoCat": 1, "CLnewNoCat": -1, "CD": -1, "C0": 1},
}
CI_CONTRASTS = {
    "NoCat_without_D": {"CLnewNoCat": 1, "CLnew": -1},
    "NoCat_with_D": {"CLDnewNoCat": 1, "CLDnew": -1},
    "E_S_given_D_F1": {"CLDnewNoCat": 1, "CD": -1},
    "J_remove_by_D_F1": {"CLDnewNoCat": 1, "CLDnew": -1,
                          "CLnewNoCat": -1, "CLnew": 1},
}
STRATA = ("all", "Lq_hit", "Lq_no_hit", "gold_hate", "gold_non_hate",
          "gold_group_0", "gold_group_1", "gold_group_2plus",
          "Lq_hit__gold_hate", "Lq_hit__gold_non_hate",
          "Lq_no_hit__gold_hate", "Lq_no_hit__gold_non_hate")
CONTEXT_INPUTS = {"contexts", "inventory", "parent_ref", "parent_plan",
                  "parent_inventory", "lexicon", "fit_catalog"}


class AnalysisError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def require(test, message, status="baseline_mismatch"):
    if not test:
        raise AnalysisError(status, message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def json_lines(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            # Explicit JSON null distinguishes missing values from empty strings.
            writer.writerow({k: canonical(v) if v is None or isinstance(v, (dict, list))
                             else v for k, v in row.items()})


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            "numeric mismatch: " + label)


def unique_index(rows, field, label):
    result = {}
    for row in rows:
        key = row[field]
        require(isinstance(key, str) and key not in result,
                "duplicate/non-string ID in " + label)
        result[key] = row
    return result


def ranking(candidates, epsilon, mode="answer_sum"):
    ordered = sorted(candidates, key=lambda c: (-c["scores"][mode], c["ordinal"]))
    best = ordered[0]
    gap = best["scores"][mode] - ordered[1]["scores"][mode]
    return {"labels": best["labels"], "ordinal": best["ordinal"],
            "top_score_gap": gap,
            "tied_top_count": sum(c["scores"][mode] == best["scores"][mode]
                                  for c in candidates),
            "within_two_epsilon": gap <= 2 * epsilon}


def is_correct(prediction, gold, task):
    return prediction == [gold] if task == "hate" else set(prediction) == set(gold)


def family_key(text):
    return digest(unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")))


def split_family(key, seed=20260907):
    h = digest(f"paired-cases-v1|{seed}|split|{key}")
    return "reserve" if int(h[:16], 16) % 3 == 0 else "discovery"


def memberships(profile, epsilon):
    def ok(c, task):
        return profile["conditions"][c][task]["correct"]

    h, g = lambda c: ok(c, "hate"), lambda c: ok(c, "group")
    rules = [
        ("H_rescue", "hate", ("CD", "CLDnew", "CLDnewNoCat"),
         h("CD") and not h("CLDnew") and h("CLDnewNoCat")),
        ("H_residual", "hate", ("CD", "CLDnew", "CLDnewNoCat"),
         h("CD") and not h("CLDnew") and not h("CLDnewNoCat")),
        ("H_removal_harm", "hate", ("CLDnew", "CLDnewNoCat"),
         h("CLDnew") and not h("CLDnewNoCat")),
        ("G_category_support", "group", ("CLnew", "CLnewNoCat"),
         g("CLnew") and not g("CLnewNoCat")),
        ("G_category_harm", "group", ("CLnew", "CLnewNoCat"),
         not g("CLnew") and g("CLnewNoCat")),
    ]
    for task, prefix in (("hate", "H"), ("group", "G")):
        rules += [(prefix + "_joint_only", task, CORE, profile["core_mask"][task] == "0001"),
                  ("Stable_correct", task, CONDITIONS, all(ok(c, task) for c in CONDITIONS)),
                  ("Stable_wrong", task, CONDITIONS, not any(ok(c, task) for c in CONDITIONS))]
    result = []
    for bucket, task, conditions, matches in rules:
        if not matches:
            continue
        boundary = [c for c in conditions
                    if profile["conditions"][c][task]["prediction"]["tied_top_count"] != 1
                    or profile["conditions"][c][task]["prediction"]["top_score_gap"] <= 2 * epsilon]
        result.append({"bucket": bucket, "focus_task": task, "required_conditions": list(conditions),
                       "numeric_eligible": not boundary, "boundary_conditions": boundary,
                       "exclusion_reason": "numeric_boundary" if boundary else None,
                       "score_mode_sensitive": any(profile["conditions"][c][task]
                                                   ["score_mode_sensitive"] for c in conditions)})
    return result


def counts_tensor(profiles):
    result = np.zeros((len(profiles), len(CONDITIONS), 7, 3), dtype=np.int64)
    for i, p in enumerate(profiles):
        gold = [p["gold"]["hate"] == label for label in ("hate", "non-hate")]
        gold += [label in p["gold"]["group"] for label in GROUP_LABELS]
        for j, c in enumerate(CONDITIONS):
            pred = [label in p["conditions"][c]["hate"]["prediction"]["labels"]
                    for label in ("hate", "non-hate")]
            pred += [label in p["conditions"][c]["group"]["prediction"]["labels"]
                     for label in GROUP_LABELS]
            for k, (truth, guess) in enumerate(zip(gold, pred)):
                result[i, j, k] = [truth and guess, not truth and guess, truth and not guess]
    return result


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator != 0)


def f1_from_counts(counts):
    tp, fp, fn = (counts[..., k] for k in range(3))
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {"hate": f1[..., :2].mean(axis=-1),
            "group": divide(2 * tp[..., 2:].sum(axis=-1),
                            (2 * tp + fp + fn)[..., 2:].sum(axis=-1))}


def bootstrap(stats, repetitions=10000, seed=42, batch_size=128):
    require(stats.ndim == 4 and stats.shape[1:] == (6, 7, 3) and len(stats) > 0
            and np.isfinite(stats).all() and (stats >= 0).all()
            and repetitions > 0 and batch_size > 0, "invalid bootstrap counts")
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = stats.reshape(len(stats), -1).astype(np.float64)
    coefficients = np.array([[weights.get(c, 0) for c in CONDITIONS]
                             for weights in CI_CONTRASTS.values()], dtype=np.float64)
    draws = {task: np.empty((repetitions, len(CI_CONTRASTS))) for task in TASKS}
    draw_hash = hashlib.sha256()
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        indices = rng.integers(0, len(stats), size=(size, len(stats)))
        draw_hash.update(indices.astype("<i8", copy=False).tobytes())
        weights = np.stack([np.bincount(row, minlength=len(stats)) for row in indices]).astype(float)
        metrics = f1_from_counts((weights @ flat).reshape(size, 6, 7, 3))
        for task in TASKS:
            draws[task][start:start + size] = metrics[task] @ coefficients.T
    points = f1_from_counts(stats.sum(axis=0))
    rows = []
    for j, contrast in enumerate(CI_CONTRASTS):
        for task in TASKS:
            lo, hi = np.quantile(draws[task][:, j], [.025, .975], method="linear")
            rows.append({"schema_version": SCHEMA, "contrast": contrast, "task": task,
                         "metric": "macro_f1" if task == "hate" else "micro_f1",
                         "n": len(stats), "difference": float(points[task] @ coefficients[j]),
                         "ci95_low": float(lo), "ci95_high": float(hi),
                         "analysis_kind": "original_endpoint_check" if j < 2 else "posthoc-exploratory",
                         "interval": "pointwise-descriptive-unadjusted", "repetitions": repetitions,
                         "seed": seed, "rng": "PCG64", "quantile_method": "linear"})
    return rows, draw_hash.hexdigest()


def select_cases(profiles, seed=20260907, quotas=None):
    """Task alternation for controls; stratum round-robin with hash-ranked queues."""
    quotas = quotas or {"discovery": 4, "reserve": 2}
    selected, events, summary = [], [], []
    used_queries, used_families = set(), set()
    for bucket in BUCKETS:
        for split in ("discovery", "reserve"):
            cells = defaultdict(list)
            raw_ids, eligible_ids = set(), set()
            for p in profiles:
                if p["split"] != split:
                    continue
                for m in p["memberships"]:
                    if m["bucket"] != bucket:
                        continue
                    task = m["focus_task"]
                    raw_ids.add(p["query_id"])
                    event = {"bucket": bucket, "split": split, "query_id": p["query_id"],
                             "family_key": p["family_key"], "focus_task": task}
                    if not m["numeric_eligible"]:
                        events.append(event | {"action": "excluded", "reason": "numeric_boundary"})
                        continue
                    eligible_ids.add(p["query_id"])
                    gold_layer = p["gold"]["hate"] if task == "hate" else str(min(2, len(p["gold"]["group"])))
                    cell = canonical({"task": task, "gold_layer": gold_layer, "lq_hit": p["lex_hit"]})
                    cells[task, cell].append((p, m, event))
            queues = {}
            for (task, cell), values in cells.items():
                queues[task, cell] = deque(sorted(values, key=lambda x:
                    (digest(f"{seed}|{bucket}|{x[0]['query_id']}"), x[0]["query_id"])))
            task_cells = {task: deque(sorted([cell for t, cell in queues if t == task],
                          key=lambda cell: (digest(f"{seed}|{bucket}|{cell}"), cell))) for task in TASKS}
            bucket_selected = []

            def take(task):
                while task_cells[task]:
                    cell = task_cells[task].popleft()
                    q = queues[task, cell]
                    p, m, event = q.popleft()
                    if q:
                        task_cells[task].append(cell)
                    reason = ("query_already_selected" if p["query_id"] in used_queries else
                              "family_already_selected" if p["family_key"] in used_families else None)
                    if reason:
                        events.append(event | {"action": "skipped", "reason": reason})
                        continue
                    result = {"schema_version": SCHEMA, "query_id": p["query_id"],
                              "family_key": p["family_key"], "split": split, "primary_bucket": bucket,
                              "focus_task": task, "lex_hit": p["lex_hit"],
                              "gold": {t: p["gold"][t] for t in TASKS},
                              "candidate_labels": p["candidate_labels"], "membership": m,
                              "context_status": p["context_status"], "selection_ordinal": len(selected),
                              "reviewer_content_opened": False}
                    selected.append(result)
                    bucket_selected.append(result)
                    used_queries.add(p["query_id"])
                    used_families.add(p["family_key"])
                    events.append(event | {"action": "selected", "reason": None, "subcell_key": cell})
                    return True
                return False

            while len(bucket_selected) < quotas[split]:
                progress = False
                # Always start with hate; the single-task buckets naturally use only one queue.
                for task in TASKS:
                    if len(bucket_selected) < quotas[split]:
                        progress = take(task) or progress
                if not progress:
                    break
            for q in queues.values():
                for _, _, event in q:
                    events.append(event | {"action": "not_selected", "reason": "quota_reached"})
            summary.append({"bucket": bucket, "split": split, "raw_queries": len(raw_ids),
                            "numeric_eligible_queries": len(eligible_ids), "target": quotas[split],
                            "selected": len(bucket_selected), "shortfall": quotas[split] - len(bucket_selected),
                            "selected_by_task": dict(Counter(r["focus_task"] for r in bucket_selected)),
                            "shortfall_reason": "eligible_families_exhausted_after_prior_buckets"
                            if len(bucket_selected) < quotas[split] else None})
    require(len(used_queries) == len(selected) == len(used_families), "selection is not unique")
    return {"selected": selected, "events": events, "quotas": summary}


def verify_inputs(config):
    paths, receipt, missing = {}, {}, []
    for name, spec in config["inputs"].items():
        path = Path(spec["path"])
        require(path.is_absolute(), "input paths must be explicit and absolute", "identity_mismatch")
        require(not any(s in path.name.lower() for s in (".test.", "test.json", "test.csv")),
                "test content is outside this analysis", "identity_mismatch")
        if not path.is_file():
            if name in CONTEXT_INPUTS:
                missing.append(name)
                receipt[name] = {"path": str(path), "sha256": None, "reason": "input_missing"}
                continue
            raise AnalysisError("input_missing", f"missing required input: {name}: {path}")
        actual = file_sha(path)
        require(actual == spec["sha256"], "input hash mismatch: " + name, "identity_mismatch")
        paths[name] = path
        receipt[name] = {"path": str(path), "sha256": actual, "reason": None}
    return paths, receipt, missing


def check_binding(config, paths):
    ref, plan = read_json(paths["plan_ref"]), read_json(paths["plan"])
    run, raw_manifest = read_json(paths["run_manifest"]), read_json(paths["raw_manifest"])
    am, report = read_json(paths["analysis_manifest"]), read_json(paths["report_manifest"])
    require(run["status"] == "complete" and report["passed"], "source run/report incomplete", "identity_mismatch")
    for name, item in (("ref", ref), ("plan", plan), ("run", run), ("analysis_manifest", am), ("report", report)):
        require(item["plan_id"] == config["plan_id"], "plan ID mismatch: " + name, "identity_mismatch")
    links = [(ref["plan_sha256"], "plan"),
             (run["raw_manifest_sha256"], "raw_manifest"),
             (run["analysis_manifest_sha256"], "analysis_manifest"),
             (raw_manifest["scores_sha256"], "raw"),
             (am["analysis_sha256"], "analysis"),
             (am["raw_manifest_sha256"], "raw_manifest"),
             (report["raw_scores_sha256"], "raw"), (report["analysis_sha256"], "analysis"),
             (report["files"]["classification.csv"], "classification"),
             (report["files"]["primary_differences.csv"], "original_ci"),
             (plan["package_manifest_sha256"], "package_manifest")]
    package = read_json(paths["package_manifest"])
    package_files = {row["path"]: row["sha256"] for row in package["files"]}
    links.append((package_files["queries.dev.jsonl"], "queries"))
    for name, filename in (("contexts", "contexts.dev.jsonl"), ("inventory", "resource_inventory.json")):
        if name in paths:
            links.append((plan["input_files"][filename], name))
    if "fit_catalog" in paths:
        links.append((package_files["fit_catalog.jsonl"], "fit_catalog"))
    if "lexicon" in paths:
        links.append((plan["config"]["resource"]["lexicon_sha256"], "lexicon"))
    if "parent_ref" in paths:
        links.append((plan["parent_ref_sha256"], "parent_ref"))
        if "parent_plan" in paths:
            pr, pp = read_json(paths["parent_ref"]), read_json(paths["parent_plan"])
            require(pr["plan_id"] == pp["plan_id"] == plan["parent_plan_id"],
                    "parent plan ID mismatch", "identity_mismatch")
            links.append((pr["plan_sha256"], "parent_plan"))
            if "parent_inventory" in paths:
                links.append((pp["input_files"]["resource_inventory.json"], "parent_inventory"))
    for expected, name in links:
        require(expected == config["inputs"][name]["sha256"],
                "manifest hash binding mismatch: " + name, "identity_mismatch")
    require(tuple(plan["config"]["conditions"]) == CONDITIONS,
            "condition order differs from frozen plan", "identity_mismatch")
    return plan


def load_context_metadata(paths, qids, plan, missing):
    reasons = list(missing)
    contexts = {}
    if "contexts" in paths:
        for row in json_lines(paths["contexts"]):
            key = (row["query_id"], row["task"], row["condition"])
            require(key not in contexts, "duplicate context block")
            contexts[key] = row
        require(set(contexts) == {(q, t, c) for q in qids for t in TASKS for c in CONDITIONS},
                "frozen context matrix mismatch")
    inv = unique_index(read_json(paths["inventory"])["per_query"], "query_id", "inventory") if "inventory" in paths else {}
    parent = unique_index(read_json(paths["parent_inventory"])["per_query"], "query_id", "parent inventory") if "parent_inventory" in paths else {}
    fit = unique_index(json_lines(paths["fit_catalog"]), "id", "fit catalog") if "fit_catalog" in paths else {}
    for name, data in (("inventory", inv), ("parent_inventory", parent)):
        if name in paths:
            require(set(data) == set(qids), name + " frame mismatch")
    descriptions = {(r["query_id"], r["task"], r["condition"]): r for r in plan["blocks"]}
    resources, lengths = {}, {}
    for q in qids:
        local_reasons = list(reasons)
        meta = {k: parent.get(q, {}).get(k) for k in (
            "lq_ids", "ld_ids", "ld_only_ids", "lq_only_ids", "intersection_ids", "union_ids",
            "lq_count", "ld_count", "demo_ids", "demo_match_ids")}
        if q in inv:
            if q in parent:
                require(inv[q]["union_ids"] == parent[q]["union_ids"], "union membership differs")
            meta["union_ids"] = inv[q]["union_ids"]
        meta["ld_only_count"] = len(meta["ld_only_ids"]) if meta["ld_only_ids"] is not None else None
        meta["union_count"] = len(meta["union_ids"]) if meta["union_ids"] is not None else None
        if meta["demo_ids"] is not None and all(d in fit for d in meta["demo_ids"]):
            demos = [fit[d] for d in meta["demo_ids"]]
            meta["demo_hate_counts"] = {label: sum(d["projection"]["hate"] == label for d in demos)
                                        for label in ("hate", "non-hate")}
            meta["demo_group_counts"] = {label: sum(label in d["projection"]["group"] for d in demos)
                                         for label in GROUP_LABELS}
        else:
            meta["demo_hate_counts"] = meta["demo_group_counts"] = None
            local_reasons.append("demo_composition_unavailable")
        for t in TASKS:
            for c in CONDITIONS:
                key = (q, t, c)
                row, descriptor = contexts.get(key), descriptions[key]
                if row:
                    for field in ("context_sha256", "prompt_sha256", "prompt_token_ids_sha256", "prompt_tokens"):
                        require(row[field] == descriptor[field], "context descriptor mismatch: " + field)
                    require(hashlib.sha256(row["prompt_text"].encode()).hexdigest() == row["prompt_sha256"],
                            "context prompt hash mismatch")
                    if q in inv:
                        require(row["prompt_tokens"] == inv[q]["condition_tokens"][f"{t}:{c}"],
                                "task/condition prompt length mismatch")
                    if meta["demo_ids"] is not None:
                        require(row["trace"]["source_demo_ids"] == meta["demo_ids"], "demo order mismatch")
                    if meta["union_ids"] is not None and c in ("CLnew", "CLDnew", "CLnewNoCat", "CLDnewNoCat"):
                        require(row["trace"]["injected_lexicon_ids"] == meta["union_ids"], "injected dictionary mismatch")
                lengths[key] = {"prompt_tokens": descriptor["prompt_tokens"],
                                "dictionary_tokens": row["trace"]["injected_blocks"]["lexicon_tokens"] if row else None,
                                "demo_tokens": row["trace"]["injected_blocks"]["demo_tokens"] if row else None,
                                "context_sha256": descriptor["context_sha256"],
                                "prompt_sha256": descriptor["prompt_sha256"],
                                "missing_reason": None if row else "context_incomplete"}
        meta["missing_reasons"] = sorted(set(local_reasons))
        resources[q] = meta
    return resources, lengths


def validate_and_profile(config, paths, missing):
    plan = check_binding(config, paths)
    qids = [row["query_id"] for row in plan["frame"]]
    frame = unique_index(plan["frame"], "query_id", "plan frame")
    require(len(qids) == config["query_count"], "query count mismatch")
    queries = unique_index(json_lines(paths["queries"]), "id", "dev queries")
    require(set(queries) == set(qids), "query/Gold frame differs")
    analysis = read_json(paths["analysis"])
    require([p["query_id"] for p in analysis["per_query"]] == qids,
            "analysis query order differs from plan.frame")
    old = unique_index(analysis["per_query"], "query_id", "per_query")
    require(tuple(analysis["conditions"]) == CONDITIONS, "analysis conditions differ")
    close(analysis["epsilon"], config["epsilon"], "epsilon")
    for q in qids:
        require(old[q]["gold"] == queries[q]["projection"], "Gold projection differs: " + q)
        require(type(old[q]["lex_hit"]) is bool and old[q]["lex_hit"] == frame[q]["lex_hit"],
                "query hit stratum differs: " + q)
        require(set(old[q]["conditions"]) == set(CONDITIONS), "missing/extra analysis condition")
        for c in CONDITIONS:
            require(set(old[q]["conditions"][c]) == set(TASKS), "missing/extra analysis task")
    expected = {(q, t, c) for q in qids for t in TASKS for c in CONDITIONS}
    blocks = {}
    descriptors = unique_index(plan["blocks"], "record_id", "plan blocks")
    require(set(descriptors) == {f"{q}:{t}:{c}" for q, t, c in expected}, "plan block matrix differs")
    candidate_count = 0
    for block in json_lines(paths["raw"]):
        q, t, c = key = (block["query_id"], block["task"], block["condition"])
        require(key in expected and key not in blocks, "duplicate/missing/unexpected raw block")
        require(block["plan_id"] == config["plan_id"], "raw plan identity differs", "identity_mismatch")
        require(block["record_id"] == f"{q}:{t}:{c}", "raw record ID differs")
        for field in ("context_sha256", "prompt_sha256"):
            require(block[field] == descriptors[block["record_id"]][field], "raw context binding differs")
        candidates = _validated_candidates(t, block["candidates"])
        for candidate, catalog in zip(candidates, plan["catalog"][t]):
            require(all(candidate.get(k) == v for k, v in catalog.items()), "raw candidate catalog differs")
        candidate_count += len(candidates)
        pred = ranking(candidates, config["epsilon"])
        saved = old[q]["conditions"][c][t]
        for field in ("labels", "ordinal", "tied_top_count", "within_two_epsilon"):
            require(pred[field] == saved["prediction"][field], "raw/analysis prediction mismatch: " + field)
        close(pred["top_score_gap"], saved["prediction"]["top_score_gap"], "top gap")
        gold_index = _gold_ordinal(t, queries[q]["projection"][t])
        scores = [candidate["scores"]["answer_sum"] for candidate in candidates]
        normalizer = logsumexp(scores)
        used = {"answer_sum/gold/best_nongold_margin": scores[gold_index] - max(
                    value for i, value in enumerate(scores) if i != gold_index),
                "answer_sum/gold/mass": math.exp(scores[gold_index] - normalizer),
                "answer_sum/gold/nll": normalizer - scores[gold_index]}
        used.update({"answer_sum/margin/" + label: value
                     for label, value in _margins(t, candidates, "answer_sum").items()})
        for name, value in used.items():
            close(value, saved["readouts"][name], "readout " + name)
        modes = {mode: ranking(candidates, config["epsilon"], mode)["labels"] for mode in SCORE_MODES}
        gold = queries[q]["projection"][t]
        blocks[key] = {"prediction": pred, "correct": is_correct(pred["labels"], gold, t),
                       "readouts": {name: saved["readouts"][name] for name in used},
                       "score_mode_predictions": modes,
                       "score_mode_sensitive": any(labels != pred["labels"] for labels in modes.values()),
                       "pred_group_size": len(pred["labels"]) if t == "group" else None,
                       "group_error_count": len(set(pred["labels"]) ^ set(gold)) if t == "group" else None}
    require(set(blocks) == expected and len(blocks) == config["block_count"], "incomplete paired raw matrix")
    require(candidate_count == config["candidate_count"], "candidate count mismatch")
    resources, lengths = load_context_metadata(paths, qids, plan, missing)
    profiles = []
    for q in qids:
        p = {"schema_version": SCHEMA, "query_id": q, "lex_hit": old[q]["lex_hit"],
             "gold": old[q]["gold"], "family_key": family_key(queries[q]["content"]),
             "conditions": {c: {t: blocks[q, t, c] | {"context": lengths[q, t, c]}
                                  for t in TASKS} for c in CONDITIONS}, "resources": resources[q]}
        p["context_status"] = "context_incomplete" if resources[q]["missing_reasons"] else "complete"
        p["split"] = split_family(p["family_key"], config["selection"]["seed"])
        p["core_mask"] = {t: "".join(str(int(blocks[q, t, c]["correct"])) for c in CORE) for t in TASKS}
        p["six_condition_mask"] = {t: "".join(str(int(blocks[q, t, c]["correct"])) for c in CONDITIONS) for t in TASKS}
        p["memberships"] = memberships(p, config["epsilon"])
        p["candidate_labels"] = [m["bucket"] + ":" + m["focus_task"] for m in p["memberships"]]
        profiles.append(p)
    return profiles, {"passed": True, "query_count": len(qids), "block_count": len(blocks),
                      "candidate_count": candidate_count, "frame_order_preserved": True,
                      "predictions_gold_and_used_readouts_aligned": True,
                      "candidate_identity_arithmetic_and_finiteness_checked": True,
                      "context_status": "context_incomplete" if any(p["context_status"] != "complete" for p in profiles) else "complete",
                      "missing_context_inputs": missing,
                      "original_full_gpu_and_prompt_audits_repeated": False}


def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "q25": None, "q75": None,
                "positive_fraction": None, "negative_fraction": None, "zero_fraction": None,
                "missing_reason": "empty_or_unavailable"}
    xs = np.asarray(values, dtype=float)
    require(np.isfinite(xs).all(), "nonfinite descriptive readout")
    q25, median, q75 = np.quantile(xs, [.25, .5, .75], method="linear")
    return {"n": len(xs), "mean": float(xs.mean()), "median": float(median),
            "q25": float(q25), "q75": float(q75),
            "positive_fraction": float((xs > 0).mean()), "negative_fraction": float((xs < 0).mean()),
            "zero_fraction": float((xs == 0).mean()), "missing_reason": None}


def stratum_names(profile):
    hit = "Lq_hit" if profile["lex_hit"] else "Lq_no_hit"
    hate = "gold_hate" if profile["gold"]["hate"] == "hate" else "gold_non_hate"
    size = len(profile["gold"]["group"])
    return ("all", hit, hate, "gold_group_" + (str(size) if size < 2 else "2plus"), hit + "__" + hate)


def composition(profiles):
    return {"gold_hate_n": sum(p["gold"]["hate"] == "hate" for p in profiles),
            "gold_non_hate_n": sum(p["gold"]["hate"] == "non-hate" for p in profiles),
            "Lq_hit_n": sum(p["lex_hit"] for p in profiles),
            **{"gold_" + label + "_n": sum(label in p["gold"]["group"] for p in profiles)
               for label in GROUP_LABELS},
            **{"gold_group_" + size + "_n": sum(min(len(p["gold"]["group"]), 2) == i for p in profiles)
               for i, size in enumerate(("0", "1", "2plus"))}}


def point_metrics(profiles, stats):
    metric_names = ("hate_accuracy", "hate_precision", "hate_recall", "hate_f1", "hate_macro_f1",
                    "group_micro_f1", "group_macro_f1", "group_exact_match", "group_sample_f1",
                    "joint_exact_match", "joint_correct")
    if not profiles:
        return [{"schema_version": SCHEMA, "condition": c, "n": 0,
                 **{name: None for name in metric_names}, "missing_reason": "empty_stratum"}
                for c in CONDITIONS]
    counts = stats.sum(axis=0)
    primary = f1_from_counts(counts)
    rows = []
    for j, c in enumerate(CONDITIONS):
        tp, fp, fn = (counts[j, :, k] for k in range(3))
        per_label = divide(2 * tp, 2 * tp + fp + fn)
        hc = [p["conditions"][c]["hate"]["correct"] for p in profiles]
        gc = [p["conditions"][c]["group"]["correct"] for p in profiles]
        joint = sum(h and g for h, g in zip(hc, gc))
        sample_f1 = []
        for p in profiles:
            pred, gold = set(p["conditions"][c]["group"]["prediction"]["labels"]), set(p["gold"]["group"])
            sample_f1.append(2 * len(pred & gold) / (len(pred) + len(gold)) if pred or gold else 1.)
        rows.append({"schema_version": SCHEMA, "condition": c, "n": len(profiles),
                     "hate_accuracy": sum(hc) / len(profiles),
                     "hate_precision": float(tp[0] / (tp[0] + fp[0])) if tp[0] + fp[0] else 0.,
                     "hate_recall": float(tp[0] / (tp[0] + fn[0])) if tp[0] + fn[0] else 0.,
                     "hate_f1": float(per_label[0]), "hate_macro_f1": float(primary["hate"][j]),
                     "group_micro_f1": float(primary["group"][j]), "group_macro_f1": float(per_label[2:].mean()),
                     "group_exact_match": sum(gc) / len(profiles), "group_sample_f1": float(np.mean(sample_f1)),
                     "joint_exact_match": joint / len(profiles), "joint_correct": joint, "missing_reason": None})
    return rows


def check_baseline_points(profiles, paths):
    points = point_metrics(profiles, counts_tensor(profiles))
    with paths["classification"].open(encoding="utf-8", newline="") as f:
        reference = list(csv.DictReader(f))
    require([r["condition"] for r in reference] == list(CONDITIONS), "classification row order differs")
    for actual, saved in zip(points, reference):
        for field, value in saved.items():
            if field == "condition":
                continue
            if field in ("n", "joint_correct"):
                require(actual[field] == int(value), "baseline integer mismatch: " + field)
            else:
                require(abs(actual[field] - float(value)) <= 1e-12,
                        "baseline classification mismatch: " + field)
    return {"points_passed": True, "conditions": len(points), "points": points,
            "original_ci_status": "pending_shared_bootstrap_in_analyze"}


def condition_rows(profiles):
    rows = []
    for p in profiles:
        for c in CONDITIONS:
            for task in TASKS:
                d = p["conditions"][c][task]
                readouts = d["readouts"]
                rows.append({"schema_version": SCHEMA, "query_id": p["query_id"], "condition": c,
                             "task": task, "lex_hit": p["lex_hit"], "gold": p["gold"][task],
                             "pred_labels": d["prediction"]["labels"], "correct": d["correct"],
                             **{k: v for k, v in d["prediction"].items() if k != "labels"},
                             "gold_margin": readouts["answer_sum/gold/best_nongold_margin"],
                             "gold_mass": readouts["answer_sum/gold/mass"],
                             "gold_nll": readouts["answer_sum/gold/nll"],
                             "hate_logodds": readouts.get("answer_sum/margin/hate"),
                             "group_label_logodds": {label: readouts["answer_sum/margin/" + label]
                                                       for label in GROUP_LABELS} if task == "group" else None,
                             "pred_group_size": d["pred_group_size"], "group_error_count": d["group_error_count"],
                             "score_mode_sensitive": d["score_mode_sensitive"],
                             "score_mode_predictions": d["score_mode_predictions"],
                             "core_mask": p["core_mask"][task], "six_condition_mask": p["six_condition_mask"][task],
                             **d["context"], "task_specific_null_reason": "field_belongs_to_other_task",
                             "context_status": p["context_status"]})
    return rows


def transition_counts(profiles, before, after, task):
    counts = Counter("".join(str(int(p["conditions"][c][task]["correct"])) for c in (before, after))
                     for p in profiles)
    row = {mask: counts[mask] for mask in ("00", "01", "10", "11")}
    require(sum(row.values()) == len(profiles), "transition counts do not close")
    if task == "group":
        deltas = [p["conditions"][after][task]["group_error_count"] - p["conditions"][before][task]["group_error_count"]
                  for p in profiles]
        sizes = [p["conditions"][after][task]["pred_group_size"] - p["conditions"][before][task]["pred_group_size"]
                 for p in profiles]
        row.update(group_error_decreased=sum(d < 0 for d in deltas),
                   group_error_increased=sum(d > 0 for d in deltas), group_error_unchanged=sum(d == 0 for d in deltas),
                   group_error_delta_mean=float(np.mean(deltas)) if deltas else None,
                   pred_group_size_delta_mean=float(np.mean(sizes)) if sizes else None,
                   still_wrong_but_error_decreased=sum(d < 0 and not p["conditions"][before][task]["correct"]
                       and not p["conditions"][after][task]["correct"] for p, d in zip(profiles, deltas)))
    return row


def label_transition(profiles, before, after, task, label):
    counts, changed = Counter(), Counter()
    for p in profiles:
        gold = p["gold"][task] == label if task == "hate" else label in p["gold"][task]
        a, b = [label in p["conditions"][c][task]["prediction"]["labels"] for c in (before, after)]
        counts[str(int(a == gold)) + str(int(b == gold))] += 1
        state_a = ("TP" if a else "FN") if gold else ("FP" if a else "TN")
        state_b = ("TP" if b else "FN") if gold else ("FP" if b else "TN")
        changed[state_a + "_to_" + state_b] += 1
    return {**{mask: counts[mask] for mask in ("00", "01", "10", "11")},
            **{a + "_to_" + b: changed[a + "_to_" + b]
               for a, b in (("FN", "TP"), ("TP", "FN"), ("FP", "TN"), ("TN", "FP"))}}


def paired_values(profile, task):
    rows = {}
    for effect, coefficients in EFFECTS.items():
        rows[effect] = math.fsum(w * profile["conditions"][c][task]["readouts"]
                                ["answer_sum/gold/best_nongold_margin"] for c, w in coefficients.items())
        if task == "hate":
            rows[effect + "_hate_logodds"] = math.fsum(w * profile["conditions"][c][task]["readouts"]
                                                       ["answer_sum/margin/hate"] for c, w in coefficients.items())
    return rows


def metadata_values(profile, before, after, task):
    resource = profile["resources"]
    a, b = [profile["conditions"][c][task]["context"] for c in (before, after)]
    return {"lq_count": resource["lq_count"], "ld_only_count": resource["ld_only_count"],
            "union_count": resource["union_count"], "prompt_tokens_before": a["prompt_tokens"],
            "prompt_tokens_after": b["prompt_tokens"],
            "prompt_tokens_before_minus_after": a["prompt_tokens"] - b["prompt_tokens"],
            "dictionary_tokens_before": a["dictionary_tokens"], "dictionary_tokens_after": b["dictionary_tokens"],
            "demo_hate_n": resource["demo_hate_counts"]["hate"] if resource["demo_hate_counts"] else None,
            **{"demo_" + label + "_n": resource["demo_group_counts"][label]
               if resource["demo_group_counts"] is not None else None for label in GROUP_LABELS}}


def analyze_tables(profiles, output, paths, config):
    tables = output / "tables"
    stats = counts_tensor(profiles)
    write_csv(tables / "condition_task_rows.csv", condition_rows(profiles))
    write_csv(tables / "classification.csv", point_metrics(profiles, stats))
    masks, transitions, labels, strata, continuous, metadata = [], [], [], [], [], []
    paired = {(p["query_id"], task): paired_values(p, task) for p in profiles for task in TASKS}
    write_csv(tables / "paired_readouts.csv", (
        {"schema_version": SCHEMA, "query_id": p["query_id"], "task": task,
         "lex_hit": p["lex_hit"], **paired[p["query_id"], task]}
        for p in profiles for task in TASKS))
    for stratum in STRATA:
        indices = [i for i, p in enumerate(profiles) if stratum in stratum_names(p)]
        subset = [profiles[i] for i in indices]
        base = {"schema_version": SCHEMA, "stratum": stratum, "n": len(subset)}
        makeup = composition(subset)
        strata.extend(base | makeup | row for row in point_metrics(subset, stats[indices]))
        for task in TASKS:
            mask_counts = Counter(p["core_mask"][task] for p in subset)
            require(sum(mask_counts.values()) == len(subset), "core mask counts do not close")
            masks.extend(base | {"task": task, "core_order": list(CORE), "core_mask": f"{i:04b}",
                                 "count": mask_counts[f"{i:04b}"]} for i in range(16))
            readout_names = list(EFFECTS) + ([name + "_hate_logodds" for name in EFFECTS] if task == "hate" else [])
            for name in readout_names:
                continuous.append(base | {"task": task, "effect": name,
                    **summary([paired[p["query_id"], task][name] for p in subset])})
        for contrast, (before, after) in PAIRS.items():
            for task in TASKS:
                key = base | {"contrast": contrast, "before": before, "after": after, "task": task}
                transitions.append(key | transition_counts(subset, before, after, task))
                for label in (("hate",) if task == "hate" else GROUP_LABELS):
                    labels.append(key | {"label": label} | label_transition(subset, before, after, task, label))
                for transition in ("00", "01", "10", "11"):
                    members = [p for p in subset if "".join(str(int(p["conditions"][c][task]["correct"]))
                               for c in (before, after)) == transition]
                    feature_rows = [metadata_values(p, before, after, task) for p in members]
                    # Use an explicit feature catalog even for empty transition cells.
                    features = list(metadata_values(profiles[0], before, after, task))
                    metadata.extend(key | {"transition": transition, "feature": feature,
                                          "stratum_n": len(subset), "transition_n": len(members),
                                          **summary([row[feature] for row in feature_rows])} for feature in features)
    for name, rows in (("core_masks.csv", masks), ("transitions.csv", transitions),
                       ("per_label_transitions.csv", labels), ("strata.csv", strata),
                       ("continuous_summaries.csv", continuous), ("metadata_by_transition.csv", metadata)):
        write_csv(tables / name, rows)
    numeric_quality = []
    for task in TASKS:
        for c in CONDITIONS:
            data = [p["conditions"][c][task] for p in profiles]
            numeric_quality.append({"task": task, "condition": c, "n": len(profiles),
                                    "exact_ties": sum(d["prediction"]["tied_top_count"] > 1 for d in data),
                                    "within_two_epsilon": sum(d["prediction"]["within_two_epsilon"] for d in data),
                                    "score_mode_sensitive": sum(d["score_mode_sensitive"] for d in data)})
    write_csv(tables / "numeric_quality.csv", numeric_quality)
    ci, draw_hash = bootstrap(stats, repetitions=config["bootstrap"]["repetitions"], seed=config["bootstrap"]["seed"])
    old_rows, new_rows = ci[:4], ci[4:]
    with paths["original_ci"].open(encoding="utf-8", newline="") as f:
        reference = list(csv.DictReader(f))
    require(len(reference) == 4, "original endpoint count differs")
    for saved in reference:
        matches = [r for r in old_rows if (r["task"], r["contrast"]) == (saved["task"], saved["contrast"])]
        require(len(matches) == 1, "original endpoint missing/duplicated")
        for field in ("difference", "ci95_low", "ci95_high"):
            close(matches[0][field], float(saved[field]), "original interval " + field)
    write_csv(tables / "original_ci_check.csv", old_rows)
    write_csv(tables / "exploratory_ci.csv", new_rows)
    baseline = read_json(output / "audit/baseline_reproduction.json")
    baseline.update(original_ci_status="passed", original_ci_recomputed=4,
                    original_ci=old_rows, shared_query_indices_sha256=draw_hash)
    write_json(output / "audit/baseline_reproduction.json", baseline)
    write_json(output / "audit/statistics_check.json", {
        "schema_version": SCHEMA, "passed": True, "full_population_n": len(profiles),
        "condition_task_rows": len(profiles) * 12, "all_16_masks_including_zero_cells": True,
        "transition_counts_closed": True, "near_ties_retained_in_population": True,
        "exploratory_endpoints": 4, "original_ci_regression_passed": True,
        "shared_query_indices_sha256": draw_hash, "quantile_method": "linear",
        "same_resamples_for_all_tasks_conditions_endpoints": True,
        "strata_n": {name: sum(name in stratum_names(p) for p in profiles) for name in STRATA},
    })


def selection_exports(profiles, output, config):
    result = select_cases(profiles, config["selection"]["seed"], config["selection"]["per_bucket_quotas"])
    require(result == select_cases(profiles, config["selection"]["seed"], config["selection"]["per_bucket_quotas"]),
            "selection did not reproduce")
    cases, lookup = output / "cases", {p["query_id"]: p for p in profiles}
    write_jsonl(cases / "candidate_pool.jsonl", (p for p in profiles if p["memberships"]))
    write_jsonl(cases / "boundary_cases.jsonl", (p for p in profiles if any(
        p["conditions"][c][t]["prediction"]["within_two_epsilon"] for c in CONDITIONS for t in TASKS)))
    write_jsonl(cases / "selection_events.jsonl", result["events"])
    selected = result["selected"]
    sides = {side: [p for p in selected if p["split"] == side] for side in ("discovery", "reserve")}
    for side, members in sides.items():
        write_jsonl(cases / (side + ".jsonl"), members)
    overlap = {}
    for field in ("union_ids", "lq_ids", "ld_only_ids", "demo_ids"):
        sets = {side: {value for p in members for value in lookup[p["query_id"]]["resources"].get(field) or []}
                for side, members in sides.items()}
        union = sets["discovery"] | sets["reserve"]
        known = all(lookup[p["query_id"]]["resources"].get(field) is not None for p in selected)
        overlap[field] = {"discovery_unique": len(sets["discovery"]) if known else None,
                          "reserve_unique": len(sets["reserve"]) if known else None,
                          "shared_unique": len(sets["discovery"] & sets["reserve"]) if known else None,
                          "jaccard": len(sets["discovery"] & sets["reserve"]) / len(union) if union and known else None,
                          "missing_reason": None if known and union else "missing_metadata" if not known else "empty_union"}
    catalog = [("H_rescue", "hate"), ("H_residual", "hate"), ("H_removal_harm", "hate"),
               ("G_category_support", "group"), ("G_category_harm", "group"),
               ("H_joint_only", "hate"), ("G_joint_only", "group"),
               *((bucket, task) for bucket in ("Stable_correct", "Stable_wrong") for task in TASKS)]
    counts = []
    for bucket, task in catalog:
        members = [(p, m) for p in profiles for m in p["memberships"]
                   if (m["bucket"], m["focus_task"]) == (bucket, task)]
        counts.append({"bucket": bucket, "task": task, "raw_candidates": len(members),
                       "numeric_eligible": sum(m["numeric_eligible"] for _, m in members),
                       "eligible_score_mode_sensitive": sum(m["numeric_eligible"] and m["score_mode_sensitive"] for _, m in members),
                       "eligible_context_complete": sum(m["numeric_eligible"] and p["context_status"] == "complete" for p, m in members)})
    write_csv(output / "tables/candidate_counts.csv", counts)
    write_csv(cases / "selection_quotas.csv", result["quotas"])
    families = Counter(p["family_key"] for p in profiles)
    # Select the first review batch across primary buckets, using frozen order within each.
    review_queues = {bucket: deque(p for p in sides["discovery"] if p["primary_bucket"] == bucket) for bucket in BUCKETS}
    batch = []
    while len(batch) < config["selection"]["initial_review_count"] and any(review_queues.values()):
        for bucket in BUCKETS:
            if review_queues[bucket] and len(batch) < config["selection"]["initial_review_count"]:
                batch.append(review_queues[bucket].popleft()["query_id"])
    require(not ({p["family_key"] for p in sides["discovery"]} & {p["family_key"] for p in sides["reserve"]}),
            "query family leaked across splits")
    write_json(cases / "selection_manifest.json", {
        "schema_version": SCHEMA, "passed": True, "selection_seed": config["selection"]["seed"],
        "algorithm": config["selection"]["algorithm"], "bucket_order": list(BUCKETS),
        "deterministic_replay_passed": True, "family_normalization": "NFC-and-LF-only-no-strip-no-casefold",
        "unique_families": len(families), "duplicate_content_families": sum(n > 1 for n in families.values()),
        "split_frame_counts": dict(Counter(p["split"] for p in profiles)),
        "selected_counts": {side: len(members) for side, members in sides.items()}, "quotas": result["quotas"],
        "candidate_counts": counts, "cross_split_resource_overlap": overlap, "initial_review_query_ids": batch,
        "query_and_family_unique": True, "reserve_cards_exported": False,
        "reserve_status": "dev-derived-mechanism-reserve-not-unseen-test",
        "reviewer_content_opened": {"discovery": False, "reserve": False},
        "automated_full_dev_feature_and_family_processing": True,
    })


def md_cell(value):
    text = canonical(value) if not isinstance(value, str) else value
    return text.replace("|", "\\|").replace("\n", "<br>")


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |",
                       *("| " + " | ".join(md_cell(v) for v in row) + " |" for row in rows)])


def review_export(profiles, output, paths, config):
    cases = output / "cases"
    selected = list(json_lines(cases / "discovery.jsonl"))
    ids = {p["query_id"] for p in selected}
    lookup = {p["query_id"]: p for p in profiles}
    queries = {r["id"]: r for r in json_lines(paths["queries"]) if r["id"] in ids}
    contexts = {(r["query_id"], r["task"], r["condition"]): r
                for r in json_lines(paths["contexts"]) if r["query_id"] in ids} if "contexts" in paths else {}
    entries = {r["lexicon_id"]: r for r in read_json(paths["lexicon"])["terms"]} if "lexicon" in paths else {}
    fit = unique_index(json_lines(paths["fit_catalog"]), "id", "fit catalog") if "fit_catalog" in paths else {}
    (cases / "cards").mkdir(parents=True, exist_ok=True)
    templates, index = [], []
    for choice in selected:
        q, task = choice["query_id"], choice["focus_task"]
        p = lookup[q]
        resource = p["resources"]
        card_id = f"{choice['selection_ordinal']:02d}-{q}"
        resource_path = cases / "cards" / (card_id + "-1-resources.md")
        trajectory_path = cases / "cards" / (card_id + "-2-trajectories.md")
        dictionary = [entries[i] for i in resource["union_ids"] or [] if i in entries]
        demo_records = [fit[d] for d in resource["demo_ids"] or [] if d in fit]
        resource_text = [f"# 查询 {q}：资源初读", "", f"Focus task: {task}。本页不展示查询 Gold、模型预测或候选桶。",
                         "", "## 查询", "", queries[q]["content"], "", "## 冻结词典", "",
                         "S 使用以下义项定义；SG 另包含显式类别字段。两个条件均使用相同词条并集。", ""]
        for entry in dictionary:
            eid = entry["lexicon_id"]
            origins = [name for name, field in (("Lq", "lq_ids"), ("Ld", "ld_ids")) if eid in (resource[field] or [])]
            resource_text += [f"### {entry['term']} ({eid}; 来源 {'+'.join(origins)})", ""]
            for sense in entry["senses"]:
                resource_text += [f"义项 `{sense['sense_id']}`；显式类别：{canonical(sense['categories'])}", "",
                                  sense["definition"] or "未提供释义（冻结时为空；未补写）", ""]
        resource_text += ["## 固定示例（原顺序）", ""]
        trace = contexts.get((q, task, "CD"), {}).get("trace", {}).get("demo", [])
        for ordinal, demo in enumerate(demo_records):
            shown_value = trace[ordinal]["rendered_output"] if len(trace) > ordinal else demo["projection"][task]
            if len(trace) > ordinal:
                require(trace[ordinal]["demo_id"] == demo["id"] and trace[ordinal]["ordinal"] == ordinal,
                        "case demo trace order differs")
                require(shown_value == demo["projection"][task], "case demo answer differs")
            # The trace stores native labels/lists; the original renderer JSON-encodes them.
            shown = canonical(shown_value)
            resource_text += [f"### 示例 {ordinal + 1} / ID {demo['id']}", "", demo["content"], "",
                              f"该任务的冻结答案：{shown}", "",
                              f"命中词条：{canonical((resource['demo_match_ids'] or {}).get(demo['id']))}", ""]
        if p["context_status"] != "complete":
            resource_text += ["## 缺失项", "", canonical(resource["missing_reasons"]), ""]
        resource_text += ["## 初读记录", "", "先记录词义歧义、引用/否定/立场、定义适配性、类别关系与示例对应性；再打开第二页。", ""]
        resource_path.write_text("\n".join(resource_text), encoding="utf-8")
        trajectory = [f"# 查询 {q}：预测轨迹", "", f"主桶：{choice['primary_bucket']}；focus task：{task}。",
                      "", "全部候选标签：" + canonical(p["candidate_labels"]), "",
                      "来源：" + canonical({name: config["inputs"][name]["sha256"] for name in ("plan", "raw", "analysis", "queries", "lexicon")}), ""]
        for t in TASKS:
            trajectory += [f"## {t}", "", f"Gold：{canonical(p['gold'][t])}；四位轨迹：{p['core_mask'][t]}；六位轨迹：{p['six_condition_mask'][t]}。", "",
                           md_table(["condition", "prediction", "correct", "top gap", "gold margin", "tie count", "mode sensitive", "prompt tokens", "dictionary tokens"],
                           [[c, p["conditions"][c][t]["prediction"]["labels"], p["conditions"][c][t]["correct"],
                             p["conditions"][c][t]["prediction"]["top_score_gap"],
                             p["conditions"][c][t]["readouts"]["answer_sum/gold/best_nongold_margin"],
                             p["conditions"][c][t]["prediction"]["tied_top_count"],
                             p["conditions"][c][t]["score_mode_sensitive"], p["conditions"][c][t]["context"]["prompt_tokens"],
                             p["conditions"][c][t]["context"]["dictionary_tokens"]] for c in CONDITIONS]), "",
                           "配对连续读数：" + canonical(paired_values(p, t)), ""]
        trajectory += ["## 审阅字段", "", "记录候选解释、替代解释、Gold 争议及下一阶段可证伪的输入干预。未执行的干预只有假设，没有结果。", ""]
        trajectory_path.write_text("\n".join(trajectory), encoding="utf-8")
        write_json(cases / "card_data" / (card_id + ".json"), {
            "schema_version": SCHEMA, "selection": choice, "profile": p, "query": queries[q],
            "lexicon_entries": dictionary, "demonstrations": demo_records,
            "contexts": [contexts[q, t, c] for t in TASKS for c in CONDITIONS if (q, t, c) in contexts],
            "source_sha256": {name: spec["sha256"] for name, spec in config["inputs"].items()},
        })
        index.append({"query_id": q, "focus_task": task, "primary_bucket": choice["primary_bucket"],
                      "resources_card": str(resource_path.relative_to(output)),
                      "trajectory_card": str(trajectory_path.relative_to(output)), "context_status": p["context_status"]})
        templates.append({"query_id": q, "focus_task": task, "primary_bucket": choice["primary_bucket"],
                          "reviewer": "", "review_kind": "", "stage1_resource_notes": "", "ambiguity_stance": "",
                          "definition_fit": "", "category_relation": "", "demo_correspondence": "",
                          "stage2_candidate_explanation": "", "alternative_explanation": "", "gold_dispute": "",
                          "falsifiable_followup": "", "patching_defer_reason": "", "review_status": "unreviewed"})
    write_csv(cases / "review_template.csv", templates)
    write_json(cases / "cards_index.json", index)
    selection = read_json(cases / "selection_manifest.json")
    batch_ids = selection["initial_review_query_ids"]
    by_id = {r["query_id"]: r for r in index}
    write_json(cases / "initial_review_batch.json", [by_id[q] for q in batch_ids])
    write_json(cases / "review_access.json", {
        "schema_version": SCHEMA, "discovery_cards_exported": len(index), "reserve_cards_exported": 0,
        "discovery_review_status": "unreviewed", "reserve_reviewer_content_opened": False,
        "automated_full_dev_features_processed": True, "human_review_performed": False,
        "ai_assisted_review_performed": False, "events": [],
    })


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_report(output):
    tables, cases = output / "tables", output / "cases"
    points = csv_rows(tables / "classification.csv")
    transitions = [r for r in csv_rows(tables / "transitions.csv") if r["stratum"] == "all"]
    ci = csv_rows(tables / "exploratory_ci.csv")
    selection = read_json(cases / "selection_manifest.json")
    quality = csv_rows(tables / "numeric_quality.csv")
    audit = read_json(output / "audit/input_audit.json")
    access = read_json(cases / "review_access.json")
    pct = lambda x: f"{100 * float(x):.2f}"
    report = ["# Qwen3-8B NoCat：逐查询配对分析与机制候选集", "",
              "已完成完整 dev 的 CPU 事后分析；新增统计均为探索性。未新增模型 forward，未访问 test，未修改原始产物。", "",
              "## 输入与完成状态", "",
              f"643 queries、7,716 blocks、131,172 candidates；输入状态：{audit['context_status']}。来源哈希、逐块预测/Gold及使用的读数已核对。原分类表和四个区间复现通过。", "",
              "运行配置与实际源码快照见 `config.frozen.json`、`manifest.json` 和 `source/`。新增实现尚未单独提交时，以源码 SHA256 为执行版本身份。", "",
              "## 六条件指标", "",
              md_table(["条件", "Hate Macro-F1 (%)", "Group Micro-F1 (%)", "两任务同时正确"],
                       [[r["condition"], pct(r["hate_macro_f1"]), pct(r["group_micro_f1"]), r["joint_correct"]] for r in points]), "",
              "## 双向转换（完整 643 条）", "",
              "下表 01 为错→对，10 为对→错；group 的对错指集合完全匹配，不等同于 Micro-F1 的逐查询贡献。", "",
              md_table(["对比", "任务", "00", "01 纠正", "10 退化", "11"],
                       [[r["contrast"], r["task"], r["00"], r["01"], r["10"], r["11"]] for r in transitions]), "",
              "## 四项新增探索性区间", "",
              "单位为百分点；逐点、描述性、未作多重比较校正。配对查询 bootstrap 10,000 次，PCG64 seed 42，所有任务、条件与端点共享抽样；每次重算 TP/FP/FN 与 F1。", "",
              md_table(["端点", "任务", "差值 (pp)", "95% 区间 (pp)"],
                       [[r["contrast"], r["task"], pct(r["difference"]),
                         f"[{pct(r['ci95_low'])}, {pct(r['ci95_high'])}]"] for r in ci]), "",
              "E_S_given_D_F1 = F(SD) − F(D)。J_remove_by_D_F1 = [F(SD) − F(SGD)] − [F(S) − F(SG)]。", "",
              "J 是聚合 F1 尺度的差中差；逐查询 gold_margin 的 I_S_D 另见 paired_readouts.csv，两者不是同一个量。", "",
              "## 候选池与固定配额", "",
              md_table(["候选桶", "任务", "行为候选", "gap 筛选后", "其中计分口径敏感"],
                       [[r["bucket"], r["task"], r["raw_candidates"], r["numeric_eligible"],
                         r["eligible_score_mode_sensitive"]] for r in selection["candidate_counts"]]), "",
              f"实际选取 discovery {selection['selected_counts']['discovery']} 条、reserve {selection['selected_counts']['reserve']} 条。配额不足不补齐；query/family 去重及选样重放通过。", "",
              md_table(["主桶", "侧", "目标", "实际", "缺额", "原因"],
                       [[r["bucket"], r["split"], r["target"], r["selected"], r["shortfall"], r["shortfall_reason"]]
                        for r in selection["quotas"]]), "",
              "完整成员资格允许重叠；主桶仅用于调度。G_category_harm 完整保留，但本版没有强制名额。每次排除和跳过见 cases/selection_events.jsonl。", "",
              "## 数值边界与计分口径", "",
              md_table(["任务", "条件", "精确并列", "gap ≤ 2ε", "四口径 argmax 有变化"],
                       [[r["task"], r["condition"], r["exact_ties"], r["within_two_epsilon"], r["score_mode_sensitive"]]
                        for r in quality]), "",
              "近并列保留在全量统计中，只限制对应候选类型所需的任务/条件。计分口径敏感性仅作标记，不更换 answer_sum 主口径。", "",
              "## 留出侧与资源重叠", "",
              md_table(["资源", "discovery 去重 ID 数", "reserve 去重 ID 数", "共有 ID 数", "Jaccard"],
                       [[field, r["discovery_unique"], r["reserve_unique"], r["shared_unique"], r["jaccard"]]
                        for field, r in selection["cross_split_resource_overlap"].items()]), "",
              "以上前两列指各侧去重后的集合大小，并非两侧互斥。reserve 仍来自已暴露 dev，且资格依赖行为结果；它仅保留给后续未参与路径定位的机制评估，不是新 test。", "",
              "## 案例审阅与下一阶段", "",
              f"已导出 {access['discovery_cards_exported']} 个定位案例的资源页、轨迹页和完整上下文 JSON；reserve 案例卡未导出。", "",
              f"审阅状态：{access['discovery_review_status']}；人工审阅完成：{access['human_review_performed']}；AI 辅助初读：{access['ai_assisted_review_performed']}。", "",
              "首批审阅按主桶轮询的固定顺序，见 cases/initial_review_batch.json。人工审阅模板见 cases/review_template.csv。若有 AI 辅助初读，其记录单列，不冒充独立人工审阅。", "",
              "优先解释 H_rescue 与 H_residual 的区别，保留 removal_harm 与 category_harm 反例；joint_only 同时保留证据相加跨阈值和信息依赖两种解释。下一阶段先设计输入对照与重复性验证，再另定激活干预协议。", "",
              "## 解释边界与其余表格", "",
              "NoCat 只删除显式类别字段；S 仍可带类别语义，D 仍含答案。S 来自查询与固定示例词条的并集，D 关闭时仍含示例贡献词条。删除同时改变长度和位置，当前结果不能分离这些因素。行为交互、正确预测或某种位型都不证明内部机制。", "",
              "全部 16 位型（含零计数）、分层构成、逐标签转换、连续读数和长度/词条数的描述性关系分别见 tables/core_masks.csv、strata.csv、per_label_transitions.csv、continuous_summaries.csv、metadata_by_transition.csv。分层与连续读数没有临时增加区间。", "",
              "案例全文、逐查询表及完整提示均保留本地；本次绑定的原始输入文件哈希在运行前后保持一致。", ""]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


def now():
    return datetime.now(timezone.utc).isoformat()


class CPUOnlyImports:
    forbidden = {"torch", "transformers", "accelerate", "vllm", "cupy", "jax",
                 "diagnostics.general_model_nolabel_execution", "diagnostics.general_model_nolabel"}

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in self.forbidden):
            raise RuntimeError("model/runtime import forbidden in CPU posthoc analysis: " + fullname)
        return None


class Runner:
    phases = ("validate", "analyze", "select", "review-export")

    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(self.config_path)
        config = self.config
        require(config["schema_version"] == SCHEMA, "config schema mismatch", "identity_mismatch")
        require(config["allow_model_forward"] is False and config["allow_test_content"] is False
                and config["overwrite_sources"] is False, "analysis scope changed", "identity_mismatch")
        require(config["prediction_score"] == "answer_sum"
                and config["tie_rule"] == "smallest-canonical-ordinal"
                and config["bootstrap"] == {"repetitions": 10000, "seed": 42, "rng": "PCG64",
                    "shared_query_draws": True, "quantile_method": "linear", "zero_division": 0}
                and config["selection"]["seed"] == 20260907
                and config["selection"]["per_bucket_quotas"] == {"discovery": 4, "reserve": 2},
                "prediction/bootstrap/selection policy differs", "identity_mismatch")
        require(config["conditions"] == list(CONDITIONS) and config["core_conditions"] == list(CORE)
                and config["strata"] == list(STRATA) and config["selection"]["bucket_order"] == list(BUCKETS)
                and config["ci_contrasts"] == CI_CONTRASTS and config["margin_effects"] == EFFECTS,
                "frozen analysis rules differ from implementation", "identity_mismatch")
        self.output = Path(config["output_dir"]).resolve()
        source_root = Path(config["source_base"]).resolve()
        require(not self.output.is_relative_to(source_root) and not source_root.is_relative_to(self.output),
                "output overlaps source experiment", "identity_mismatch")
        for spec in config["inputs"].values():
            require(not Path(spec["path"]).resolve().is_relative_to(self.output),
                    "output contains a source input", "identity_mismatch")
        source_files = [Path(__file__).resolve(),
                        ROOT / "scripts/stage1/general_model_nolabel_paired_cases.py",
                        ROOT / "src/diagnostics/general_model_numeric_analysis.py",
                        ROOT / "src/tests/test_general_model_nolabel_paired_cases.py"]
        code_hashes = {str(path.relative_to(ROOT)): file_sha(path) for path in source_files}
        identity = digest(canonical({"config": config, "code_sha256": code_hashes}))
        self.manifest_path = self.output / "manifest.json"
        if self.output.exists():
            require(self.manifest_path.is_file(), "existing output has no identity manifest", "identity_mismatch")
            self.manifest = read_json(self.manifest_path)
            require(self.manifest["identity"] == identity,
                    "output identity differs; use a new run directory", "identity_mismatch")
        else:
            self.output.mkdir(parents=True)
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            self.manifest = {"schema_version": SCHEMA, "identity": identity, "status": "initialized",
                             "created_at": now(), "base_commit": config["base_commit"], "execution_commit": commit,
                             "source_run": config["source_run"], "config_file_sha256": file_sha(self.config_path),
                             "code_sha256": code_hashes, "source_snapshot_is_execution_identity": True,
                             "environment": {"python": platform.python_version(), "executable": sys.executable,
                                             "numpy": np.__version__, "platform": platform.platform(),
                                             "unicode_database": unicodedata.unidata_version,
                                             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                             "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                                             "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
                             "phases": {}, "artifacts": {}, "model_forward_executed": False, "test_content_read": False}
            write_json(self.output / "config.frozen.json", config)
            for source in source_files:
                target = self.output / "source" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            write_json(self.manifest_path, self.manifest)
        self.profiles = None

    def checkpoint(self):
        self.manifest["artifacts"] = {str(p.relative_to(self.output)): file_sha(p)
                                      for p in sorted(self.output.rglob("*"))
                                      if p.is_file() and p != self.manifest_path and not p.name.endswith(".tmp")}
        self.manifest["updated_at"] = now()
        write_json(self.manifest_path, self.manifest)

    def check_artifact(self, relative):
        require(relative in self.manifest["artifacts"] and file_sha(self.output / relative) == self.manifest["artifacts"][relative],
                "derived artifact changed: " + relative, "identity_mismatch")

    def load_profiles(self):
        if self.profiles is None:
            self.check_artifact("tables/query_profiles.jsonl")
            self.profiles = list(json_lines(self.output / "tables/query_profiles.jsonl"))
        return self.profiles

    def run(self, phase):
        guard = CPUOnlyImports()
        require(not any(name in sys.modules for name in guard.forbidden), "model runtime already imported")
        sys.meta_path.insert(0, guard)
        try:
            paths, source_receipt, missing = verify_inputs(self.config)
            if "source_receipt" in self.manifest:
                require(source_receipt == self.manifest["source_receipt"],
                        "input availability changed; create a new run", "identity_mismatch")
            self.manifest["source_receipt"] = source_receipt
            phases = self.phases if phase == "all" else (() if phase == "report" else (phase,))
            for stage in phases:
                if self.manifest["phases"].get(stage, {}).get("status") == "complete":
                    print(canonical({"phase": stage, "status": "already_complete"}), flush=True)
                    continue
                index = self.phases.index(stage)
                if index:
                    previous = self.phases[index - 1]
                    require(self.manifest["phases"].get(previous, {}).get("status") == "complete",
                            "run preceding phase first: " + previous, "analysis_failed")
                self.manifest["status"] = "running"
                self.manifest["current_phase"] = stage
                self.manifest["phases"][stage] = {"status": "running", "started_at": now()}
                write_json(self.manifest_path, self.manifest)
                print(canonical({"phase": stage, "status": "running"}), flush=True)
                if stage == "validate":
                    self.profiles, audit = validate_and_profile(self.config, paths, missing)
                    audit.update(schema_version=SCHEMA, source_receipt=source_receipt,
                                 checks=["source_identity", "paired_alignment", "selection_pending"])
                    write_json(self.output / "audit/input_audit.json", audit)
                    baseline = check_baseline_points(self.profiles, paths)
                    write_json(self.output / "audit/baseline_reproduction.json", baseline)
                    write_jsonl(self.output / "tables/query_profiles.jsonl", self.profiles)
                elif stage == "analyze":
                    analyze_tables(self.load_profiles(), self.output, paths, self.config)
                elif stage == "select":
                    self.check_artifact("audit/statistics_check.json")
                    selection_exports(self.load_profiles(), self.output, self.config)
                elif stage == "review-export":
                    for relative in ("cases/discovery.jsonl", "cases/reserve.jsonl", "cases/selection_manifest.json"):
                        self.check_artifact(relative)
                    review_export(self.load_profiles(), self.output, paths, self.config)
                self.manifest["phases"][stage].update(status="complete", completed_at=now())
                self.checkpoint()
                print(canonical({"phase": stage, "status": "complete"}), flush=True)
            if all(self.manifest["phases"].get(p, {}).get("status") == "complete" for p in self.phases):
                for relative in ("tables/classification.csv", "tables/transitions.csv", "tables/exploratory_ci.csv",
                                 "cases/selection_manifest.json", "audit/input_audit.json"):
                    self.check_artifact(relative)
                make_report(self.output)
                _, after, _ = verify_inputs(self.config)
                require(after == source_receipt, "source inputs changed during analysis", "identity_mismatch")
                write_json(self.output / "audit/source_unchanged.json", {
                    "schema_version": SCHEMA, "passed": True, "before_equals_after": True,
                    "scope": "all_explicitly_bound_input_files", "input_count": len(after),
                    "source_sha256": {k: v["sha256"] for k, v in after.items()},
                })
                self.manifest["status"] = read_json(self.output / "audit/input_audit.json")["context_status"]
                self.manifest["automated_analysis_complete"] = True
                self.manifest["review_status"] = read_json(self.output / "cases/review_access.json")["discovery_review_status"]
            else:
                self.manifest["status"] = "partial"
            self.checkpoint()
            return {"status": self.manifest["status"], "output": str(self.output), "phases": self.manifest["phases"]}
        except Exception as exc:
            status = getattr(exc, "status", "analysis_failed")
            self.manifest.update(status=status, error=str(exc), failed_at=now())
            write_json(self.output / "audit/failure.json", {"status": status, "message": str(exc),
                       "phase": self.manifest.get("current_phase"), "time": now()})
            self.checkpoint()
            raise
        finally:
            sys.meta_path.remove(guard)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=(*Runner.phases, "all", "report"), default="all")
    args = parser.parse_args(argv)
    try:
        result = Runner(args.config).run(args.phase)
    except Exception as exc:
        print(canonical({"status": getattr(exc, "status", "analysis_failed"), "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
