"""Frozen paired classification endpoints for explicit category-field removal."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from diagnostics.general_model_numeric_analysis import (
    GROUP_LABELS, SCORE_MODES, _gold_ordinal, _summary, block_analysis,
)
from diagnostics.general_model_tasks import evaluate_predictions
from diagnostics.general_model_nolabel_package import CONDITIONS


TASKS = ("hate", "group")
PRIMARY_CONTRASTS = ("NoCat_without_D", "NoCat_with_D")
CONTRASTS = {
    "NoCat_without_D": {"CLnewNoCat":1, "CLnew":-1},
    "NoCat_with_D": {"CLDnewNoCat":1, "CLDnew":-1},
    "D_vs_C0": {"CD":1, "C0":-1},
    "Full_vs_C0": {"CLnew":1, "C0":-1},
    "NoCat_vs_C0": {"CLnewNoCat":1, "C0":-1},
    "Full_given_D": {"CLDnew":1, "CD":-1},
    "NoCat_given_D": {"CLDnewNoCat":1, "CD":-1},
}
CLASSIFICATION_POLICY = {
    "primary_metrics":{"hate":"macro_f1", "group":"micro_f1"},
    "primary_contrasts":list(PRIMARY_CONTRASTS), "primary_ci_targets":4,
    "decision":"argmax-answer_sum", "tie_rule":"smallest-canonical-ordinal",
    "group_decision":"32-candidate-set-argmax-no-hate-postprocessing",
    "zero_division":0, "bootstrap_repetitions":10000, "bootstrap_seed":42,
    "bootstrap_rng":"NumPy-PCG64", "bootstrap_unit":"paired-query",
    "bootstrap_population":"all-643-dev", "bootstrap_interval":"percentile-95-pointwise-descriptive",
    "bootstrap_f1":"recompute-from-resampled-TP-FP-FN-not-mean-query-F1",
    "multiplicity_correction":False, "auxiliary_ci":False,
    "group_labels":list(GROUP_LABELS), "hate_labels":["hate", "non-hate"],
}


def ratios(num, den):
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)


def primary_from_counts(counts):
    """Trailing dimensions are 7 labels x (TP, FP, FN), with two hate labels first."""
    tp, fp, fn = (counts[..., i] for i in range(3))
    per_label = ratios(2*tp, 2*tp+fp+fn)
    return {"hate":per_label[..., :2].mean(axis=-1),
            "group":ratios(2*tp[..., 2:].sum(axis=-1),
                           (2*tp+fp+fn)[..., 2:].sum(axis=-1))}


def paired_f1_bootstrap(stats, *, repetitions=10000, seed=42):
    """One shared draw across all conditions/tasks; recompute nonlinear F1 each draw."""
    if (stats.ndim != 4 or stats.shape[1:] != (len(CONDITIONS), 7, 3)
            or len(stats) == 0 or repetitions <= 0
            or not np.isfinite(stats).all() or (stats < 0).any()):
        raise ValueError("invalid query-level classification counts")
    n = len(stats)
    rng = np.random.Generator(np.random.PCG64(seed))
    flat = stats.reshape(n, -1).astype(np.float64)
    positions = {c:i for i,c in enumerate(CONDITIONS)}
    draws = {t:np.empty((repetitions,len(PRIMARY_CONTRASTS))) for t in TASKS}
    for start in range(0, repetitions, 128):
        size = min(128, repetitions-start)
        indices = rng.integers(0,n,size=(size,n))
        weights = np.stack([np.bincount(row,minlength=n) for row in indices]).astype(np.float64)
        metrics = primary_from_counts((weights @ flat).reshape(size,len(CONDITIONS),7,3))
        for j,contrast in enumerate(PRIMARY_CONTRASTS):
            for t in TASKS:
                draws[t][start:start+size,j] = sum(
                    weight*metrics[t][:,positions[c]] for c,weight in CONTRASTS[contrast].items())
    point = primary_from_counts(stats.sum(axis=0))
    results = []
    for j,contrast in enumerate(PRIMARY_CONTRASTS):
        for t in TASKS:
            value = sum(weight*point[t][positions[c]] for c,weight in CONTRASTS[contrast].items())
            lo,hi = np.quantile(draws[t][:,j],[.025,.975])
            results.append({"task":t,"metric":CLASSIFICATION_POLICY["primary_metrics"][t],
                "contrast":contrast,"stratum":"all","n":n,"difference":float(value),
                "descriptive_ci95":[float(lo),float(hi)]})
    return results


def analyze_blocks(blocks, *, frame, gold_by_query, epsilon,
                   expected_query_count=643, bootstrap_replicates=10000, bootstrap_seed=42) -> dict[str, Any]:
    if len(frame) != expected_query_count or not frame:
        raise ValueError("full registered query frame required")
    qids = [r["query_id"] for r in frame]
    if (len(set(qids)) != len(qids) or set(qids) != set(gold_by_query)
            or any(not isinstance(q,str) for q in qids)
            or any(type(r.get("lex_hit")) is not bool for r in frame)
            or not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError("invalid frame, Gold, or epsilon")
    for gold in gold_by_query.values():
        for task in TASKS:
            _gold_ordinal(task,gold[task])
    indexed, predictions, rankings = {}, {}, {}
    expected = {(q,t,c) for q in qids for t in TASKS for c in CONDITIONS}
    for b in blocks:
        key = (b["query_id"],b["task"],b["condition"])
        if key not in expected or key in indexed:
            raise ValueError("duplicated or unexpected block")
        q,t,c = key
        indexed[key] = block_analysis(t,b["candidates"],gold=gold_by_query[q][t])
        ordered = sorted(b["candidates"],key=lambda x:(-x["scores"]["answer_sum"],x["ordinal"]))
        best = ordered[0]
        gap = best["scores"]["answer_sum"]-ordered[1]["scores"]["answer_sum"]
        predictions[key] = {"query_id":q,"gold":gold_by_query[q][t],
                            "prediction":best["canonical_answer"]}
        rankings[key] = {"labels":best["labels"],"ordinal":best["ordinal"],
                         "top_score_gap":gap,"tied_top_count":sum(
                             x["scores"]["answer_sum"]==best["scores"]["answer_sum"] for x in ordered),
                         "within_two_epsilon":gap <= 2*epsilon}
    if set(indexed) != expected:
        raise ValueError("incomplete paired matrix")
    strata = {"all":qids,
        "Lq_hit":[r["query_id"] for r in frame if r["lex_hit"]],
        "Lq_no_hit":[r["query_id"] for r in frame if not r["lex_hit"]],
        **{name:[q for q in qids if pred(len(gold_by_query[q]["group"]))]
           for name,pred in (("gold_size_0",lambda n:n==0),("gold_size_1",lambda n:n==1),
                             ("gold_size_2plus",lambda n:n>=2))}}
    classification = []
    for c in CONDITIONS:
        for stratum,subset in strata.items():
            if not subset:
                classification.append({"condition":c,"stratum":stratum,"n":0,"status":"undefined"})
                continue
            result = {t:evaluate_predictions(t,[predictions[q,t,c] for q in subset]) for t in TASKS}
            joint = sum(all(result[t]["records"][i]["correct"] for t in TASKS) for i in range(len(subset)))
            classification.append({"condition":c,"stratum":stratum,"n":len(subset),
                "metrics":{t:result[t]["metrics"] for t in TASKS},
                "joint_correct":joint,"joint_exact_match":joint/len(subset),
                "top_ties":{t:sum(rankings[q,t,c]["tied_top_count"]>1 for q in subset) for t in TASKS},
                "near_ties_two_epsilon":{t:sum(rankings[q,t,c]["within_two_epsilon"] for q in subset) for t in TASKS}})
    stats = np.zeros((len(qids),len(CONDITIONS),7,3),dtype=np.int64)
    for i,q in enumerate(qids):
        for j,c in enumerate(CONDITIONS):
            g = [gold_by_query[q]["hate"]==l for l in ("hate","non-hate")] + [l in gold_by_query[q]["group"] for l in GROUP_LABELS]
            p = [l in rankings[q,"hate",c]["labels"] for l in ("hate","non-hate")] + [l in rankings[q,"group",c]["labels"] for l in GROUP_LABELS]
            for k,(truth,pred) in enumerate(zip(g,p,strict=True)):
                stats[i,j,k] = [truth and pred,not truth and pred,truth and not pred]
    point = primary_from_counts(stats.sum(axis=0))
    for j,c in enumerate(CONDITIONS):
        reference = next(r for r in classification if r["condition"]==c and r["stratum"]=="all")
        for t in TASKS:
            if abs(point[t][j]-reference["metrics"][t][CLASSIFICATION_POLICY["primary_metrics"][t]]) > 1e-12:
                raise ValueError("bootstrap sufficient statistics disagree with project scorer")
    primary = paired_f1_bootstrap(stats,repetitions=bootstrap_replicates,seed=bootstrap_seed)
    condition_summaries, contrast_summaries = [], []
    # Preserve all original Gold, cardinality, EOS and length-sensitivity readouts.
    for t in TASKS:
        keys = tuple(indexed[qids[0],t,"C0"])
        for metric in keys:
            for c in CONDITIONS:
                for stratum,subset in strata.items():
                    condition_summaries.append({"task":t,"metric":metric,"condition":c,"stratum":stratum,
                        **_summary([indexed[q,t,c][metric] for q in subset])})
            for contrast,weights in CONTRASTS.items():
                values = {}
                for q in qids:
                    xs = [indexed[q,t,c][metric] for c in weights]
                    if all(x is None for x in xs):
                        values[q] = None
                    elif any(x is None for x in xs):
                        raise ValueError("metric undefined in only one paired condition")
                    else:
                        values[q] = math.fsum(w*indexed[q,t,c][metric] for c,w in weights.items())
                for stratum,subset in strata.items():
                    contrast_summaries.append({"task":t,"metric":metric,"contrast":contrast,"stratum":stratum,
                        **_summary([values[q] for q in subset])})
    return {"schema_version":"general-model-nolabel-scientific-analysis/v1",
        "query_count":len(qids),"block_count":len(indexed),"conditions":list(CONDITIONS),
        "classification_policy":CLASSIFICATION_POLICY,
        "bootstrap":{"repetitions":bootstrap_replicates,"seed":bootstrap_seed,
                     "shared_query_draws":True,"classification_primary_only":True},
        "classification":classification,"primary_classification_differences":primary,
        "condition_summaries":condition_summaries,"contrast_summaries":contrast_summaries,
        "stratum_counts":{k:len(v) for k,v in strata.items()},"epsilon":epsilon,
        "per_query":[{"query_id":q,"lex_hit":frame[i]["lex_hit"],"gold":gold_by_query[q],
            "conditions":{c:{t:{"prediction":rankings[q,t,c],"readouts":indexed[q,t,c]} for t in TASKS}
                          for c in CONDITIONS}} for i,q in enumerate(qids)],
        "interpretation":{"primary":"descriptive-dev-classification-differences",
            "category_removal":"explicit-field-removal-also-shortens-input",
            "definitions_may_reveal_categories":True,"demo_answers_preserved":True,
            "free_generation":False,"test_access":False,"internal_mechanism_identified":False}}
