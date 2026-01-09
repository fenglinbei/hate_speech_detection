
from __future__ import annotations

import math
import random
import numpy as np
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.log import logger
from rag.tools import sha1_text


import numpy as np

def eligibility_filter(
    query_id: str,
    query_text: str,
    query_label: Optional[str],
    cand_ids: List[str],
    cand_sims: np.ndarray,
    doc_store: Dict[str, Dict[str, Any]],
    sim_min: float,
    sim_max: float,
    label_policy: str,
    allow_cross_label_ratio: float,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns filtered candidate list with reasons tracked.
    """
    filtered = []
    trace = []

    logger.debug(f"cand_id sample={cand_ids[0]} type={type(cand_ids[0])}")
    k = next(iter(doc_store.keys()))
    logger.debug(f"doc_store key sample={k} type={type(k)}")
    logger.debug(f"cand_id in doc_store? {cand_ids[0] in doc_store}")

    # dedup by text hash (avoid near duplicates by exact match first)
    seen_text_hash = set()

    for doc_id, sim in zip(cand_ids, cand_sims.tolist()):
        logger.debug(f"Evaluating candidate doc_id={doc_id} with sim={sim:.4f}")
        if doc_id == query_id:
            trace.append({"doc_id": doc_id, "reason": "self"})
            continue
        doc = doc_store.get(doc_id)
        if doc is None:
            trace.append({"doc_id": doc_id, "reason": "missing_doc"})
            continue

        text = doc["text"]
        th = sha1_text(text)
        if th in seen_text_hash:
            trace.append({"doc_id": doc_id, "reason": "dup_text"})
            continue

        if sim < sim_min:
            trace.append({"doc_id": doc_id, "reason": "sim_too_low", "sim": sim})
            continue
        if sim > sim_max:
            trace.append({"doc_id": doc_id, "reason": "sim_too_high", "sim": sim})
            continue

        # label constraints
        if label_policy == "strict" and query_label is not None:
            if doc.get("label") != query_label:
                trace.append({"doc_id": doc_id, "reason": "label_mismatch_strict", "sim": sim})
                continue

        seen_text_hash.add(th)
        filtered.append({
            "doc_id": doc_id,
            "text": text,
            "label": doc.get("label"),
            "sim": float(sim),
            "meta": doc.get("meta", {}),
        })

    # soft policy cross-label control is applied later in selection stage
    # (we still keep all filtered candidates here)
    return [c["doc_id"] for c in filtered], trace


def mmr_select(
    query_vec: np.ndarray,
    candidates: List[Dict[str, Any]],
    cand_vecs: np.ndarray,
    n_shot: int,
    lambda_mmr: float,
    allow_cross_label_ratio: float,
    query_label: Optional[str],
    seed: int,
) -> List[int]:
    """
    Returns indices of selected candidates.

    candidates: list of dict with fields sim (relevance proxy), label, ...
    cand_vecs: (C, dim) L2-normalized
    query_vec: (dim,) L2-normalized
    """
    rng = random.Random(seed)

    C = len(candidates)
    if C == 0:
        return []

    # relevance: default use candidates[i]["sim"] (cosine)
    rel = np.array([c["sim"] for c in candidates], dtype=np.float32)

    selected = []
    selected_set = set()

    # track label balance
    cross_label_budget = int(math.floor(n_shot * allow_cross_label_ratio))
    cross_label_used = 0

    # precompute candidate-candidate similarity matrix lazily for speed
    # We'll compute on demand using dot products.
    def max_sim_to_selected(i: int) -> float:
        if not selected:
            return 0.0
        v = cand_vecs[i]
        sims = cand_vecs[selected] @ v  # (|S|,)
        return float(np.max(sims))

    # first pick: highest relevance with tie-breaking random among top few
    top_pool = np.argsort(-rel)[: min(5, C)]
    first = int(rng.choice(list(top_pool)))
    selected.append(first)
    selected_set.add(first)

    # update cross-label usage
    if query_label is not None and candidates[first].get("label") != query_label:
        cross_label_used += 1

    while len(selected) < min(n_shot, C):
        best_score = -1e9
        best_i = None

        for i in range(C):
            if i in selected_set:
                continue

            # cross-label control (soft)
            if query_label is not None:
                is_cross = (candidates[i].get("label") != query_label)
                if is_cross and cross_label_used >= cross_label_budget:
                    continue

            diversity_penalty = max_sim_to_selected(i)
            score = lambda_mmr * float(rel[i]) - (1.0 - lambda_mmr) * float(diversity_penalty)

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        selected.append(best_i)
        selected_set.add(best_i)

        if query_label is not None and candidates[best_i].get("label") != query_label:
            cross_label_used += 1

    return selected

def filter_with_fallback(
    *,
    query_id: str,
    query_label: Optional[Any],
    cand_ids: List[str],
    cand_sims: np.ndarray,
    doc_store: Dict[str, Dict[str, Any]],
    sim_min: float,
    sim_max: float,
    n_shot: int,
    label_policy: str = "soft",               # "strict" | "soft" | "none"
    allow_cross_label_ratio: float = 0.15,    # used in soft mode; strict is handled here, too
    # optional: allow increasing top_k and re-search
    search_more_fn: Optional[Callable[[int], Tuple[List[str], np.ndarray]]] = None,
    top_k_stages: Tuple[int, ...] = (0,),     # e.g. (0, 300, 500). 0 => do not re-search
    # dynamic threshold
    quantile_q: float = 0.70,                 # p70 for dynamic sim_min
    global_sim_floor: float = 0.10,           # never go below this unless final emergency stage
    relax_sim_max_to: float = 0.98,           # relaxed upper bound
    keep_at_least: int = 1,                   # ensure at least this many candidates for downstream
    debug_keep_reasons_sample: int = 50,      # how many per-candidate traces to keep
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Multi-stage eligibility filtering with fallback.

    Returns:
      kept_ids: list of candidate ids (guaranteed len >= keep_at_least if possible)
      trace: dict with reason_counts + sample details
      fb_meta: dict describing which stage and thresholds were used

    Stages:
      A: strict [sim_min, sim_max] (+ strict label if enabled)
      B: dynamic lower bound using quantile (min(sim_min, p_q), floored), keep sim_max
      C: drop lower bound; only enforce relaxed sim_max (<= relax_sim_max_to)
      D: "anything but self/missing_doc" (last resort)
      E: if still empty: return empty (should be rare; retrieve_shots will handle)
    """
    if cand_sims is None or len(cand_ids) == 0:
        trace = {"reason_counts": {"empty_retrieval": 1}, "samples": []}
        fb_meta = {"stage": "E", "note": "no candidates from search"}
        return [], trace, fb_meta

    if not isinstance(cand_sims, np.ndarray):
        cand_sims = np.array(cand_sims, dtype=np.float32)
    else:
        cand_sims = cand_sims.astype(np.float32, copy=False)

    # Helper: build sim map once
    sim_map: Dict[str, float] = {cid: float(s) for cid, s in zip(cand_ids, cand_sims.tolist())}

    # Helper: label check
    def label_ok(doc_label: Any) -> bool:
        if label_policy == "none" or query_label is None:
            return True
        if label_policy == "strict":
            # support doc_label being list/str
            if isinstance(doc_label, (list, tuple, set)):
                return query_label in doc_label
            return doc_label == query_label
        # soft: allow both; ratio is controlled later in MMR stage (not here)
        return True

    # Helper: run one pass filter with configurable bounds
    def run_pass(
        *,
        use_min: bool,
        use_max: bool,
        _min: float,
        _max: float,
        stage_name: str,
    ) -> Tuple[List[str], Counter, List[Dict[str, Any]]]:
        counts = Counter()
        samples: List[Dict[str, Any]] = []
        kept: List[str] = []

        for cid in cand_ids:
            s = sim_map.get(cid, -1.0)

            # record a small sample of evaluations for debugging
            if len(samples) < debug_keep_reasons_sample:
                samples.append({"doc_id": cid, "sim": s, "stage": stage_name})

            if cid == query_id:
                counts["self"] += 1
                continue

            doc = doc_store.get(cid)
            if doc is None:
                counts["missing_doc"] += 1
                continue

            # label strict check here
            if label_policy == "strict":
                if not label_ok(doc.get("label")):
                    counts["label_mismatch_strict"] += 1
                    continue

            if use_min and s < _min:
                counts["sim_too_low"] += 1
                continue
            if use_max and s > _max:
                counts["sim_too_high"] += 1
                continue

            kept.append(cid)
            counts["kept"] += 1

        return kept, counts, samples

    # Optionally allow re-search with larger topK
    # We unify all candidate lists by taking latest search results
    stage_searches = list(top_k_stages) if top_k_stages else [0]
    if stage_searches and stage_searches[0] == 0:
        # do nothing, use provided cand_ids/sims
        pass
    elif search_more_fn is not None and stage_searches:
        # If user passes (120, 300, 500) we consider first stage already done in caller.
        # Here we only re-search for larger K if requested later.
        pass

    # Compute quantile for dynamic min (based on current sims)
    sims_only = np.array([sim_map[cid] for cid in cand_ids if cid in sim_map], dtype=np.float32)
    p_q = float(np.quantile(sims_only, quantile_q)) if sims_only.size > 0 else sim_min
    dyn_min = max(global_sim_floor, min(sim_min, p_q))

    # Stage A
    kept_A, cnt_A, samp_A = run_pass(use_min=True, use_max=True, _min=sim_min, _max=sim_max, stage_name="A")
    if len(kept_A) >= max(keep_at_least, 1):
        trace = {"reason_counts": dict(cnt_A), "samples": samp_A}
        fb_meta = {"stage": "A", "used_sim_min": sim_min, "used_sim_max": sim_max, "p_q": p_q, "dyn_min": dyn_min}
        return kept_A, trace, fb_meta

    # Stage B (dynamic lower bound)
    kept_B, cnt_B, samp_B = run_pass(use_min=True, use_max=True, _min=dyn_min, _max=sim_max, stage_name="B")
    if len(kept_B) >= max(keep_at_least, 1):
        trace = {"reason_counts": dict(cnt_B), "samples": samp_B}
        fb_meta = {"stage": "B", "used_sim_min": dyn_min, "used_sim_max": sim_max, "p_q": p_q, "dyn_min": dyn_min}
        return kept_B, trace, fb_meta

    # Stage C (drop lower bound, relax upper bound)
    kept_C, cnt_C, samp_C = run_pass(use_min=False, use_max=True, _min=dyn_min, _max=relax_sim_max_to, stage_name="C")
    if len(kept_C) >= max(keep_at_least, 1):
        trace = {"reason_counts": dict(cnt_C), "samples": samp_C}
        fb_meta = {"stage": "C", "used_sim_min": None, "used_sim_max": relax_sim_max_to, "p_q": p_q, "dyn_min": dyn_min}
        return kept_C, trace, fb_meta

    # Stage D (last resort): keep anything that exists in doc_store and not self
    cnt_D = Counter()
    kept_D = []
    samples_D = []
    for cid in cand_ids:
        s = sim_map.get(cid, -1.0)
        if len(samples_D) < debug_keep_reasons_sample:
            samples_D.append({"doc_id": cid, "sim": s, "stage": "D"})
        if cid == query_id:
            cnt_D["self"] += 1
            continue
        if cid not in doc_store:
            cnt_D["missing_doc"] += 1
            continue
        if label_policy == "strict":
            if not label_ok(doc_store[cid].get("label")):
                cnt_D["label_mismatch_strict"] += 1
                continue
        kept_D.append(cid)
        cnt_D["kept"] += 1

    if len(kept_D) >= max(keep_at_least, 1):
        trace = {"reason_counts": dict(cnt_D), "samples": samples_D}
        fb_meta = {"stage": "D", "used_sim_min": None, "used_sim_max": None, "p_q": p_q, "dyn_min": dyn_min}
        return kept_D, trace, fb_meta

    # Stage E: still empty
    trace = {"reason_counts": {"empty_after_fallback": 1, **dict(cnt_A)}, "samples": samp_A}
    fb_meta = {"stage": "E", "used_sim_min": None, "used_sim_max": None, "p_q": p_q, "dyn_min": dyn_min}
    return [], trace, fb_meta