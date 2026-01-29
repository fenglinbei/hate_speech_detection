#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Long-tail breakdown for quadruple extraction results.

Input: result JSONs in the format like yours:
{
  "info": {...},
  "results": [
    {
      "id": ...,
      "gt_quadruples": [{"target":..., "argument":..., "targeted_group":..., "hateful":...}, ...],
      "pred_quadruples": [...],
      ...
    }, ...
  ],
  "metric": {...}  # optional, not used
}

Outputs:
- per_class_metrics.csv
- bucket_metrics.csv
- (optional) per_class_stability.csv (if multiple seeds/runs per method)
"""

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# -----------------------------
# Utilities
# -----------------------------

def read_json_or_jsonl(path: str) -> Any:
    if path.endswith(".jsonl"):
        arr = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                arr.append(json.loads(line))
        return arr
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def infer_method_name(path: str) -> str:
    # e.g. ours_prompt_al1280_qwen3.json -> ours
    base = os.path.basename(path)
    m = re.split(r"[._\-]+", base)
    return m[0] if m and m[0] else "method"

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def f1(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # 你也可以按需更激进：去空格、全角半角等
    return str(s).strip()

@dataclass(frozen=True)
class Quad:
    target: str
    argument: str
    targeted_group: str
    hateful: str

    def as_tuple(self):
        return (self.target, self.argument, self.targeted_group, self.hateful)

def normalize_quad(q: Dict[str, Any]) -> Quad:
    return Quad(
        target=normalize_text(q.get("target", "")),
        argument=normalize_text(q.get("argument", "")),
        targeted_group=normalize_text(q.get("targeted_group", "")),
        hateful=normalize_text(q.get("hateful", "")),
    )

def get_label(quad: Quad, label_mode: str) -> Optional[str]:
    """
    label_mode:
      - targeted_group: 只按 targeted_group 分解（通常仅关心 hateful==hate）
      - combined: hate -> targeted_group, non-hate -> non-hate（把非仇恨作为一类）
      - hateful: 仅按 hateful 二分类（不推荐做长尾）
    """
    if label_mode == "hateful":
        return quad.hateful or "unknown"
    if label_mode == "combined":
        if quad.hateful.lower() == "hate":
            return quad.targeted_group or "unknown"
        return "non-hate"
    # targeted_group
    return quad.targeted_group or "unknown"

def filter_quads(quads: List[Quad], label_mode: str, hate_only: bool) -> List[Quad]:
    if not hate_only:
        return quads
    # hate_only 时，过滤掉 hateful != hate
    if label_mode in ("targeted_group", "combined"):
        return [q for q in quads if (q.hateful or "").lower() == "hate"]
    return quads

# -----------------------------
# Similarity backends for SOFT matching
# -----------------------------

class Similarity:
    def sim(self, a: str, b: str) -> float:
        raise NotImplementedError

class RatioSimilarity(Similarity):
    def sim(self, a: str, b: str) -> float:
        # 简单字符相似度（无依赖）
        # 也可以换成 edit distance / token-jaccard
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()

class EmbeddingSimilarity(Similarity):
    def __init__(self, model_name_or_path: str, batch_size: int = 128):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except Exception as e:
            raise RuntimeError(
                "需要 sentence-transformers + numpy。请先安装：pip install sentence-transformers numpy"
            ) from e
        self.np = __import__("numpy")
        self.model = SentenceTransformer(model_name_or_path)
        self.batch_size = batch_size
        self.cache: Dict[str, Any] = {}

    def _encode(self, texts: List[str]):
        # sentence-transformers 输出 numpy
        import numpy as np
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化后余弦=点积
        )
        return embs

    def sim(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        # 缓存单句 embedding
        if a not in self.cache:
            self.cache[a] = self._encode([a])[0]
        if b not in self.cache:
            self.cache[b] = self._encode([b])[0]
        va = self.cache[a]
        vb = self.cache[b]
        # 归一化后用点积即可
        return float(self.np.dot(va, vb))

# -----------------------------
# Matching
# -----------------------------

def hard_match_count(pred: List[Quad], gold: List[Quad]) -> int:
    """
    多重集 strict match: 完全一致的四元组一一匹配
    """
    cp = Counter([q.as_tuple() for q in pred])
    cg = Counter([q.as_tuple() for q in gold])
    tp = 0
    for k, v in cp.items():
        tp += min(v, cg.get(k, 0))
    return tp

def soft_match_count(
    pred: List[Quad],
    gold: List[Quad],
    sim_backend: Similarity,
    gamma: float = 0.5,
) -> int:
    """
    在同一 label 内做 soft matching（由外层按 label 分组保证）
    soft 条件：sim(target) >= gamma AND sim(argument) >= gamma
    使用贪心最大匹配（按平均相似度从高到低）
    """
    if not pred or not gold:
        return 0

    # 构造候选边
    candidates: List[Tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            st = sim_backend.sim(p.target, g.target)
            sa = sim_backend.sim(p.argument, g.argument)
            if st >= gamma and sa >= gamma:
                score = (st + sa) / 2.0
                candidates.append((score, i, j))

    if not candidates:
        return 0

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_p = set()
    used_g = set()
    tp = 0
    for score, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return tp

# -----------------------------
# Core evaluation per-class
# -----------------------------

@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

def update_counts(counts: Counts, tp: int, fp: int, fn: int):
    counts.tp += tp
    counts.fp += fp
    counts.fn += fn

def counts_to_metrics(c: Counts) -> Dict[str, float]:
    p = safe_div(c.tp, c.tp + c.fp)
    r = safe_div(c.tp, c.tp + c.fn)
    return {"p": p, "r": r, "f1": f1(p, r), "tp": c.tp, "fp": c.fp, "fn": c.fn}

def evaluate_one_run(
    run_json_path: str,
    label_mode: str,
    hate_only: bool,
    sim_backend: Similarity,
    gamma: float,
) -> Dict[str, Any]:
    d = read_json_or_jsonl(run_json_path)
    if isinstance(d, dict) and "results" in d:
        results = d["results"]
        info = d.get("info", {}) or {}
    elif isinstance(d, list):
        results = d
        info = {}
    else:
        raise ValueError(f"Unrecognized format: {run_json_path}")

    method = infer_method_name(run_json_path)
    seed = None
    try:
        seed = info.get("seed", None)
    except Exception:
        seed = None

    hard_counts_by_class: Dict[str, Counts] = defaultdict(Counts)
    soft_counts_by_class: Dict[str, Counts] = defaultdict(Counts)

    for item in results:
        gt = [normalize_quad(q) for q in item.get("gt_quadruples", [])]
        pr = [normalize_quad(q) for q in item.get("pred_quadruples", [])]

        gt = filter_quads(gt, label_mode=label_mode, hate_only=hate_only)
        pr = filter_quads(pr, label_mode=label_mode, hate_only=hate_only)

        # 按 label 分组
        gt_by = defaultdict(list)
        pr_by = defaultdict(list)

        for q in gt:
            lab = get_label(q, label_mode)
            if lab is not None:
                gt_by[lab].append(q)
        for q in pr:
            lab = get_label(q, label_mode)
            if lab is not None:
                pr_by[lab].append(q)

        labels = set(gt_by.keys()) | set(pr_by.keys())
        for lab in labels:
            g_list = gt_by.get(lab, [])
            p_list = pr_by.get(lab, [])

            # hard
            tp_h = hard_match_count(p_list, g_list)
            fp_h = len(p_list) - tp_h
            fn_h = len(g_list) - tp_h
            update_counts(hard_counts_by_class[lab], tp_h, fp_h, fn_h)

            # soft
            tp_s = soft_match_count(p_list, g_list, sim_backend, gamma=gamma)
            fp_s = len(p_list) - tp_s
            fn_s = len(g_list) - tp_s
            update_counts(soft_counts_by_class[lab], tp_s, fp_s, fn_s)

    return {
        "method": method,
        "seed": seed,
        "hard_counts_by_class": hard_counts_by_class,
        "soft_counts_by_class": soft_counts_by_class,
    }

# -----------------------------
# Frequency / Bucketing (Head/Med/Tail)
# -----------------------------

def extract_quads_from_dataset_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 兼容多种字段名
    for k in ["quadruples", "gt_quadruples", "labels", "label", "annotation", "annotations"]:
        if k in item and isinstance(item[k], list):
            # 只有 quadruples / gt_quadruples 一般是 list[dict]
            if item[k] and isinstance(item[k][0], dict) and "target" in item[k][0]:
                return item[k]
    return []

def compute_train_freq(
    train_path: str,
    label_mode: str,
    hate_only: bool,
) -> Dict[str, int]:
    data = read_json_or_jsonl(train_path)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Train file should be list/jsonl: {train_path}")

    freq = Counter()
    for item in data:
        quads = extract_quads_from_dataset_item(item)
        if not quads:
            # 如果你的训练集结构不同，可以在这里加解析规则
            continue
        for q in quads:
            qq = normalize_quad(q)
            qqs = filter_quads([qq], label_mode=label_mode, hate_only=hate_only)
            if not qqs:
                continue
            lab = get_label(qqs[0], label_mode)
            if lab is not None:
                freq[lab] += 1
    return dict(freq)

def make_buckets(
    freq: Dict[str, int],
    bucket_method: str = "percentile",
    head_pct: float = 0.3,
    tail_pct: float = 0.3,
    tail_max_freq: int = -1,
) -> Dict[str, str]:
    """
    Return: class -> bucket in {"head","medium","tail"}
    Methods:
      - percentile: head=top head_pct classes, tail=bottom tail_pct classes
      - threshold: tail if freq <= tail_max_freq, head otherwise (medium not used much)
    """
    classes = sorted(freq.keys(), key=lambda c: (freq[c], c))
    n = len(classes)
    if n == 0:
        return {}

    buckets = {}

    if bucket_method == "threshold":
        if tail_max_freq <= 0:
            # 默认：尾类取 <= 中位数
            sorted_freq = sorted(freq.values())
            tail_max_freq = sorted_freq[n // 2]
        for c in classes:
            buckets[c] = "tail" if freq[c] <= tail_max_freq else "head"
        return buckets

    # percentile default
    tail_n = max(1, int(math.ceil(n * tail_pct)))
    head_n = max(1, int(math.ceil(n * head_pct)))

    tail_set = set(classes[:tail_n])
    head_set = set(classes[-head_n:])

    for c in classes:
        if c in tail_set and c in head_set:
            buckets[c] = "medium"
        elif c in tail_set:
            buckets[c] = "tail"
        elif c in head_set:
            buckets[c] = "head"
        else:
            buckets[c] = "medium"

    return buckets

# -----------------------------
# Aggregation / Output
# -----------------------------

def summarize_per_class(run_eval: Dict[str, Any]) -> List[Dict[str, Any]]:
    method = run_eval["method"]
    seed = run_eval["seed"]
    hard = run_eval["hard_counts_by_class"]
    soft = run_eval["soft_counts_by_class"]

    rows = []
    all_classes = set(hard.keys()) | set(soft.keys())
    for c in sorted(all_classes):
        hm = counts_to_metrics(hard.get(c, Counts()))
        sm = counts_to_metrics(soft.get(c, Counts()))
        rows.append({
            "method": method,
            "seed": seed,
            "class": c,
            "hard_p": hm["p"], "hard_r": hm["r"], "hard_f1": hm["f1"],
            "soft_p": sm["p"], "soft_r": sm["r"], "soft_f1": sm["f1"],
            "avg_f1": (hm["f1"] + sm["f1"]) / 2.0,
            "hard_tp": hm["tp"], "hard_fp": hm["fp"], "hard_fn": hm["fn"],
            "soft_tp": sm["tp"], "soft_fp": sm["fp"], "soft_fn": sm["fn"],
        })
    return rows

def macro_over_classes(rows: List[Dict[str, Any]], classes: List[str]) -> Dict[str, float]:
    if not classes:
        return {"macro_hard_f1": 0.0, "macro_soft_f1": 0.0, "macro_avg_f1": 0.0,
                "macro_hard_r": 0.0, "macro_soft_r": 0.0}
    sel = [r for r in rows if r["class"] in classes]
    if not sel:
        return {"macro_hard_f1": 0.0, "macro_soft_f1": 0.0, "macro_avg_f1": 0.0,
                "macro_hard_r": 0.0, "macro_soft_r": 0.0}
    return {
        "macro_hard_f1": sum(r["hard_f1"] for r in sel) / len(sel),
        "macro_soft_f1": sum(r["soft_f1"] for r in sel) / len(sel),
        "macro_avg_f1":  sum(r["avg_f1"]  for r in sel) / len(sel),
        "macro_hard_r":  sum(r["hard_r"]  for r in sel) / len(sel),
        "macro_soft_r":  sum(r["soft_r"]  for r in sel) / len(sel),
    }

def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Result JSON paths or glob patterns, e.g. outputs/ours_*.json outputs/topk_*.json")
    ap.add_argument("--out_dir", default="analysis_long_tail", help="Output directory")
    ap.add_argument("--train_path", default="", help="Optional: training set path to compute class frequency for bucketing")
    ap.add_argument("--label_mode", default="targeted_group",
                    choices=["targeted_group", "combined", "hateful"],
                    help="How to define class label for breakdown")
    ap.add_argument("--hate_only", action="store_true",
                    help="If set, only keep hateful==hate (recommended for targeted_group long-tail)")
    ap.add_argument("--gamma", type=float, default=0.5, help="Soft match threshold")
    ap.add_argument("--sim", default="ratio", choices=["ratio", "embedding"], help="Soft similarity backend")
    ap.add_argument("--embed_model", default="BAAI/bge-large-zh-v1.5",
                    help="SentenceTransformer model name/path (only used when --sim embedding)")
    ap.add_argument("--bucket_method", default="percentile", choices=["percentile", "threshold"])
    ap.add_argument("--head_pct", type=float, default=0.3)
    ap.add_argument("--tail_pct", type=float, default=0.3)
    ap.add_argument("--tail_max_freq", type=int, default=-1)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # Expand globs
    paths = []
    for p in args.inputs:
        g = glob.glob(p)
        if g:
            paths.extend(g)
        else:
            paths.append(p)
    paths = sorted(list(dict.fromkeys(paths)))
    if not paths:
        raise SystemExit("No input files found.")

    # Build similarity backend
    if args.sim == "embedding":
        sim_backend = EmbeddingSimilarity(args.embed_model)
    else:
        sim_backend = RatioSimilarity()

    # Evaluate each run
    run_rows_all = []
    run_rows_by_method_seed = defaultdict(list)

    for rp in paths:
        run_eval = evaluate_one_run(
            rp,
            label_mode=args.label_mode,
            hate_only=args.hate_only,
            sim_backend=sim_backend,
            gamma=args.gamma,
        )
        rows = summarize_per_class(run_eval)
        run_rows_all.extend(rows)
        key = (run_eval["method"], run_eval["seed"])
        run_rows_by_method_seed[key] = rows

    # Attach frequency and bucket (if train provided)
    freq = {}
    buckets = {}
    if args.train_path:
        freq = compute_train_freq(args.train_path, label_mode=args.label_mode, hate_only=args.hate_only)
        buckets = make_buckets(
            freq, bucket_method=args.bucket_method,
            head_pct=args.head_pct, tail_pct=args.tail_pct,
            tail_max_freq=args.tail_max_freq
        )

    for r in run_rows_all:
        c = r["class"]
        r["freq_train"] = freq.get(c, 0) if freq else 0
        r["bucket"] = buckets.get(c, "unknown") if buckets else "unknown"

    # Save per-class metrics (run-level)
    per_class_path = os.path.join(args.out_dir, "per_class_metrics.csv")
    write_csv(per_class_path, run_rows_all)

    # Bucket summary per run (macro)
    bucket_rows = []
    for (method, seed), rows in run_rows_by_method_seed.items():
        # decide class sets from this run
        classes_in_run = sorted({r["class"] for r in rows})
        if buckets:
            head = [c for c in classes_in_run if buckets.get(c) == "head"]
            mid  = [c for c in classes_in_run if buckets.get(c) == "medium"]
            tail = [c for c in classes_in_run if buckets.get(c) == "tail"]
        else:
            # if no bucketing info, treat all as one bucket
            head, mid, tail = [], [], []

        if buckets:
            for bname, clz in [("head", head), ("medium", mid), ("tail", tail)]:
                m = macro_over_classes(rows, clz)
                bucket_rows.append({
                    "method": method, "seed": seed, "bucket": bname,
                    **m, "num_classes": len(clz)
                })
        else:
            m = macro_over_classes(rows, classes_in_run)
            bucket_rows.append({
                "method": method, "seed": seed, "bucket": "all",
                **m, "num_classes": len(classes_in_run)
            })

    bucket_path = os.path.join(args.out_dir, "bucket_metrics.csv")
    write_csv(bucket_path, bucket_rows)

    # Stability (if multiple runs per method)
    # mean/std over seeds for each (method, class)
    stability_rows = []
    by_method_class = defaultdict(list)
    for r in run_rows_all:
        by_method_class[(r["method"], r["class"])].append(r)

    for (method, c), items in by_method_class.items():
        if len(items) <= 1:
            continue
        def mean_std(xs):
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
            return mu, math.sqrt(var)

        hard_r_mu, hard_r_sd = mean_std([it["hard_r"] for it in items])
        soft_r_mu, soft_r_sd = mean_std([it["soft_r"] for it in items])
        avg_mu, avg_sd       = mean_std([it["avg_f1"] for it in items])

        stability_rows.append({
            "method": method,
            "class": c,
            "n_runs": len(items),
            "freq_train": items[0].get("freq_train", 0),
            "bucket": items[0].get("bucket", "unknown"),
            "hard_r_mean": hard_r_mu, "hard_r_std": hard_r_sd,
            "soft_r_mean": soft_r_mu, "soft_r_std": soft_r_sd,
            "avg_f1_mean": avg_mu, "avg_f1_std": avg_sd,
            "hard_r_cv": safe_div(hard_r_sd, hard_r_mu),
            "soft_r_cv": safe_div(soft_r_sd, soft_r_mu),
        })

    if stability_rows:
        stability_path = os.path.join(args.out_dir, "per_class_stability.csv")
        write_csv(stability_path, stability_rows)

    print(f"[OK] Saved: {per_class_path}")
    print(f"[OK] Saved: {bucket_path}")
    if stability_rows:
        print(f"[OK] Saved: {stability_path}")
    else:
        print("[INFO] Only one run per class; stability file not generated.")

if __name__ == "__main__":
    main()
