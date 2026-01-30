#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# -------------------------
# Helpers
# -------------------------

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def infer_method_name(path: str) -> str:
    # e.g. ours_prompt_al1280_qwen3.json -> ours
    base = os.path.basename(path)
    tok = re.split(r"[._\-]+", base)
    return tok[0] if tok and tok[0] else "method"

def split_labels(s: str) -> List[str]:
    """
    支持: "Racism, Sexism\\LGBTQ/Region" 等
    分隔符：逗号/中文逗号/斜杠/反斜杠/竖线/分号
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = re.split(r"[,\，/\\|;；]+", s)
    return [p.strip() for p in parts if p.strip()]

def norm_text(x) -> str:
    return "" if x is None else str(x).strip()

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0

def f1(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0


@dataclass(frozen=True)
class Quad:
    target: str
    argument: str
    targeted_group: str
    hateful: str

    def key(self) -> Tuple[str, str, str, str]:
        return (self.target, self.argument, self.targeted_group, self.hateful)


def normalize_quad(q: dict) -> Quad:
    return Quad(
        target=norm_text(q.get("target")),
        argument=norm_text(q.get("argument")),
        targeted_group=norm_text(q.get("targeted_group")),
        hateful=norm_text(q.get("hateful")),
    )


def expand_by_group(q: Quad) -> List[Quad]:
    """
    多类别 targeted_group 展开成多个单类别 quad
    """
    labs = split_labels(q.targeted_group)
    if not labs:
        labs = ["unknown"]
    return [Quad(q.target, q.argument, lab, q.hateful) for lab in labs]


def hard_match_tp(pred: List[Quad], gold: List[Quad]) -> int:
    """
    strict multiset match within one sample & one class
    """
    cp = Counter([x.key() for x in pred])
    cg = Counter([x.key() for x in gold])
    tp = 0
    for k, v in cp.items():
        tp += min(v, cg.get(k, 0))
    return tp


# -------------------------
# Evaluation (per-class)
# -------------------------

def eval_one_file(
    path: str,
    classes: List[str],
    hate_only: bool,
) -> Dict[str, Dict[str, float]]:
    """
    Return: class -> {p, r, f1, tp, fp, fn}
    """
    d = read_json(path)
    results = d.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Bad format: {path}")

    # counts per class
    tp = Counter()
    fp = Counter()
    fn = Counter()

    class_set = set(classes)

    for item in results:
        gt_raw = item.get("gt_quadruples", []) or []
        pr_raw = item.get("pred_quadruples", []) or []

        gt_all = []
        pr_all = []

        for q in gt_raw:
            qq = normalize_quad(q)
            if hate_only and (qq.hateful.lower() != "hate"):
                continue
            gt_all.extend(expand_by_group(qq))

        for q in pr_raw:
            qq = normalize_quad(q)
            if hate_only and (qq.hateful.lower() != "hate"):
                continue
            pr_all.extend(expand_by_group(qq))

        # group by class for this sample
        gt_by = defaultdict(list)
        pr_by = defaultdict(list)

        for q in gt_all:
            if q.targeted_group in class_set:
                gt_by[q.targeted_group].append(q)
        for q in pr_all:
            if q.targeted_group in class_set:
                pr_by[q.targeted_group].append(q)

        for c in classes:
            g = gt_by.get(c, [])
            p = pr_by.get(c, [])

            tpc = hard_match_tp(p, g)
            fpc = len(p) - tpc
            fnc = len(g) - tpc

            tp[c] += tpc
            fp[c] += fpc
            fn[c] += fnc

    out = {}
    for c in classes:
        p = safe_div(tp[c], tp[c] + fp[c])
        r = safe_div(tp[c], tp[c] + fn[c])
        out[c] = {
            "precision": p,
            "recall": r,
            "f1": f1(p, r),
            "tp": float(tp[c]),
            "fp": float(fp[c]),
            "fn": float(fn[c]),
        }
    return out


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return mu, math.sqrt(var)


# -------------------------
# Plot
# -------------------------

def plot_grouped_bars(
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    classes: List[str],
    methods: List[str],
    metric: str,
    out_path: str,
    title: str = "",
):
    """
    agg[method][class] = (mean, std)
    """
    n_cls = len(classes)
    n_m = len(methods)

    x = list(range(n_cls))
    total_width = 0.8
    bar_w = total_width / max(1, n_m)
    left = [i - total_width / 2 for i in x]

    plt.figure(figsize=(12, 4.8))
    for mi, m in enumerate(methods):
        vals = [agg[m][c][0] for c in classes]
        errs = [agg[m][c][1] for c in classes]
        xpos = [left[i] + mi * bar_w for i in range(n_cls)]
        plt.bar(xpos, vals, width=bar_w, yerr=errs, capsize=3, label=m)

    # center ticks
    tick_pos = [left[i] + total_width / 2 - bar_w / 2 for i in range(n_cls)]
    plt.xticks(tick_pos, classes, rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel(metric)
    if title:
        plt.title(title)
    plt.legend(ncol=min(4, n_m), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_csv(path: str, rows: List[dict]):
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
                    help="Result JSON paths or glob patterns")
    ap.add_argument("--method_names", nargs="*",
                    help="Optional: method names aligned with expanded inputs. If omitted, inferred from filename prefix.")
    ap.add_argument("--out_dir", default="analysis_6class_bar")
    ap.add_argument("--classes", default="Racism,Sexism,LGBTQ,Region,others,non-hate",
                    help="Comma-separated 6 classes in desired order")
    ap.add_argument("--hate_only", action="store_true",
                    help="If set, only evaluate hateful=='hate' (注意：这样会排除 non-hate 类)")
    ap.add_argument("--metrics", default="recall,f1",
                    help="Comma-separated metrics to plot: precision,recall,f1")
    ap.add_argument("--title_prefix", default="Six-class breakdown")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # expand globs
    paths = []
    for p in args.inputs:
        g = glob.glob(p)
        if g:
            paths.extend(g)
        else:
            paths.append(p)
    paths = sorted(paths)
    if not paths:
        raise SystemExit("No input files found.")

    # method names
    if args.method_names:
        if len(args.method_names) != len(paths):
            raise SystemExit("Length of --method_names must match number of expanded inputs.")
        mnames = args.method_names
    else:
        mnames = [infer_method_name(p) for p in paths]

    # run eval per file
    # store rows for CSV and for aggregation
    csv_rows = []
    per_method_class_metric = defaultdict(lambda: defaultdict(list))  # m -> c -> metric list

    for path, method in zip(paths, mnames):
        res = eval_one_file(path, classes=classes, hate_only=args.hate_only)
        run_id = os.path.basename(path)
        for c in classes:
            row = {
                "method": method,
                "run": run_id,
                "class": c,
                "precision": res[c]["precision"],
                "recall": res[c]["recall"],
                "f1": res[c]["f1"],
                "tp": res[c]["tp"],
                "fp": res[c]["fp"],
                "fn": res[c]["fn"],
            }
            csv_rows.append(row)
            for met in ["precision", "recall", "f1"]:
                per_method_class_metric[method][(c, met)].append(row[met])

    # save csv
    csv_path = os.path.join(args.out_dir, "six_class_metrics.csv")
    write_csv(csv_path, csv_rows)

    methods = sorted(set(mnames), key=lambda x: mnames.index(x))  # preserve first appearance order

    # aggregate mean/std per method/class/metric
    agg = {m: {c: {} for c in classes} for m in methods}
    for m in methods:
        for c in classes:
            for met in ["precision", "recall", "f1"]:
                vals = per_method_class_metric[m].get((c, met), [])
                mu, sd = mean_std(vals)
                agg[m][c][met] = (mu, sd)

    # plot each requested metric
    for met in metrics:
        # reformat for plotting: agg_plot[m][c] = (mean,std)
        agg_plot = {m: {c: agg[m][c][met] for c in classes} for m in methods}
        out_png = os.path.join(args.out_dir, f"bar_{met}.png")
        title = f"{args.title_prefix} ({met})"
        plot_grouped_bars(
            agg=agg_plot,
            classes=classes,
            methods=methods,
            metric=met,
            out_path=out_png,
            title=title,
        )

    print(f"[OK] CSV: {csv_path}")
    for met in metrics:
        print(f"[OK] PNG: {os.path.join(args.out_dir, f'bar_{met}.png')}")


if __name__ == "__main__":
    main()
