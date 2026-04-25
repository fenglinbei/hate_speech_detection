#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""long_tail_breakdown.py

Per-class (long-tail) breakdown for structured quadruple extraction.

This script aligns with the tuple-level F1 logic used in core.py / metric_llm.py:
- Hard match: (target, argument, targeted_group, hateful) all exactly equal (1-1 greedy matching).
- Soft match: targeted_group + hateful must match exactly, while target/argument are matched by
  SequenceMatcher similarity > threshold (1-1 greedy matching).
- Avg-F1: (Hard-F1 + Soft-F1) / 2.

It also provides a publication-style grouped bar chart with adjustable:
- bar width
- intra-group bar gap
- inter-group (class) gap
and uses a light/pastel color palette by default.
"""

import argparse
import csv
import glob
import json
import math
import copy
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

def setup_paper_style(fontsize: int = 10):
    """Match 'paper figure' aesthetics similar to your example."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],

        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize + 2,

        "axes.linewidth": 1.3,

        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,

        "lines.linewidth": 1.2,
        "lines.markersize": 2,

        "legend.frameon": False,

        # avoid Type3 fonts in pdf
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# Better PDF text embedding for paper figures (editable text in many viewers)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# -------------------------
# Helpers~
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
    """Split multi-label strings.

    Supports: "Racism, Sexism\\LGBTQ/Region" etc.
    Delimiters: comma / Chinese comma / slash / backslash / pipe / semicolon.
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


def norm_label(x) -> str:
    """Normalize label strings for robust matching (case/underscore/dash/spaces)."""
    s = norm_text(x).lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "-")
    return s


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def f1_from_pr(p: float, r: float) -> float:
    return safe_div(2 * p * r, p + r) if (p + r) != 0 else 0.0


@dataclass(frozen=True)
class Quad:
    target: str
    argument: str
    targeted_group: str
    hateful: str

    def key(self) -> Tuple[str, str, str, str]:
        return (self.target, self.argument, self.targeted_group, self.hateful)


NON_HATE_SET = {"non-hate", "nonhate"}


def normalize_quad(q: dict) -> Quad:
    return Quad(
        target=norm_text(q.get("target")),
        argument=norm_text(q.get("argument")),
        targeted_group=norm_label(q.get("targeted_group")),
        hateful=norm_label(q.get("hateful")),
    )


def expand_by_group(q: Quad) -> List[Quad]:
    """Expand multi-label targeted_group into single-label quads.

    Compatibility trick (common in this project):
    - If hateful is non-hate/non_hate, map it into a dedicated pseudo class "non-hate".
      This makes per-class plots with a "non-hate" column robust even if the original
      targeted_group field is not meaningful for non-hate examples.
    """
    if q.hateful in NON_HATE_SET:
        return [Quad(q.target, q.argument, "non-hate", "non-hate")]

    labs = [norm_label(x) for x in split_labels(q.targeted_group)]
    labs = [x for x in labs if x]
    if not labs:
        labs = ["unknown"]
    return [Quad(q.target, q.argument, lab, q.hateful) for lab in labs]


# -------------------------
# Matching (aligned with core.py / metric_llm.py)
# -------------------------

def greedy_match_counts(
    pred: List[Quad],
    gold: List[Quad],
    match_func,
) -> Tuple[int, int, int]:
    """Greedy 1-1 matching to compute TP/FP/FN.

    This mirrors compute_metrics(...) used in core.py / metric_llm.py.
    """
    matched_golds = set()
    matched_preds = set()
    tp = 0
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j in matched_golds:
                continue
            if match_func(p, g):
                matched_golds.add(j)
                matched_preds.add(i)
                tp += 1
                break
    fp = len(pred) - len(matched_preds)
    fn = len(gold) - len(matched_golds)
    return tp, fp, fn


def hard_match(p: Quad, g: Quad) -> bool:
    return p.key() == g.key()


def soft_match(p: Quad, g: Quad, threshold: float = 0.5) -> bool:
    """Soft match used in core.py / metric_llm.py.

    - targeted_group and hateful must match exactly
    - target and argument matched by similarity > threshold
    """
    if p.targeted_group != g.targeted_group or p.hateful != g.hateful:
        return False
    ts = SequenceMatcher(None, p.target, g.target).ratio()
    a_s = SequenceMatcher(None, p.argument, g.argument).ratio()
    return ts > threshold and a_s > threshold


# -------------------------
# Evaluation (per-class)
# -------------------------

def eval_one_file(
    path: str,
    classes: List[str],
    hate_only: bool,
    soft_threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate one result JSON.

    Returns:
        class_display -> {
            hard_precision, hard_recall, hard_f1,
            soft_precision, soft_recall, soft_f1,
            avg_f1,
            hard_tp/hard_fp/hard_fn,
            soft_tp/soft_fp/soft_fn
        }

    Notes:
    - Counts are micro-aggregated over the whole file (like core.py).
    - For multi-label targeted_group, each label is expanded as a separate quad.
    """
    d = read_json(path)
    results = d.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Bad format: {path}")

    cls_display = list(classes)
    cls_norm = [norm_label(c) for c in cls_display]
    cls_norm_set = set(cls_norm)

    # counts per class (hard/soft)
    tp_h = Counter()
    fp_h = Counter()
    fn_h = Counter()
    tp_s = Counter()
    fp_s = Counter()
    fn_s = Counter()

    for item in results:
        gt_raw = item.get("gt_quadruples", []) or []
        pr_raw = item.get("pred_quadruples", []) or []

        gt_all: List[Quad] = []
        pr_all: List[Quad] = []

        for q in gt_raw:
            qq = normalize_quad(q)
            if hate_only and qq.hateful != "hate":
                continue
            gt_all.extend(expand_by_group(qq))

        for q in pr_raw:
            qq = normalize_quad(q)
            if hate_only and qq.hateful != "hate":
                continue
            pr_all.extend(expand_by_group(qq))

        # group by class for this sample
        gt_by = defaultdict(list)
        pr_by = defaultdict(list)
        for q in gt_all:
            if q.targeted_group in cls_norm_set:
                gt_by[q.targeted_group].append(q)
        for q in pr_all:
            if q.targeted_group in cls_norm_set:
                pr_by[q.targeted_group].append(q)

        # accumulate per class
        for c_norm in cls_norm:
            g = gt_by.get(c_norm, [])
            p = pr_by.get(c_norm, [])

            th, fh, nh = greedy_match_counts(p, g, hard_match)
            tp_h[c_norm] += th
            fp_h[c_norm] += fh
            fn_h[c_norm] += nh

            ts, fs, ns = greedy_match_counts(p, g, lambda a, b: soft_match(a, b, threshold=soft_threshold))
            tp_s[c_norm] += ts
            fp_s[c_norm] += fs
            fn_s[c_norm] += ns

    out: Dict[str, Dict[str, float]] = {}
    for c_disp, c_norm in zip(cls_display, cls_norm):
        hp = safe_div(tp_h[c_norm], tp_h[c_norm] + fp_h[c_norm])
        hr = safe_div(tp_h[c_norm], tp_h[c_norm] + fn_h[c_norm])
        hf = f1_from_pr(hp, hr)

        sp = safe_div(tp_s[c_norm], tp_s[c_norm] + fp_s[c_norm])
        sr = safe_div(tp_s[c_norm], tp_s[c_norm] + fn_s[c_norm])
        sf = f1_from_pr(sp, sr)

        out[c_disp] = {
            "hard_precision": float(hp),
            "hard_recall": float(hr),
            "hard_f1": float(hf),
            "soft_precision": float(sp),
            "soft_recall": float(sr),
            "soft_f1": float(sf),
            "avg_f1": float((hf + sf) / 2),
            "hard_tp": float(tp_h[c_norm]),
            "hard_fp": float(fp_h[c_norm]),
            "hard_fn": float(fn_h[c_norm]),
            "soft_tp": float(tp_s[c_norm]),
            "soft_fp": float(fp_s[c_norm]),
            "soft_fn": float(fn_s[c_norm]),
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

def make_pastel_colors(n: int, cmap_name: str = "tab10", blend: float = 0.4) -> List[Tuple[float, float, float]]:
    """Generate a light/pastel palette by blending a base colormap with white."""
    cmap = plt.get_cmap(cmap_name)
    colors = []
    for i in range(n):
        r, g, b, _ = cmap(i % cmap.N)
        r = r * (1 - blend) + 1.0 * blend
        g = g * (1 - blend) + 1.0 * blend
        b = b * (1 - blend) + 1.0 * blend
        colors.append((r, g, b))
    return colors


def _pretty_class_label(c: str) -> str:
    """Pretty display names for x-axis ticks."""
    cn = norm_label(c)
    mapping = {
        "racism": "Racism",
        "sexism": "Sexism",
        "lgbtq": "LGBTQ",
        "region": "Region",
        "others": "Others",
        "other": "Others",
        "non-hate": "Non-Hate",
        "nonhate": "Non-Hate",
        "unknown": "Unknown",
    }
    return mapping.get(cn, c)


def _legend_reorder_for_row_major(
    handles: List,
    labels: List[str],
    ncol: int,
) -> Tuple[List, List[str]]:
    """Reorder legend entries so the *visible* left-to-right (row-major) order
    matches the bar drawing order.

    Matplotlib packs legend entries column-first when ncol>1. This helper
    permutes (handles, labels) so that the final layout reads row-major.
    """
    n = len(labels)
    if n <= 1 or ncol <= 1:
        return handles, labels
    ncol = min(int(ncol), n)

    base = n // ncol
    rem = n % ncol
    col_lens = [base + 1] * rem + [base] * (ncol - rem)
    nrows = max(col_lens) if col_lens else 0

    cols: List[List[int]] = [[] for _ in range(ncol)]
    idx = 0
    for r in range(nrows):
        for c in range(ncol):
            if r < col_lens[c] and idx < n:
                cols[c].append(idx)
                idx += 1

    order = [i for col in cols for i in col]
    return [handles[i] for i in order], [labels[i] for i in order]


def plot_grouped_bars(
    agg: Dict[str, Dict[str, Tuple[float, float]]],
    classes: List[str],
    methods: List[str],
    metric: str,
    out_path: str,
    title: str = "",
    bar_width: float = 0.12,
    bar_gap: float = 0.03,
    group_gap: float = 0.28,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    out_format: str = "pdf",
    dpi: int = 240,
    legend_ncol: int = 4,
    legend_row_major: bool = True,
):
    """Grouped bar plot.

    Args:
        agg: agg[method][class] = (mean, std)
        classes: x-axis class names (display)
        methods: method names
        metric: label
        bar_width: width of each bar
        bar_gap: gap between bars within one class
        group_gap: gap between class groups
    """
    n_cls = len(classes)
    n_m = len(methods)

    if colors is None:
        colors = make_pastel_colors(n_m)

    cluster_w = n_m * bar_width + (n_m - 1) * bar_gap
    group_stride = cluster_w + group_gap
    group_lefts = [i * group_stride for i in range(n_cls)]
    tick_pos = [gl + cluster_w / 2 for gl in group_lefts]

    plt.figure(figsize=(6, 3))
    ax = plt.gca()
    ax.set_axisbelow(True)
    # ax.yaxis.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for mi, m in enumerate(methods):
        vals = [agg[m][c][0] for c in classes]
        errs = [agg[m][c][1] for c in classes]
        xpos = [group_lefts[i] + mi * (bar_width + bar_gap) for i in range(n_cls)]
        ax.bar(
            xpos,
            vals,
            width=bar_width,
            yerr=errs,
            align="edge",
            capsize=0.0,
            linewidth=0.0,
            color=colors[mi],
            label=m,
        )

    ax.set_xticks(tick_pos)
    ax.set_xticklabels([_pretty_class_label(c) for c in classes], rotation=0)

    ylab = {
        "hard_f1": "Hard-F1",
        "soft_f1": "Soft-F1",
        "avg_f1": "Avg-F1",
    }.get(metric, metric)
    ax.set_ylabel(ylab)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        vmax = 0.0
        for m in methods:
            for c in classes:
                vmax = max(vmax, agg[m][c][0] + agg[m][c][1])
        ax.set_ylim(0, max(0.05, min(1.0, vmax * 1.25)))

    if title:
        ax.set_title(title)

    # Keep legend order consistent with bar/color order.
    handles, labels = ax.get_legend_handles_labels()
    if legend_row_major:
        handles, labels = _legend_reorder_for_row_major(handles, labels, ncol=legend_ncol)
    ax.legend(
        handles,
        labels,
        ncol=min(int(legend_ncol), n_m),
        fontsize=9,
        frameon=False,
    )
    plt.tight_layout()

    fmt = (out_format or "pdf").lower()
    if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
        plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    else:
        # Vector formats: dpi is irrelevant
        plt.savefig(out_path, bbox_inches="tight")
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
    ap.add_argument("--inputs", nargs="+", required=True, help="Result JSON paths or glob patterns")
    ap.add_argument(
        "--method_names",
        nargs="*",
        help=(
            "Optional: method names aligned with expanded inputs. "
            "If omitted, inferred from filename prefix."
        ),
    )
    ap.add_argument("--out_dir", default="analysis_long_tail_bar")
    ap.add_argument(
        "--classes",
        default="Racism,Sexism,LGBTQ,Region,others,non-hate",
        help="Comma-separated classes in desired order",
    )
    ap.add_argument(
        "--hate_only",
        action="store_true",
        help="If set, only evaluate hateful=='hate' (NOTE: this will exclude non-hate class)",
    )
    ap.add_argument(
        "--soft_threshold",
        type=float,
        default=0.5,
        help="Soft-match similarity threshold for target/argument (SequenceMatcher ratio).",
    )
    ap.add_argument(
        "--metrics",
        default="avg_f1",
        help="Comma-separated metrics to plot: hard_f1, soft_f1, avg_f1",
    )
    ap.add_argument("--bar_width", type=float, default=0.2, help="Bar width for each method bar")
    ap.add_argument("--bar_gap", type=float, default=0.03, help="Gap between method bars in a group")
    ap.add_argument("--group_gap", type=float, default=0.28, help="Gap between class groups")
    ap.add_argument(
        "--out_format",
        default="pdf",
        choices=["pdf", "png", "svg"],
        help="Figure output format.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Raster DPI (used for png only).",
    )
    ap.add_argument(
        "--legend_ncol",
        type=int,
        default=5,
        help="Number of legend columns.",
    )
    ap.add_argument(
        "--legend_col_major",
        action="store_true",
        help="Use Matplotlib's default column-major packing for legend entries.",
    )
    ap.add_argument(
        "--method_order",
        default="",
        help="Optional comma-separated method order for plotting/legend (e.g., 'IDS,CoT,DPP,Zero-Shot,SRAG,Ours').",
    )
    ap.add_argument(
        "--ylim",
        default="",
        help="Optional y-limits as 'ymin,ymax', e.g., '0,0.5'. Leave empty for auto.",
    )
    ap.add_argument("--title_prefix", default="Per-class")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    setup_paper_style(fontsize=10)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # expand globs
    paths: List[str] = []
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

    metric_fields = [
        "hard_precision",
        "hard_recall",
        "hard_f1",
        "soft_precision",
        "soft_recall",
        "soft_f1",
        "avg_f1",
    ]

    # run eval per file
    csv_rows: List[dict] = []
    per_method_class_metric = defaultdict(lambda: defaultdict(list))  # m -> (c, met) -> values

    for path, method in zip(paths, mnames):
        res = eval_one_file(
            path,
            classes=classes,
            hate_only=args.hate_only,
            soft_threshold=args.soft_threshold,
        )
        run_id = os.path.basename(path)
        for c in classes:
            row = {"method": method, "run": run_id, "class": c}
            for met in metric_fields:
                row[met] = res[c][met]
            for met in [
                "hard_tp",
                "hard_fp",
                "hard_fn",
                "soft_tp",
                "soft_fp",
                "soft_fn",
            ]:
                row[met] = res[c][met]

            csv_rows.append(row)

            for met in metric_fields:
                per_method_class_metric[method][(c, met)].append(row[met])

    # save csv
    csv_path = os.path.join(args.out_dir, "per_class_metrics.csv")
    write_csv(csv_path, csv_rows)

    methods = sorted(set(mnames), key=lambda x: mnames.index(x))  # preserve first appearance order

    # Optional: user-specified plotting order (also controls colors)
    if args.method_order.strip():
        desired = [x.strip() for x in args.method_order.split(",") if x.strip()]

        def _mkey(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())

        m_map = {_mkey(m): m for m in methods}
        ordered: List[str] = []
        used = set()
        for d in desired:
            mm = m_map.get(_mkey(d))
            if mm and mm not in used:
                ordered.append(mm)
                used.add(mm)
        for m in methods:
            if m not in used:
                ordered.append(m)
        methods = ordered

    # aggregate mean/std per method/class/metric
    agg = {m: {c: {} for c in classes} for m in methods}
    for m in methods:
        for c in classes:
            for met in metric_fields:
                vals = per_method_class_metric[m].get((c, met), [])
                mu, sd = mean_std(vals)
                agg[m][c][met] = (mu, sd)

    # parse ylim
    ylim: Optional[Tuple[float, float]] = None
    if args.ylim.strip():
        try:
            a, b = [float(x) for x in args.ylim.split(",")]
            ylim = (a, b)
        except Exception:
            raise SystemExit("Bad --ylim format. Expected 'ymin,ymax'.")

    # plot each requested metric
    for met in metrics:
        if met not in metric_fields:
            raise SystemExit(f"Unknown metric '{met}'. Allowed: {', '.join(metric_fields)}")

        agg_plot = {m: {c: agg[m][c][met] for c in classes} for m in methods}
        out_fig = os.path.join(args.out_dir, f"bar_{met}.{args.out_format}")
        plot_grouped_bars(
            agg=agg_plot,
            classes=classes,
            methods=methods,
            metric=met,
            out_path=out_fig,
            title=None,
            bar_width=args.bar_width,
            bar_gap=args.bar_gap,
            group_gap=args.group_gap,
            ylim=ylim,
            out_format=args.out_format,
            dpi=args.dpi,
            legend_ncol=args.legend_ncol,
            legend_row_major=(not args.legend_col_major),
        )

    print(f"[OK] CSV: {csv_path}")
    for met in metrics:
        print(f"[OK] FIG: {os.path.join(args.out_dir, f'bar_{met}.{args.out_format}')}")


if __name__ == "__main__":
    main()
