#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import matplotlib
from pathlib import Path
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def setup_paper_style(fontsize: int = 10):
    """Match 'paper figure' aesthetics similar to your example."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

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


def parse_ke_from_filename(p: Path) -> Optional[int]:
    # accepts k2_multi_seed.json, k10_multi_seed.json
    m = re.match(r"^k(\d+)_multi_seed\.json$", p.name)
    return int(m.group(1)) if m else None


def load_aggregate(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f).get("aggregate", {}) or {}


def get_mean_std(agg: Dict, key: str) -> Tuple[float, float, int]:
    """
    key example: 'f1_avg', 'f1_hard', 'f1_soft'
    returns (mean, std, n)
    """
    node = agg.get(key, {}) or {}
    mean = float(node.get("mean"))
    std = float(node.get("std", 0.0))
    n = int(node.get("n", 1))
    return mean, std, n


def compute_err(std: float, n: int, err_type: str) -> float:
    if err_type == "std":
        return std
    if err_type == "sem":
        return std / np.sqrt(max(n, 1))
    raise ValueError(f"Unknown err_type: {err_type}")


def collect_points(input_dir: Path, err_type: str):
    files = sorted(input_dir.glob("k*_multi_seed.json"))
    rows = []

    for p in files:
        ke = parse_ke_from_filename(p)
        if ke is None:
            continue
        agg = load_aggregate(p)

        avg_m, avg_s, avg_n = get_mean_std(agg, "f1_avg")
        hard_m, hard_s, hard_n = get_mean_std(agg, "f1_hard")
        soft_m, soft_s, soft_n = get_mean_std(agg, "f1_soft")

        rows.append({
            "ke": ke,
            "avg_mean": avg_m,
            "avg_err": compute_err(avg_s, avg_n, err_type),
            "hard_mean": hard_m,
            "hard_err": compute_err(hard_s, hard_n, err_type),
            "soft_mean": soft_m,
            "soft_err": compute_err(soft_s, soft_n, err_type),
        })

    rows.sort(key=lambda r: r["ke"])
    return rows


def plot_single(ax: Axes, x, y, yerr, title, xlabel, ylabel, invert_x=False, percent=False):
    if percent:
        y = np.array(y) * 100.0
        yerr = np.array(yerr) * 100.0

    # shading: mean ± err
    ax.fill_between(x, y - yerr, y + yerr, alpha=0.18, linewidth=0)
    ax.plot(x, y, marker="s")

    # ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # clean background like your example: no grid
    ax.grid(False)

    # full box spines like the example figure
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)

    # fewer x ticks for readability (similar density to example)
    if len(x) >= 12:
        step = 2
        ax.set_xticks(x[::step])
    else:
        ax.set_xticks(x)

    if invert_x:
        ax.invert_xaxis()


def plot_triple(ax, x, series, title, xlabel, ylabel, invert_x=False, percent=False):
    # series: list of (label, y, yerr, marker)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(False)
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)

    for label, y, yerr, marker in series:
        y = np.array(y, dtype=float)
        yerr = np.array(yerr, dtype=float)
        if percent:
            y *= 100.0
            yerr *= 100.0

        ax.fill_between(x, y - yerr, y + yerr, alpha=0.12, linewidth=0)
        ax.plot(x, y, marker=marker, label=label)

    if len(x) >= 12:
        ax.set_xticks(x[::2])
    else:
        ax.set_xticks(x)

    if invert_x:
        ax.invert_xaxis()

    ax.legend(loc="upper right")


def save(fig, out_base: Path, dpi: int):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="runner/output/k_ablation")
    ap.add_argument("--out_dir", type=str, default="runner/output/k_ablation/figs")
    ap.add_argument("--mode", type=str, default="single", choices=["single", "triple"],
                    help="single: only Avg-F1; triple: Avg/Hard/Soft in one plot")
    ap.add_argument("--err", type=str, default="std", choices=["std", "sem"],
                    help="shading uses mean±std or mean±sem")
    ap.add_argument("--metric", type=str, default="avg", choices=["avg", "hard", "soft"],
                    help="only for mode=single")
    ap.add_argument("--title", type=str, default=r"Impact of $K_e$")
    ap.add_argument("--xlabel", type=str, default=r"$k_e$")
    ap.add_argument("--ylabel", type=str, default="Avg-F1")
    ap.add_argument("--percent", action="store_true",
                    help="multiply y by 100 (e.g., 0.35 -> 35)")
    ap.add_argument("--invert_x", action="store_true",
                    help="reverse x-axis direction (like lambda 1->0 plots)")
    ap.add_argument("--width_in", type=float, default=3.6, help="single-column ~3.35-3.6")
    ap.add_argument("--height_in", type=float, default=2.4)
    ap.add_argument("--fontsize", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    setup_paper_style(fontsize=args.fontsize)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    rows = collect_points(input_dir, args.err)
    if not rows:
        raise SystemExit(f"No k*_multi_seed.json found in {input_dir}")

    x = [r["ke"] for r in rows]

    fig, ax = plt.subplots(figsize=(args.width_in, args.height_in))

    if args.mode == "single":
        if args.metric == "avg":
            y = [r["avg_mean"] for r in rows]
            yerr = [r["avg_err"] for r in rows]
            label_title = args.title
        elif args.metric == "hard":
            y = [r["hard_mean"] for r in rows]
            yerr = [r["hard_err"] for r in rows]
            label_title = args.title.replace(r"$K_e$", r"$K_e$")  # keep as-is
        else:
            y = [r["soft_mean"] for r in rows]
            yerr = [r["soft_err"] for r in rows]
            label_title = args.title.replace(r"$K_e$", r"$K_e$")

        plot_single(
            ax, np.array(x), np.array(y), np.array(yerr),
            title=label_title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            invert_x=args.invert_x,
            percent=args.percent,
        )
        out_base = out_dir / f"ke_{args.metric}_{args.err}_paper"
    else:
        series = [
            ("Avg-F1",  [r["avg_mean"] for r in rows],  [r["avg_err"] for r in rows],  "s"),
            ("Hard-F1", [r["hard_mean"] for r in rows], [r["hard_err"] for r in rows], "s"),
            ("Soft-F1", [r["soft_mean"] for r in rows], [r["soft_err"] for r in rows], "s"),
        ]
        plot_triple(
            ax, x, series,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            invert_x=args.invert_x,
            percent=args.percent,
        )
        out_base = out_dir / f"ke_triple_{args.err}_paper"

    save(fig, out_base, dpi=args.dpi)
    print(f"Saved: {out_base.with_suffix('.pdf')}")
    print(f"Saved: {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
