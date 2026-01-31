#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot K_e ablation curves from runner/output/k_ablation/k{6..20}.json

Figure 1: K_e vs Avg-F1 (main) with Hard-F1 & Soft-F1.
Figure 2: K_e vs Avg K_used + truncation ratio (K_used < K_e).

K_used definition (fixed):
- Count actual demonstration examples inserted in the prompt.
- Prefer counting occurrences of "[END]" inside the "示例：" section.
  (Each demo ends with [END], while the final query does not.)
- Fallback: count occurrences of "### 句子：" inside "示例：" section minus 1 (exclude final query).

Outputs:
- ke_perf_curve.pdf/.png
- ke_cost_curve.pdf/.png
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------- filename helpers -----------------

def extract_ke_from_filename(p: Path) -> Optional[int]:
    m = re.match(r"^k(\d+)\.json$", p.name)
    return int(m.group(1)) if m else None


# ----------------- prompt parsing -----------------

def extract_user_prompt_from_result_item(item: Dict[str, Any]) -> Optional[str]:
    """
    k10.json shows:
      item["messages_list"] == [ [ {"role":"system",...}, {"role":"user","content":...} ] ]
    We take the last conversation, last user message.
    """
    ml = item.get("messages_list")
    if not ml or not isinstance(ml, list):
        return None
    convo = ml[-1]
    if not isinstance(convo, list):
        return None
    user_msgs = [m for m in convo if isinstance(m, dict) and m.get("role") == "user"]
    if not user_msgs:
        return None
    return user_msgs[-1].get("content", "")


def get_examples_section(prompt_text: str) -> Optional[str]:
    """
    Return substring after "示例：" (inclusive split), or None if not found.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return None
    m = re.search(r"\n?\s*示例\s*[:：]\s*", prompt_text)
    if not m:
        return None
    return prompt_text[m.end():]


def count_k_used_from_prompt(prompt_text: str) -> Optional[int]:
    """
    K_used = number of demonstrations actually included.

    Prefer: count "[END]" in examples section.
    Fallback: count "### 句子：" in examples section minus 1 (exclude the final query).
    """
    ex = get_examples_section(prompt_text)
    if ex is None:
        return None

    n_end = len(re.findall(r"\[END\]", ex))
    if n_end > 0:
        return n_end

    # fallback: count sentence blocks, exclude last query if present
    n_sent = len(re.findall(r"###\s*句子\s*[:：]", ex))
    if n_sent == 0:
        return None
    return max(n_sent - 1, 0)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_one_json(p: Path) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], int]:
    """
    Returns:
    (K_e, f1_avg, f1_hard, f1_soft, avg_k_used, trunc_ratio, n_samples_used_for_cost)
    """
    ke = extract_ke_from_filename(p)
    if ke is None:
        raise ValueError(f"Bad filename (expect k{{int}}.json): {p.name}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metric = data.get("metric", {}) or {}
    f1_avg = safe_float(metric.get("f1_avg"))
    f1_hard = safe_float(metric.get("f1_hard"))
    f1_soft = safe_float(metric.get("f1_soft"))

    results = data.get("results", []) or []
    k_used_list: List[int] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        prompt = extract_user_prompt_from_result_item(item)
        if prompt is None:
            continue
        ku = count_k_used_from_prompt(prompt)
        if ku is None:
            continue
        k_used_list.append(int(ku))

    if len(k_used_list) == 0:
        avg_k_used = None
        trunc_ratio = None
        n_cost = 0
    else:
        avg_k_used = float(np.mean(k_used_list))
        trunc_ratio = float(np.mean([1.0 if ku < ke else 0.0 for ku in k_used_list]))
        n_cost = len(k_used_list)

    return ke, f1_avg, f1_hard, f1_soft, avg_k_used, trunc_ratio, n_cost


# ----------------- plotting -----------------

def setup_matplotlib_paper(fontsize: int = 8) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": fontsize,

        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,

        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })



def save_fig(fig, out_base: Path, dpi: int = 300) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


import matplotlib.ticker as mticker

def plot_perf_curve_paper(x, avg, hard, soft, width_in, height_in,
                          avg_err=None, hard_err=None, soft_err=None,
                          title=None, use_percent=False):
    import matplotlib.pyplot as plt

    x = np.array(x)
    avg = np.array(avg, dtype=float)
    hard = np.array(hard, dtype=float)
    soft = np.array(soft, dtype=float)

    scale = 100.0 if use_percent else 1.0
    avg *= scale; hard *= scale; soft *= scale
    if avg_err is not None: avg_err = np.array(avg_err, dtype=float) * scale
    if hard_err is not None: hard_err = np.array(hard_err, dtype=float) * scale
    if soft_err is not None: soft_err = np.array(soft_err, dtype=float) * scale

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    # 阴影带（如果有误差）
    if avg_err is not None:
        ax.fill_between(x, avg-avg_err, avg+avg_err, alpha=0.15, linewidth=0)
    if hard_err is not None:
        ax.fill_between(x, hard-hard_err, hard+hard_err, alpha=0.12, linewidth=0)
    if soft_err is not None:
        ax.fill_between(x, soft-soft_err, soft+soft_err, alpha=0.12, linewidth=0)

    # 线+小marker（像图1）
    ax.plot(x, avg,  marker="s", label="Avg-F1")
    ax.plot(x, hard, marker="s", label="Hard-F1")
    ax.plot(x, soft, marker="s", label="Soft-F1")

    ax.set_xlabel(r"$K_e$")
    ax.set_ylabel("F1 (%)" if use_percent else "F1")

    # x刻度稀疏一点（像图1那样干净）
    step = 2 if len(x) >= 10 else 1
    ax.set_xticks(x[::step])

    # 不要网格（图1就是干净底）
    ax.grid(False)

    # 全边框保留（图1风格）
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(True)

    if title:
        ax.set_title(title, fontweight="bold")

    # legend 放在图内上方，别挡线（也更像图1）
    ax.legend(loc="upper right")

    return fig

def plot_perf_curve(x, f1_avg, f1_hard, f1_soft, width_in, height_in):
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    
    # 线条：稍细；marker：更小；markevery：减少marker密度
    ax.plot(x, f1_avg,  marker="o", markevery=2, label="Avg-F1")
    ax.plot(x, f1_hard, marker="s", markevery=2, label="Hard-F1")
    ax.plot(x, f1_soft, marker="^", markevery=2, label="Soft-F1")

    ax.set_xlabel(r"$K_e$")
    ax.set_ylabel("F1")

    # x轴：减少刻度密度（每2个一个），不旋转
    if len(x) > 0:
        step = 2 if len(x) >= 10 else 1
        ax.set_xticks(x[::step])

    # y轴：合理刻度，且略收紧范围（留一点padding）
    y_all = np.array(f1_avg + f1_hard + f1_soft, dtype=float)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    pad = (y_max - y_min) * 0.15 if y_max > y_min else 0.02
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # 网格：只保留y方向，变淡
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    # 边框：去掉上/右，更“论文风”
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例：移到图外上方，横排，不占数据区
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    # 给图例留空间（上方）
    fig.subplots_adjust(top=0.80)

    return fig



def plot_cost_curve(x, avg_k_used, trunc_ratio, width_in, height_in):
    fig, ax1 = plt.subplots(figsize=(width_in, height_in), constrained_layout=True)

    l1 = ax1.plot(x, avg_k_used, marker="o", label=r"Avg $K_{used}$")
    ax1.set_xlabel(r"$K_e$")
    ax1.set_ylabel(r"Avg $K_{used}$")
    ax1.set_xticks(x)
    if len(x) >= 10:
        ax1.tick_params(axis="x", labelrotation=45)
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    ax2 = ax1.twinx()
    l2 = ax2.plot(x, trunc_ratio, marker="s", linestyle="--", label="Truncation ratio")
    ax2.set_ylabel("Truncation ratio")
    ax2.set_ylim(0.0, 1.0)

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")
    return fig


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default="runner/output/k_ablation")
    ap.add_argument("--out_dir", type=str, default="analyse/k_ablation")
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=20)

    # single-column width ~3.35in; double-column width ~6.9in
    ap.add_argument("--width_in", type=float, default=3.35)
    ap.add_argument("--height_in", type=float, default=2.25)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--font_size", type=int, default=8)

    args = ap.parse_args()

    setup_matplotlib_paper(fontsize=args.font_size)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    json_files = []
    for p in in_dir.glob("k*.json"):
        ke = extract_ke_from_filename(p)
        if ke is None:
            continue
        if args.k_min <= ke <= args.k_max:
            json_files.append(p)

    if not json_files:
        raise SystemExit(f"No k*.json found in {in_dir} within [{args.k_min}, {args.k_max}]")

    rows = [load_one_json(p) for p in sorted(json_files, key=lambda x: extract_ke_from_filename(x) or 10**9)]
    rows.sort(key=lambda r: r[0])

    x = [r[0] for r in rows]
    f1_avg = [r[1] for r in rows]
    f1_hard = [r[2] for r in rows]
    f1_soft = [r[3] for r in rows]
    avg_k_used = [r[4] for r in rows]
    trunc_ratio = [r[5] for r in rows]
    n_cost = [r[6] for r in rows]

    # summary
    print("\n=== K_e Ablation Summary ===")
    print("K_e\tAvg-F1\tHard-F1\tSoft-F1\tAvgK_used\tTruncRatio\tN(cost)")
    for ke, a, h, s, ku, tr, nc in rows:
        print(f"{ke}\t{a if a is not None else 'NA'}\t{h if h is not None else 'NA'}\t"
              f"{s if s is not None else 'NA'}\t{ku if ku is not None else 'NA'}\t"
              f"{tr if tr is not None else 'NA'}\t{nc}")

    # performance figure (require metrics exist)
    keep_perf = [i for i in range(len(x)) if f1_avg[i] is not None and f1_hard[i] is not None and f1_soft[i] is not None]
    x_perf = [x[i] for i in keep_perf]
    fig1 = plot_perf_curve_paper(
        x_perf,
        [f1_avg[i] for i in keep_perf],
        [f1_hard[i] for i in keep_perf],
        [f1_soft[i] for i in keep_perf],
        width_in=args.width_in,
        height_in=args.height_in,
    )
    save_fig(fig1, out_dir / "ke_perf_curve", dpi=args.dpi)

    # cost figure (require K_used stats exist)
    keep_cost = [i for i in range(len(x)) if avg_k_used[i] is not None and trunc_ratio[i] is not None]
    if keep_cost:
        x_cost = [x[i] for i in keep_cost]
        fig2 = plot_cost_curve(
            x_cost,
            [avg_k_used[i] for i in keep_cost],
            [trunc_ratio[i] for i in keep_cost],
            width_in=args.width_in,
            height_in=args.height_in,
        )
        save_fig(fig2, out_dir / "ke_cost_curve", dpi=args.dpi)
    else:
        print("\n[WARN] Could not compute K_used/trunc ratio from prompts. "
              "Check if results/messages_list contains the full prompt with '示例：' and '[END]' markers.")

    print(f"\nSaved figures to: {out_dir.resolve()}")
    print(" - ke_perf_curve.pdf / ke_perf_curve.png")
    print(" - ke_cost_curve.pdf / ke_cost_curve.png (if cost stats available)\n")


if __name__ == "__main__":
    main()
