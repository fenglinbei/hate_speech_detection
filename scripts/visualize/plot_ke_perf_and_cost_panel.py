#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 直接复用你现有脚本的函数（两者都在 if __name__ == "__main__" 保护下，import 不会自动跑 main）
from scripts.visualize.plot_ke_multi_seed_paper import setup_paper_style, collect_points, plot_single
from scripts.visualize.draw_k_ablation_graph import load_one_json, extract_ke_from_filename

def plot_cost_on_ax(ax1, x, avg_k_used, trunc_ratio, cost_color="red", trunc_color="red"):
    # 左 y 轴：Avg K_used
    l1 = ax1.plot(x, avg_k_used, marker="o", label=r"Avg $k_b$", color=cost_color)
    ax1.set_xlabel(r"$k_e$")
    ax1.set_ylabel(r"$k_b$")

    step = 2 if len(x) >= 10 else 1
    ax1.set_xticks(x[::step])

    # 右 y 轴：Truncation ratio
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, trunc_ratio, marker="s", linestyle="--", label="Truncation ratio", color=trunc_color)
    ax2.set_ylabel("Truncation ratio")
    ax2.set_ylim(0.0, 1.0)

    # 合并图例
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, frameon=False, loc="best")

    ax1.grid(False)
    ax1.tick_params(axis="x", labelrotation=0)
    return ax2


def main():
    ap = argparse.ArgumentParser()
    # 左图（multi-seed aggregate）
    ap.add_argument("--perf_dir", type=str, default="runner/output/k_ablation")  # 存放 k*_multi_seed.json 的目录
    ap.add_argument("--err", type=str, default="std", choices=["std", "sem"])
    ap.add_argument("--percent", action="store_true")

    # 右图（cost）
    ap.add_argument("--cost_dir", type=str, default="runner/output/k_ablation")  # 存放 k*.json 的目录
    ap.add_argument("--k_min", type=int, default=1)
    ap.add_argument("--k_max", type=int, default=20)

    # 输出
    ap.add_argument("--out_dir", type=str, default="analyse/k_ablation")
    ap.add_argument("--out_name", type=str, default="ke_perf_cost_panel")
    ap.add_argument("--dpi", type=int, default=300)

    # 版式
    ap.add_argument("--fontsize", type=int, default=9)
    ap.add_argument("--width_in", type=float, default=7.0)   # 横向大图（双栏常用宽度）
    ap.add_argument("--height_in", type=float, default=2.4)

    args = ap.parse_args()

    # 统一风格（沿用 perf 脚本的 paper 风格）
    setup_paper_style(fontsize=args.fontsize)

    # -------- 左图：performance（Avg F1）--------
    perf_rows = collect_points(Path(args.perf_dir), args.err)
    x_perf = np.array([r["ke"] for r in perf_rows])
    y_perf = np.array([r["avg_mean"] for r in perf_rows])
    yerr_perf = np.array([r["avg_err"] for r in perf_rows])

    # -------- 右图：cost（Avg K_used + trunc ratio）--------
    cost_dir = Path(args.cost_dir)
    json_files = []
    for p in cost_dir.glob("k*.json"):
        ke = extract_ke_from_filename(p)
        if ke is None:
            continue
        if args.k_min <= ke <= args.k_max:
            json_files.append(p)

    rows = [load_one_json(p) for p in sorted(json_files, key=lambda x: extract_ke_from_filename(x) or 10**9)]
    rows.sort(key=lambda r: r[0])

    x_cost = [r[0] for r in rows]
    avg_k_used = [r[4] for r in rows]
    trunc_ratio = [r[5] for r in rows]

    # 过滤掉 cost 不可用的点
    keep = [i for i in range(len(x_cost)) if avg_k_used[i] is not None and trunc_ratio[i] is not None]
    x_cost = [x_cost[i] for i in keep]
    avg_k_used = [avg_k_used[i] for i in keep]
    trunc_ratio = [trunc_ratio[i] for i in keep]

    # -------- 拼图：1×2 横向 --------
    fig, axes = plt.subplots(1, 2, figsize=(args.width_in, args.height_in), constrained_layout=True)

    # 左：performance
    plot_single(
        axes[0],
        x_perf, y_perf, yerr_perf,
        title=r"Impact of $k_e$",
        xlabel=r"$k_e$",
        ylabel="Avg-F1",
        invert_x=False,
        percent=args.percent,
    )
    axes[0].tick_params(axis="x", labelrotation=0)

    # 右：cost（两条曲线都设为红色）
    plot_cost_on_ax(axes[1], x_cost, avg_k_used, trunc_ratio, cost_color="#C44E52", trunc_color="#C44E52")

    # 保存
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / args.out_name
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_base.with_suffix('.pdf')}")
    print(f"Saved: {out_base.with_suffix('.png')}")

if __name__ == "__main__":
    main()
