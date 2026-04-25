# plot_target_groups_barh.py
# Usage examples:
# 1) Normal but clean + transparent:
#    python plot_target_groups_barh.py --input train.json --output out.png --transparent
#
# 2) Ultra-minimal for method overview (only bars):
#    python plot_target_groups_barh.py --input train.json --output out.png --transparent --minimal --tight
#
# 3) Minimal but keep labels:
#    python plot_target_groups_barh.py --input train.json --output out.png --transparent --minimal --keep_labels

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ORDER = ["Racism", "Sexism", "LGBTQ", "Region", "others"]
MORANDI_HEX = [
    "#ECA8A9",
    "#74AED4",
    "#D3E2B7",
    "#CFAFD4",
    "#F7C97E",
]

def pick_colors(n: int, palette: str):
    if palette == "morandi":
        base = MORANDI_HEX
        # repeat if categories > palette size
        return [base[i % len(base)] for i in range(n)]
    elif palette == "gray":
        # monotone gray ramp (still distinct)
        # avoid too light colors
        vals = [0.25 + 0.5 * (i / max(1, n-1)) for i in range(n)]
        return [(v, v, v) for v in vals]
    else:
        # matplotlib default
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(n)]



def load_target_groups(input_path: Path) -> dict:
    """Load target_groups from aggregated stats JSON.
    Fallback supports raw format in a best-effort way."""
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Aggregated stats format
    if isinstance(data, dict) and "target_groups" in data and isinstance(data["target_groups"], dict):
        return {k: int(v) for k, v in data["target_groups"].items()}

    # Fallback: raw format (best-effort)
    groups = {k: 0 for k in DEFAULT_ORDER}
    items = data if isinstance(data, list) else data.get("data", [])
    for item in items:
        quads = item.get("quadruples", []) or item.get("labels", []) or []
        for q in quads:
            g = q.get("targeted_group") or q.get("target_group")
            if not isinstance(g, str):
                continue
            # keep only single-label
            if any(sep in g for sep in [",", "\\", "/", "|"]):
                continue
            if g in groups:
                groups[g] += 1
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to train.json")
    ap.add_argument("--output", type=str, required=True, help="Path to output PNG")
    ap.add_argument("--dpi", type=int, default=400, help="PNG dpi")
    ap.add_argument("--figsize", type=float, nargs=2, default=[6.8, 2.2], help="Figure size: W H")
    ap.add_argument("--bar_height", type=float, default=0.98, help="Bar height (close to 1.0 -> touch)")
    ap.add_argument("--sort", choices=["fixed", "desc", "asc"], default="fixed", help="Order of categories")

    # New features
    ap.add_argument("--transparent", action="store_true", default=True,
                    help="Save with transparent background (default: on)")
    ap.add_argument("--no_transparent", action="store_true",
                    help="Disable transparent background (overrides --transparent)")

    ap.add_argument("--minimal", action="store_true",
                    help="Hide everything except bars (method-overview style)")
    ap.add_argument("--keep_labels", action="store_true",
                    help="In minimal mode, keep y tick labels (category names)")
    ap.add_argument("--keep_axis", action="store_true",
                    help="In minimal mode, keep axis/ticks (not recommended for overview icons)")
    ap.add_argument("--tight", action="store_true",
                    help="Remove extra paddings/margins as much as possible")
    ap.add_argument(
        "--palette",
        choices=["morandi", "tab10", "gray"],
        default="morandi",
        help="Color palette for bars (default: morandi, print-friendly)."
    )


    # Additional “flat” tweaks
    ap.add_argument("--no_spines", action="store_true", default=True,
                    help="Remove plot spines (default: on)")
    ap.add_argument("--value_labels", action="store_true",
                    help="Annotate numbers at the end of each bar")
    ap.add_argument("--xpad_ratio", type=float, default=0.03,
                    help="Right padding ratio for text labels (if enabled)")

    args = ap.parse_args()

    # resolve transparency flags
    transparent = args.transparent and (not args.no_transparent)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tg = load_target_groups(input_path)

    # order
    keys = [k for k in DEFAULT_ORDER if k in tg] if args.sort == "fixed" else \
           sorted(tg.keys(), key=lambda k: tg[k], reverse=(args.sort == "desc"))

    values = [tg[k] for k in keys]

    # colors: different category different color
    colors = pick_colors(len(keys), args.palette)

    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)

    y = list(range(len(keys)))
    ax.barh(y, values, height=args.bar_height, color=colors)

    # Put x-axis on TOP, and make the first bar appear at the top (top-left origin feel)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()

    # x-axis starts from 0 on the left
    ax.set_xlim(left=0)

    # default labels (non-minimal)
    if not args.minimal:
        ax.set_yticks(y)
        ax.set_yticklabels(keys)
        ax.set_xlabel("Count")
        ax.set_title("Target group (single-label) distribution")
    else:
        # Minimal mode: hide everything except bars
        if args.keep_labels:
            ax.set_yticks(y)
            ax.set_yticklabels(keys)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        if not args.keep_axis:
            ax.set_xticks([])
            ax.set_xlabel("")
        ax.set_title("")

    # remove spines for a flatter look
    if args.no_spines:
        for s in ["top", "right", "bottom", "left"]:
            ax.spines[s].set_visible(False)

    # optional numeric labels (useful if you keep labels/axes)
    if args.value_labels:
        xmax = max(values) if values else 0
        pad = max(1, int(args.xpad_ratio * xmax))
        for yi, v in zip(y, values):
            ax.text(v + pad, yi, str(v), va="center", ha="left", fontsize=9)

    # margins / whitespace control
    if args.tight:
        # remove outer paddings
        ax.margins(x=0.0, y=0.0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # save
    fig.savefig(
        output_path,
        format="png",
        bbox_inches="tight" if not args.tight else None,  # tight mode uses subplots_adjust
        pad_inches=0 if args.tight else 0.02,
        transparent=transparent,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
