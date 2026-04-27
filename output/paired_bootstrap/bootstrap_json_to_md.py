#!/usr/bin/env python3
"""
Convert paired-bootstrap JSON result files to Markdown tables.

Default output contains two tables:

1. Summary delta matrix, suitable for the main text.
2. Detailed paired-bootstrap table, suitable for reports or appendices.

The script is tailored for bootstrap JSON files shaped like:
{
  "metrics": ["f1_hard", "f1_soft", "f1_avg", "f1_target", "f1_hate"],
  "system_a": {"observed": {"f1_hard": ...}},
  "system_b": {"observed": {"f1_hard": ...}},
  "comparison": {
    "delta_label": "system_a_minus_system_b",
    "f1_hard": {
      "observed_delta": ...,
      "ci_lower": ...,
      "ci_upper": ...,
      "prob_a_better": ...,
      "prob_b_better": ...,
      "p_value_two_sided": ...
    }
  }
}

By default, system_a is treated as the Variant and system_b as Main.
The output delta is normalized to: Variant - Main.

Metric display names and default order are fixed as:
  Tar-F1, Hate-F1, Hard-F1, Soft-F1, Avg-F1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Display order requested for both the summary table and the detailed table.
METRIC_DISPLAY_ORDER: List[Tuple[str, str]] = [
    ("f1_target", "Tar-F1"),
    ("f1_hate", "Hate-F1"),
    ("f1_hard", "Hard-F1"),
    ("f1_soft", "Soft-F1"),
    ("f1_avg", "Avg-F1"),
]

DEFAULT_METRIC_ORDER: List[str] = [key for key, _ in METRIC_DISPLAY_ORDER]
METRIC_LABELS: Dict[str, str] = dict(METRIC_DISPLAY_ORDER)

# A small alias map so users can pass --metrics Tar-F1 Hate-F1 etc. if desired.
METRIC_ALIASES: Dict[str, str] = {
    "f1target": "f1_target",
    "targetf1": "f1_target",
    "tarf1": "f1_target",
    "target": "f1_target",
    "tar": "f1_target",
    "f1hate": "f1_hate",
    "hatef1": "f1_hate",
    "hate": "f1_hate",
    "f1hard": "f1_hard",
    "hardf1": "f1_hard",
    "hard": "f1_hard",
    "f1soft": "f1_soft",
    "softf1": "f1_soft",
    "soft": "f1_soft",
    "f1avg": "f1_avg",
    "avgf1": "f1_avg",
    "averagef1": "f1_avg",
    "avg": "f1_avg",
    "average": "f1_avg",
}

ResultRow = Dict[str, Any]


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def md_escape(value: object) -> str:
    """Escape values that will appear in Markdown table cells."""
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace("|", r"\|")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def markdown_table(headers: Sequence[str], align: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a Markdown table."""
    if len(headers) != len(align):
        raise ValueError("headers and align must have the same length")

    lines = [
        "| " + " | ".join(md_escape(h) for h in headers) + " |",
        "| " + " | ".join(align) + " |",
    ]

    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row has {len(row)} cells but expected {len(headers)}: {row}")
        lines.append("| " + " | ".join(md_escape(x) for x in row) + " |")

    return "\n".join(lines)


def infer_variant_name(path: Path) -> str:
    """Infer a readable variant name from file names like wo_semantic_vs_main.json."""
    stem = path.stem

    # Remove common comparison suffixes.
    stem = re.sub(r"(?:_vs_|-vs-|_against_|-against_).*$", "", stem, flags=re.IGNORECASE)

    # Make common ablation names easier to read.
    stem = re.sub(r"^(wo|w/o|without)[_-]+", "w/o_", stem, flags=re.IGNORECASE)
    stem = stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()

    if not stem:
        stem = path.stem

    # Keep "w/o semantic" lower-case, but title-case simple names like "uniform".
    if stem.lower().startswith("w/o "):
        return "w/o " + stem[4:].strip()
    return stem[:1].upper() + stem[1:]


def normalize_metric_name(metric: str) -> str:
    """Normalize metric names and paper-style aliases to the JSON key."""
    if metric in METRIC_LABELS:
        return metric

    compact = re.sub(r"[^A-Za-z0-9]+", "", metric).lower()
    return METRIC_ALIASES.get(compact, metric)


def metric_label(metric: str) -> str:
    """Return the display label used in all Markdown tables."""
    return METRIC_LABELS.get(metric, metric)


def ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def order_metrics(metrics: Sequence[str]) -> List[str]:
    """
    Order known metrics as:
      Tar-F1, Hate-F1, Hard-F1, Soft-F1, Avg-F1

    Unknown metrics are appended after known metrics in their first-seen order.
    """
    normalized = ordered_unique([normalize_metric_name(m) for m in metrics])
    present = set(normalized)

    known = [m for m in DEFAULT_METRIC_ORDER if m in present]
    unknown = [m for m in normalized if m not in DEFAULT_METRIC_ORDER]
    return known + unknown


def fmt_float(value: Optional[float], digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return ""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return str(value)
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}"


def fmt_pvalue(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return str(value)
    threshold = 10 ** (-digits)
    if 0 < float(value) < threshold:
        return f"<{threshold:.{digits}f}"
    return f"{float(value):.{digits}f}"


def significance_stars(p_value: Optional[float]) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def result_label(delta_variant_minus_main: float, p_value: Optional[float], alpha: float) -> str:
    if p_value is None or p_value >= alpha:
        return "n.s."
    if delta_variant_minus_main > 0:
        return "Variant better"
    if delta_variant_minus_main < 0:
        return "Main better"
    return "tie"


def get_metric_list(data: Dict[str, Any], override_metrics: Optional[Sequence[str]]) -> List[str]:
    if override_metrics:
        return order_metrics(list(override_metrics))

    if isinstance(data.get("metrics"), list) and data["metrics"]:
        return order_metrics(list(data["metrics"]))

    comparison = data.get("comparison", {})
    return order_metrics([k for k, v in comparison.items() if isinstance(v, dict)])


def get_observed_score(data: Dict[str, Any], system_key: str, metric: str) -> Optional[float]:
    system = data.get(system_key, {})
    observed = system.get("observed", {})
    if metric in observed:
        return observed[metric]

    # Fallback: some outputs store the observed score under system.bootstrap[metric].observed.
    bootstrap = system.get("bootstrap", {})
    if metric in bootstrap and isinstance(bootstrap[metric], dict):
        return bootstrap[metric].get("observed")
    return None


def normalize_comparison_to_variant_minus_main(
    comp_metric: Dict[str, Any],
    delta_label: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Return: delta, ci_lower, ci_upper, prob_main_better.

    The output delta is always Variant - Main, where Variant=system_a and Main=system_b.
    """
    observed_delta = comp_metric.get("observed_delta", comp_metric.get("mean_delta"))
    ci_lower = comp_metric.get("ci_lower")
    ci_upper = comp_metric.get("ci_upper")

    if delta_label == "system_a_minus_system_b":
        delta = observed_delta
        prob_main_better = comp_metric.get("prob_b_better")
        return delta, ci_lower, ci_upper, prob_main_better

    if delta_label == "system_b_minus_system_a":
        delta = -observed_delta if observed_delta is not None else None
        new_lower = -ci_upper if ci_upper is not None else None
        new_upper = -ci_lower if ci_lower is not None else None
        prob_main_better = comp_metric.get("prob_a_better")
        return delta, new_lower, new_upper, prob_main_better

    # Unknown label: use the stored delta and probabilities as-is, but keep the table usable.
    return observed_delta, ci_lower, ci_upper, comp_metric.get("prob_b_better")


def collect_results(
    json_paths: Sequence[Path],
    names: Optional[Sequence[str]] = None,
    metrics_override: Optional[Sequence[str]] = None,
) -> Tuple[List[ResultRow], List[str], List[Tuple[int, str]]]:
    """
    Load all JSON files and collect normalized per-metric rows.

    Returns:
      rows: per-file, per-metric result rows.
      metric_order: global metric order for both output tables.
      variant_order: input-order pairs of (variant_index, variant_name).
    """
    if names is not None and len(names) != len(json_paths):
        raise SystemExit("--names 的数量必须和输入 JSON 文件数量一致。")

    rows: List[ResultRow] = []
    metric_candidates: List[str] = []
    variant_order: List[Tuple[int, str]] = []

    for variant_idx, path in enumerate(json_paths):
        data = load_json(path)
        variant_name = names[variant_idx] if names is not None else infer_variant_name(path)
        variant_order.append((variant_idx, variant_name))

        delta_label = data.get("comparison", {}).get("delta_label", "system_a_minus_system_b")
        metrics = get_metric_list(data, metrics_override)

        for metric in metrics:
            comp_metric = data.get("comparison", {}).get(metric)
            if not isinstance(comp_metric, dict):
                print(f"Warning: metric {metric!r} not found in comparison of {path}", file=sys.stderr)
                continue

            metric_candidates.append(metric)
            variant_score = get_observed_score(data, "system_a", metric)
            main_score = get_observed_score(data, "system_b", metric)
            delta, ci_lower, ci_upper, prob_main_better = normalize_comparison_to_variant_minus_main(
                comp_metric, delta_label
            )
            p_value = comp_metric.get("p_value_two_sided")

            rows.append(
                {
                    "variant_idx": variant_idx,
                    "variant_name": variant_name,
                    "metric": metric,
                    "variant_score": variant_score,
                    "main_score": main_score,
                    "delta": delta,
                    "ci": comp_metric.get("ci"),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "prob_main_better": prob_main_better,
                    "p_value": p_value,
                }
            )

    metric_order = order_metrics(metric_candidates)
    return rows, metric_order, variant_order


def format_delta_cell(delta: Optional[float], p_value: Optional[float], digits: int, no_stars: bool) -> str:
    if delta is None:
        return ""
    stars = "" if no_stars else significance_stars(p_value)
    return fmt_float(delta, digits=digits, signed=True) + stars


def make_summary_table(
    rows: Sequence[ResultRow],
    metric_order: Sequence[str],
    variant_order: Sequence[Tuple[int, str]],
    digits: int = 4,
    no_stars: bool = False,
) -> str:
    """Create the Scheme-1 summary delta matrix."""
    row_lookup = {(r["variant_idx"], r["metric"]): r for r in rows}

    headers = ["Variant"] + [metric_label(m) for m in metric_order]
    align = [":-------------"] + ["--------:" for _ in metric_order]

    table_rows: List[List[str]] = []
    for variant_idx, variant_name in variant_order:
        cells = [variant_name]
        for metric in metric_order:
            r = row_lookup.get((variant_idx, metric))
            if r is None:
                cells.append("")
            else:
                cells.append(format_delta_cell(r["delta"], r["p_value"], digits, no_stars))
        table_rows.append(cells)

    return markdown_table(headers, align, table_rows)


def make_detail_table(
    rows: Sequence[ResultRow],
    metric_order: Sequence[str],
    variant_order: Sequence[Tuple[int, str]],
    baseline_name: str = "Main",
    digits: int = 4,
    alpha: float = 0.05,
    no_stars: bool = False,
) -> str:
    """Create the Scheme-2 detailed paired-bootstrap table."""
    row_lookup = {(r["variant_idx"], r["metric"]): r for r in rows}

    ci_values_clean = [r["ci"] for r in rows if r.get("ci") is not None]
    if ci_values_clean and len(set(ci_values_clean)) == 1:
        ci_label = f"{fmt_float(float(ci_values_clean[0]), digits=0)}% CI of Δ"
    else:
        ci_label = "CI of Δ"

    headers = [
        "Variant",
        "Metric",
        "Variant F1",
        f"{baseline_name} F1",
        f"Δ Variant−{baseline_name}",
        ci_label,
        f"P({baseline_name} > Variant)",
        "p(two-sided)",
        "Result",
    ]

    align = [
        ":-------------",
        ":----------",
        "-----------:",
        "--------:",
        ":---------------",
        ":--------------------",
        "------------------:",
        "-------------:",
        ":------------",
    ]

    table_rows: List[List[str]] = []
    for variant_idx, _variant_name in variant_order:
        for metric in metric_order:
            r = row_lookup.get((variant_idx, metric))
            if r is None:
                continue

            delta = r["delta"]
            if delta is None:
                result = ""
            else:
                result = result_label(float(delta), r["p_value"], alpha=alpha)

            ci_lower = r["ci_lower"]
            ci_upper = r["ci_upper"]
            if ci_lower is None or ci_upper is None:
                ci_cell = ""
            else:
                ci_cell = f"[{fmt_float(ci_lower, digits=digits)}, {fmt_float(ci_upper, digits=digits)}]"

            table_rows.append(
                [
                    r["variant_name"],
                    metric_label(metric),
                    fmt_float(r["variant_score"], digits=digits),
                    fmt_float(r["main_score"], digits=digits),
                    format_delta_cell(delta, r["p_value"], digits, no_stars),
                    ci_cell,
                    fmt_float(r["prob_main_better"], digits=digits),
                    fmt_pvalue(r["p_value"], digits=digits),
                    result,
                ]
            )

    return markdown_table(headers, align, table_rows)


def make_markdown_document(
    json_paths: Sequence[Path],
    names: Optional[Sequence[str]] = None,
    baseline_name: str = "Main",
    metrics_override: Optional[Sequence[str]] = None,
    digits: int = 4,
    alpha: float = 0.05,
    no_stars: bool = False,
    tables: str = "both",
    headings: bool = True,
) -> str:
    rows, metric_order, variant_order = collect_results(
        json_paths=json_paths,
        names=names,
        metrics_override=metrics_override,
    )

    parts: List[str] = []

    if tables in {"both", "summary"}:
        if headings:
            parts.append("### Table 1. Summary of bootstrap deltas")
        parts.append(make_summary_table(rows, metric_order, variant_order, digits=digits, no_stars=no_stars))

    if tables in {"both", "detail"}:
        if parts:
            parts.append("")
        if headings:
            title = "### Table 2. Detailed paired bootstrap comparison" if tables == "both" else "### Detailed paired bootstrap comparison"
            parts.append(title)
        parts.append(
            make_detail_table(
                rows,
                metric_order,
                variant_order,
                baseline_name=baseline_name,
                digits=digits,
                alpha=alpha,
                no_stars=no_stars,
            )
        )

    return "\n".join(parts)


# Backward-compatible function name used by the first version of this script.
def make_markdown_table(
    json_paths: Sequence[Path],
    names: Optional[Sequence[str]] = None,
    baseline_name: str = "Main",
    metrics_override: Optional[Sequence[str]] = None,
    digits: int = 4,
    alpha: float = 0.05,
    pretty_metric_names: bool = False,  # Kept for compatibility; labels are fixed now.
    no_stars: bool = False,
) -> str:
    _ = pretty_metric_names
    return make_markdown_document(
        json_paths=json_paths,
        names=names,
        baseline_name=baseline_name,
        metrics_override=metrics_override,
        digits=digits,
        alpha=alpha,
        no_stars=no_stars,
        tables="both",
        headings=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired-bootstrap JSON files to Markdown result tables."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more bootstrap result JSON files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .md file. If omitted, the tables are printed to stdout.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="Readable variant names, one per JSON file, e.g. --names Uniform 'w/o semantic'.",
    )
    parser.add_argument(
        "--baseline",
        default="Main",
        help="Baseline name shown in table headers. Default: Main.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metrics to include. Known metrics are always ordered as "
            "Tar-F1, Hate-F1, Hard-F1, Soft-F1, Avg-F1. "
            "You may pass JSON keys or labels, e.g. --metrics Tar-F1 Hate-F1 Avg-F1."
        ),
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of decimal places. Default: 4.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold used in the Result column. Default: 0.05.",
    )
    parser.add_argument(
        "--tables",
        choices=["both", "summary", "detail"],
        default="both",
        help="Which table(s) to output. Default: both.",
    )
    parser.add_argument(
        "--no-headings",
        action="store_true",
        help="Do not print Markdown section headings before tables.",
    )
    parser.add_argument(
        "--pretty-metric-names",
        action="store_true",
        help="Compatibility option. Metric labels are fixed as Tar-F1, Hate-F1, Hard-F1, Soft-F1, Avg-F1.",
    )
    parser.add_argument(
        "--no-stars",
        action="store_true",
        help="Do not append significance stars to delta values.",
    )
    parser.add_argument(
        "--note",
        action="store_true",
        help="Prepend a short Markdown note explaining Δ, metric order, and significance stars.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    output = make_markdown_document(
        json_paths=args.json_files,
        names=args.names,
        baseline_name=args.baseline,
        metrics_override=args.metrics,
        digits=args.digits,
        alpha=args.alpha,
        no_stars=args.no_stars,
        tables=args.tables,
        headings=not args.no_headings,
    )

    if args.note:
        star_note = "" if args.no_stars else " `* p < .05`, `** p < .01`, `*** p < .001`."
        note = (
            f"Δ = Variant − {args.baseline}. "
            f"Negative values mean the variant performs worse than {args.baseline}. "
            "Metrics are ordered as Tar-F1, Hate-F1, Hard-F1, Soft-F1, Avg-F1."
            f"{star_note}\n\n"
        )
        output = note + output

    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote Markdown tables to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
