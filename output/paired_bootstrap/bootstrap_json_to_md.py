#!/usr/bin/env python3
"""
Convert paired-bootstrap JSON result files to a Markdown table.

Expected JSON shape is like:
{
  "metrics": ["f1_hard", ...],
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
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_METRIC_LABELS = {
    "f1_hard": "f1_hard",
    "f1_soft": "f1_soft",
    "f1_avg": "f1_avg",
    "f1_target": "f1_target",
    "f1_hate": "f1_hate",
}


PRETTY_METRIC_LABELS = {
    "f1_hard": "Hard F1",
    "f1_soft": "Soft F1",
    "f1_avg": "Avg F1",
    "f1_target": "Target F1",
    "f1_hate": "Hate F1",
}


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
        return list(override_metrics)
    if isinstance(data.get("metrics"), list) and data["metrics"]:
        return list(data["metrics"])
    comparison = data.get("comparison", {})
    return [k for k, v in comparison.items() if isinstance(v, dict)]


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


def make_markdown_table(
    json_paths: Sequence[Path],
    names: Optional[Sequence[str]] = None,
    baseline_name: str = "Main",
    metrics_override: Optional[Sequence[str]] = None,
    digits: int = 4,
    alpha: float = 0.05,
    pretty_metric_names: bool = False,
    no_stars: bool = False,
) -> str:
    if names is not None and len(names) != len(json_paths):
        raise SystemExit("--names 的数量必须和输入 JSON 文件数量一致。")

    rows: List[List[str]] = []
    ci_values = []

    for idx, path in enumerate(json_paths):
        data = load_json(path)
        variant_name = names[idx] if names is not None else infer_variant_name(path)
        delta_label = data.get("comparison", {}).get("delta_label", "system_a_minus_system_b")
        metrics = get_metric_list(data, metrics_override)

        for metric in metrics:
            comp_metric = data.get("comparison", {}).get(metric)
            if not isinstance(comp_metric, dict):
                print(f"Warning: metric {metric!r} not found in comparison of {path}", file=sys.stderr)
                continue

            variant_score = get_observed_score(data, "system_a", metric)
            main_score = get_observed_score(data, "system_b", metric)
            delta, ci_lower, ci_upper, prob_main_better = normalize_comparison_to_variant_minus_main(
                comp_metric, delta_label
            )
            p_value = comp_metric.get("p_value_two_sided")
            ci_values.append(comp_metric.get("ci"))

            metric_label_map = PRETTY_METRIC_LABELS if pretty_metric_names else DEFAULT_METRIC_LABELS
            metric_label = metric_label_map.get(metric, metric)

            if delta is None:
                delta_cell = ""
                result = ""
            else:
                stars = "" if no_stars else significance_stars(p_value)
                delta_cell = fmt_float(delta, digits=digits, signed=True) + stars
                result = result_label(float(delta), p_value, alpha=alpha)

            if ci_lower is None or ci_upper is None:
                ci_cell = ""
            else:
                ci_cell = f"[{fmt_float(ci_lower, digits=digits)}, {fmt_float(ci_upper, digits=digits)}]"

            rows.append([
                variant_name,
                metric_label,
                fmt_float(variant_score, digits=digits),
                fmt_float(main_score, digits=digits),
                delta_cell,
                ci_cell,
                fmt_float(prob_main_better, digits=digits),
                fmt_pvalue(p_value, digits=digits),
                result,
            ])

    # Header uses the shared CI level if all rows use the same level, otherwise a generic label.
    ci_values_clean = [v for v in ci_values if v is not None]
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

    # Markdown alignment: left for names/result, right for numeric columns.
    align = [":-------------", ":----------", "-----------:", "--------:", ":---------------", ":--------------------", "------------------:", "-------------:", ":------------"]

    lines = []
    lines.append("| " + " | ".join(md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join(align) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(x) for x in row) + " |")

    return "\n".join(lines)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired-bootstrap JSON files to a Markdown result table."
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more bootstrap result JSON files.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output .md file. If omitted, the table is printed to stdout.",
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
        help="Metric names to include. Default: use the JSON's metrics list.",
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
        "--pretty-metric-names",
        action="store_true",
        help="Use labels like 'Hard F1' instead of raw keys like 'f1_hard'.",
    )
    parser.add_argument(
        "--no-stars",
        action="store_true",
        help="Do not append significance stars to delta values.",
    )
    parser.add_argument(
        "--note",
        action="store_true",
        help="Prepend a short Markdown note explaining Δ and significance stars.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    table = make_markdown_table(
        json_paths=args.json_files,
        names=args.names,
        baseline_name=args.baseline,
        metrics_override=args.metrics,
        digits=args.digits,
        alpha=args.alpha,
        pretty_metric_names=args.pretty_metric_names,
        no_stars=args.no_stars,
    )

    if args.note:
        star_note = "" if args.no_stars else " `* p < .05`, `** p < .01`, `*** p < .001`."
        note = (
            f"Δ = Variant − {args.baseline}. Negative values mean the variant performs worse than {args.baseline}."
            f"{star_note}\n\n"
        )
        table = note + table

    if args.output:
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"Wrote Markdown table to {args.output}", file=sys.stderr)
    else:
        print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
