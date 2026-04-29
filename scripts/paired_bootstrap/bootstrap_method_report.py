#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build Markdown bootstrap reports for selected method comparisons.

This script produces:
- overall single-system CIs and paired comparisons for five F1 metrics
- per-class Avg-F1 single-system CIs and paired comparisons

Bootstrap units are test instances. Pairwise comparisons resample the same
instance ids for both systems.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
VIS_DIR = REPO_ROOT / "scripts" / "visualize"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-hsd-bootstrap")
for import_path in (REPO_ROOT, SRC_DIR, VIS_DIR, SCRIPT_DIR):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import paired_bootstrap_llm as overall_bootstrap
from long_tail_breakdown import (
    expand_by_group,
    f1_from_pr,
    greedy_match_counts,
    hard_match,
    norm_label,
    normalize_quad,
    safe_div,
    soft_match,
)


DEFAULT_METHODS = [
    (
        "SRAG",
        "output/runner/no-rag-1-llamafactory-2ep.json",
    ),
    (
        "DPP",
        "exps/baselines/dpp/exp_8399c5dbba/runner_output/exp_8399c5dbba_s42.json",
    ),
    (
        "Zero-shot",
        "output/runner/method_comparison/general_prompt.json",
    ),
    (
        "Ours",
        "output/runner/simlex5_rag11_multi_class.json",
    ),
]

OVERALL_METRICS = [
    ("f1_target", "Tar-F1"),
    ("f1_hate", "Hate-F1"),
    ("f1_hard", "Hard-F1"),
    ("f1_soft", "Soft-F1"),
    ("f1_avg", "Avg-F1"),
]

DEFAULT_CLASSES = ["Racism", "Sexism", "LGBTQ", "Region", "others", "non-hate"]
DEFAULT_OUTPUT_DIR = Path("output/paired_bootstrap/seed42_selected")


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ClassCounts:
    hard: overall_bootstrap.TupleCounts
    soft: overall_bootstrap.TupleCounts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate bootstrap Markdown reports for selected methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["overall", "per-class-avg", "both"],
        default="both",
        help="Report mode to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Markdown and JSON outputs.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Method specification. Repeat to override defaults. "
            "Order controls pairwise comparisons."
        ),
    )
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class list for per-class Avg-F1 mode.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Percentile confidence interval width.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Soft-match threshold for target and argument.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Decimal places in Markdown tables.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact output.",
    )
    return parser.parse_args()


def parse_methods(raw_methods: Sequence[str] | None) -> list[MethodSpec]:
    if not raw_methods:
        return [MethodSpec(name=name, path=Path(path)) for name, path in DEFAULT_METHODS]

    methods: list[MethodSpec] = []
    for raw in raw_methods:
        if "=" not in raw:
            raise ValueError(f"--method must use NAME=PATH format: {raw!r}")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"--method must use NAME=PATH format: {raw!r}")
        methods.append(MethodSpec(name=name, path=Path(path)))

    if len(methods) < 2:
        raise ValueError("At least two methods are required.")
    return methods


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def validate_args(args: argparse.Namespace) -> None:
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if not 0 < args.ci < 100:
        raise ValueError("--ci must be in the open interval (0, 100).")
    if not 0 <= args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be between 0 and 1.")
    if args.digits < 0:
        raise ValueError("--digits must be non-negative.")


def load_methods(methods: Sequence[MethodSpec]) -> dict[str, overall_bootstrap.SystemData]:
    loaded: dict[str, overall_bootstrap.SystemData] = {}
    seen_names: set[str] = set()
    for method in methods:
        if method.name in seen_names:
            raise ValueError(f"Duplicate method name: {method.name}")
        seen_names.add(method.name)
        loaded[method.name] = overall_bootstrap.load_system(resolve_path(method.path))
    return loaded


def common_ordered_ids(
    methods: Sequence[MethodSpec],
    systems: dict[str, overall_bootstrap.SystemData],
) -> list[str]:
    first = systems[methods[0].name]
    first_ids = set(first.ids)
    for method in methods[1:]:
        current = systems[method.name]
        current_ids = set(current.ids)
        if current_ids != first_ids:
            only_first = sorted(first_ids - current_ids)[:10]
            only_current = sorted(current_ids - first_ids)[:10]
            raise ValueError(
                f"Method id sets differ: {methods[0].name} vs {method.name}. "
                f"Only in {methods[0].name}: {only_first}; "
                f"only in {method.name}: {only_current}."
            )
    return list(first.ids)


def sample_indices(n: int, n_bootstrap: int, rng: random.Random) -> Iterable[list[int]]:
    for _ in range(n_bootstrap):
        yield [rng.randrange(n) for _ in range(n)]


def metric_summary(scores: Sequence[float], observed: float, ci: float) -> dict[str, float]:
    sorted_scores = sorted(scores)
    alpha = (100.0 - ci) / 2.0
    return {
        "observed": observed,
        "mean": mean(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "ci": ci,
        "ci_lower": overall_bootstrap.percentile(sorted_scores, alpha),
        "ci_upper": overall_bootstrap.percentile(sorted_scores, 100.0 - alpha),
    }


def paired_summary(deltas: Sequence[float], observed_delta: float, ci: float) -> dict[str, float]:
    summary = metric_summary(deltas, observed_delta, ci)
    prob_delta_gt_0 = sum(delta > 0 for delta in deltas) / len(deltas)
    prob_delta_ge_0 = sum(delta >= 0 for delta in deltas) / len(deltas)
    prob_delta_le_0 = sum(delta <= 0 for delta in deltas) / len(deltas)
    p_value = min(1.0, 2.0 * min(prob_delta_le_0, prob_delta_ge_0))
    return {
        "observed_delta": summary["observed"],
        "mean_delta": summary["mean"],
        "std_delta": summary["std"],
        "ci": summary["ci"],
        "ci_lower": summary["ci_lower"],
        "ci_upper": summary["ci_upper"],
        "prob_a_better": prob_delta_gt_0,
        "p_value_two_sided": p_value,
    }


def bh_fdr(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running_min = 1.0
    for rank, (idx, p_value) in reversed(list(enumerate(indexed, start=1))):
        candidate = min(1.0, p_value * n / rank)
        running_min = min(running_min, candidate)
        adjusted[idx] = running_min
    return adjusted


def add_q_values(rows: list[dict]) -> None:
    q_values = bh_fdr([float(row["p_value_two_sided"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["q_value_bh"] = q_value


def stars(q_value: float) -> str:
    if q_value < 0.001:
        return "***"
    if q_value < 0.01:
        return "**"
    if q_value < 0.05:
        return "*"
    return ""


def result_label(delta: float, q_value: float) -> str:
    if q_value >= 0.05:
        return "n.s."
    if delta > 0:
        return "A better"
    if delta < 0:
        return "B better"
    return "tie"


def fmt_float(value: float | None, digits: int, signed: bool = False) -> str:
    if value is None:
        return ""
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}"


def fmt_pvalue(value: float, digits: int) -> str:
    threshold = 10 ** (-digits)
    if 0 < value < threshold:
        return f"<{threshold:.{digits}f}"
    return f"{value:.{digits}f}"


def md_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", r"\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], align: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(md_escape(header) for header in headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict, indent: int) -> None:
    json_indent = None if indent == 0 else indent
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=json_indent) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_observed_against_payload(
    systems: dict[str, overall_bootstrap.SystemData],
    observed: dict[str, dict[str, float]],
    tolerance: float = 1e-3,
) -> list[dict]:
    metric_paths = {
        "f1_hard": ("f1_hard",),
        "f1_soft": ("f1_soft",),
        "f1_avg": ("f1_avg",),
        "f1_target": ("field_metrics", "targeted_group", "f1"),
        "f1_hate": ("field_metrics", "hateful", "f1"),
    }
    validations: list[dict] = []
    failures: list[str] = []

    for method_name, system in systems.items():
        with system.path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload_metric = payload.get("metric")
        if not isinstance(payload_metric, dict):
            continue
        for metric, path_items in metric_paths.items():
            expected = payload_metric
            for item in path_items:
                expected = expected[item]
            actual = observed[method_name][metric]
            diff = abs(float(expected) - actual)
            item = {
                "method": method_name,
                "metric": metric,
                "payload_value": float(expected),
                "computed_value": actual,
                "absolute_difference": diff,
                "tolerance": tolerance,
                "passed": diff <= tolerance,
            }
            validations.append(item)
            if not item["passed"]:
                failures.append(
                    f"{method_name} {metric}: payload={expected}, computed={actual}, diff={diff}"
                )

    if failures:
        raise ValueError(
            "Computed overall metrics differ from runner payload beyond tolerance: "
            + "; ".join(failures)
        )
    return validations


def build_overall_report(
    methods: Sequence[MethodSpec],
    systems: dict[str, overall_bootstrap.SystemData],
    ids: Sequence[str],
    n_bootstrap: int,
    seed: int,
    ci: float,
    similarity_threshold: float,
    digits: int,
) -> tuple[str, dict]:
    metric_keys = [metric for metric, _ in OVERALL_METRICS]
    method_counts = {
        method.name: overall_bootstrap.compute_counts_for_ids(
            systems[method.name],
            ids,
            similarity_threshold,
        )
        for method in methods
    }
    observed = {
        method.name: overall_bootstrap.observed_metrics(method_counts[method.name])
        for method in methods
    }

    validate_items = validate_observed_against_payload(systems, observed)

    single_scores = {
        method.name: {metric: [] for metric in metric_keys}
        for method in methods
    }
    pair_rows: list[dict] = []
    pair_deltas: dict[tuple[str, str], dict[str, list[float]]] = {}

    for method_a, method_b in combinations(methods, 2):
        pair_deltas[(method_a.name, method_b.name)] = {metric: [] for metric in metric_keys}

    rng = random.Random(seed)
    n = len(ids)
    for indices in sample_indices(n, n_bootstrap, rng):
        sampled = {
            method.name: overall_bootstrap.scores_from_selected_counts(
                method_counts[method.name],
                indices,
            )
            for method in methods
        }
        for method in methods:
            for metric in metric_keys:
                single_scores[method.name][metric].append(sampled[method.name][metric])
        for method_a, method_b in combinations(methods, 2):
            for metric in metric_keys:
                pair_deltas[(method_a.name, method_b.name)][metric].append(
                    sampled[method_a.name][metric] - sampled[method_b.name][metric]
                )

    single_summary = {
        method.name: {
            metric: metric_summary(
                single_scores[method.name][metric],
                observed[method.name][metric],
                ci,
            )
            for metric in metric_keys
        }
        for method in methods
    }

    for method_a, method_b in combinations(methods, 2):
        for metric in metric_keys:
            delta = observed[method_a.name][metric] - observed[method_b.name][metric]
            summary = paired_summary(
                pair_deltas[(method_a.name, method_b.name)][metric],
                delta,
                ci,
            )
            pair_rows.append(
                {
                    "method_a": method_a.name,
                    "method_b": method_b.name,
                    "metric": metric,
                    "metric_label": dict(OVERALL_METRICS)[metric],
                    "score_a": observed[method_a.name][metric],
                    "score_b": observed[method_b.name][metric],
                    **summary,
                }
            )

    add_q_values(pair_rows)
    payload = {
        "mode": "overall",
        "methods": [
            {
                "name": method.name,
                "path": relative_path(systems[method.name].path),
                "n_instances": systems[method.name].total,
                "success": systems[method.name].success,
                "success_rate": systems[method.name].success / systems[method.name].total,
            }
            for method in methods
        ],
        "metrics": [{"key": key, "label": label} for key, label in OVERALL_METRICS],
        "n_instances": len(ids),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci": ci,
        "ci_method": "percentile",
        "similarity_threshold": similarity_threshold,
        "p_adjustment": "BH-FDR over all overall pairwise metric tests",
        "single_system": single_summary,
        "pairwise": pair_rows,
        "validation": {"overall_metric_payload_check": validate_items},
    }

    md = render_overall_markdown(payload, methods, digits)
    return md, payload


def per_class_counts_for_row(
    row: dict,
    classes: Sequence[str],
    soft_threshold: float,
) -> dict[str, ClassCounts]:
    cls_norm = [norm_label(c) for c in classes]
    cls_norm_set = set(cls_norm)

    gt_by: dict[str, list] = {c: [] for c in cls_norm}
    pred_by: dict[str, list] = {c: [] for c in cls_norm}

    for q in row.get("gt_quadruples", []) or []:
        for expanded in expand_by_group(normalize_quad(q)):
            if expanded.targeted_group in cls_norm_set:
                gt_by[expanded.targeted_group].append(expanded)

    for q in row.get("pred_quadruples", []) or []:
        for expanded in expand_by_group(normalize_quad(q)):
            if expanded.targeted_group in cls_norm_set:
                pred_by[expanded.targeted_group].append(expanded)

    out: dict[str, ClassCounts] = {}
    for class_name, class_norm in zip(classes, cls_norm):
        hard_tp, hard_fp, hard_fn = greedy_match_counts(
            pred_by[class_norm],
            gt_by[class_norm],
            hard_match,
        )
        soft_tp, soft_fp, soft_fn = greedy_match_counts(
            pred_by[class_norm],
            gt_by[class_norm],
            lambda pred, gold: soft_match(pred, gold, threshold=soft_threshold),
        )
        out[class_name] = ClassCounts(
            hard=overall_bootstrap.TupleCounts(hard_tp, hard_fp, hard_fn),
            soft=overall_bootstrap.TupleCounts(soft_tp, soft_fp, soft_fn),
        )
    return out


def compute_per_class_counts_for_ids(
    system: overall_bootstrap.SystemData,
    ids: Sequence[str],
    classes: Sequence[str],
    soft_threshold: float,
) -> list[dict[str, ClassCounts]]:
    return [
        per_class_counts_for_row(
            system.results_by_id[instance_id],
            classes,
            soft_threshold,
        )
        for instance_id in ids
    ]


def sum_tuple_counts(counts: Iterable[overall_bootstrap.TupleCounts]) -> overall_bootstrap.TupleCounts:
    return overall_bootstrap.sum_counts(counts)


def class_avg_f1_from_counts(class_counts: Sequence[dict[str, ClassCounts]], class_name: str) -> float:
    hard = overall_bootstrap.prf(sum_tuple_counts(item[class_name].hard for item in class_counts))
    soft = overall_bootstrap.prf(sum_tuple_counts(item[class_name].soft for item in class_counts))
    return (hard["f1"] + soft["f1"]) / 2


def class_avg_f1_from_indices(
    class_counts: Sequence[dict[str, ClassCounts]],
    class_name: str,
    indices: Sequence[int],
) -> float:
    hard = overall_bootstrap.prf(sum_tuple_counts(class_counts[i][class_name].hard for i in indices))
    soft = overall_bootstrap.prf(sum_tuple_counts(class_counts[i][class_name].soft for i in indices))
    return (hard["f1"] + soft["f1"]) / 2


def build_per_class_report(
    methods: Sequence[MethodSpec],
    systems: dict[str, overall_bootstrap.SystemData],
    ids: Sequence[str],
    classes: Sequence[str],
    n_bootstrap: int,
    seed: int,
    ci: float,
    similarity_threshold: float,
    digits: int,
) -> tuple[str, dict]:
    method_counts = {
        method.name: compute_per_class_counts_for_ids(
            systems[method.name],
            ids,
            classes,
            similarity_threshold,
        )
        for method in methods
    }
    observed = {
        method.name: {
            class_name: class_avg_f1_from_counts(method_counts[method.name], class_name)
            for class_name in classes
        }
        for method in methods
    }
    single_scores = {
        method.name: {class_name: [] for class_name in classes}
        for method in methods
    }
    pair_deltas: dict[tuple[str, str], dict[str, list[float]]] = {}
    for method_a, method_b in combinations(methods, 2):
        pair_deltas[(method_a.name, method_b.name)] = {class_name: [] for class_name in classes}

    rng = random.Random(seed)
    n = len(ids)
    for indices in sample_indices(n, n_bootstrap, rng):
        sampled = {
            method.name: {
                class_name: class_avg_f1_from_indices(
                    method_counts[method.name],
                    class_name,
                    indices,
                )
                for class_name in classes
            }
            for method in methods
        }
        for method in methods:
            for class_name in classes:
                single_scores[method.name][class_name].append(sampled[method.name][class_name])
        for method_a, method_b in combinations(methods, 2):
            for class_name in classes:
                pair_deltas[(method_a.name, method_b.name)][class_name].append(
                    sampled[method_a.name][class_name] - sampled[method_b.name][class_name]
                )

    single_summary = {
        method.name: {
            class_name: metric_summary(
                single_scores[method.name][class_name],
                observed[method.name][class_name],
                ci,
            )
            for class_name in classes
        }
        for method in methods
    }

    pair_rows: list[dict] = []
    for method_a, method_b in combinations(methods, 2):
        for class_name in classes:
            delta = observed[method_a.name][class_name] - observed[method_b.name][class_name]
            summary = paired_summary(
                pair_deltas[(method_a.name, method_b.name)][class_name],
                delta,
                ci,
            )
            pair_rows.append(
                {
                    "method_a": method_a.name,
                    "method_b": method_b.name,
                    "class": class_name,
                    "score_a": observed[method_a.name][class_name],
                    "score_b": observed[method_b.name][class_name],
                    **summary,
                }
            )

    add_q_values(pair_rows)
    payload = {
        "mode": "per-class-avg",
        "methods": [
            {
                "name": method.name,
                "path": relative_path(systems[method.name].path),
                "n_instances": systems[method.name].total,
                "success": systems[method.name].success,
                "success_rate": systems[method.name].success / systems[method.name].total,
            }
            for method in methods
        ],
        "classes": list(classes),
        "metric": "Avg-F1",
        "n_instances": len(ids),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "ci": ci,
        "ci_method": "percentile",
        "similarity_threshold": similarity_threshold,
        "p_adjustment": "BH-FDR over all per-class pairwise tests",
        "single_system": single_summary,
        "pairwise": pair_rows,
    }

    md = render_per_class_markdown(payload, methods, classes, digits)
    return md, payload


def ci_cell(summary: dict, digits: int) -> str:
    return (
        f"{fmt_float(summary['observed'], digits)} "
        f"[{fmt_float(summary['ci_lower'], digits)}, {fmt_float(summary['ci_upper'], digits)}]"
    )


def delta_cell(row: dict, digits: int) -> str:
    return fmt_float(row["observed_delta"], digits, signed=True) + stars(row["q_value_bh"])


def source_note(payload: dict) -> str:
    rows = [
        f"- {item['name']}: `{item['path']}` "
        f"(n={item['n_instances']}, success={item['success']})"
        for item in payload["methods"]
    ]
    return "\n".join(rows)


def render_overall_markdown(payload: dict, methods: Sequence[MethodSpec], digits: int) -> str:
    metric_labels = {item["key"]: item["label"] for item in payload["metrics"]}
    metric_keys = [item["key"] for item in payload["metrics"]]
    parts = [
        "# Overall Bootstrap CI and Pairwise Tests",
        "",
        (
            f"Bootstrap units are test instances. n_bootstrap={payload['n_bootstrap']}, "
            f"seed={payload['seed']}, CI={payload['ci']}%, "
            f"soft threshold={payload['similarity_threshold']}. "
            "Pairwise p values are adjusted with BH-FDR; stars use q values."
        ),
        "",
        "## Sources",
        source_note(payload),
        "",
        "## Table 1. Single-system F1 with 95% CI",
    ]

    headers = ["Method"] + [metric_labels[metric] for metric in metric_keys]
    align = [":----------"] + ["----------------:" for _ in metric_keys]
    rows = []
    for method in methods:
        rows.append(
            [method.name]
            + [
                ci_cell(payload["single_system"][method.name][metric], digits)
                for metric in metric_keys
            ]
        )
    parts.append(markdown_table(headers, align, rows))

    parts.extend(["", "## Table 2. Paired bootstrap pairwise tests"])
    headers = [
        "Method A",
        "Method B",
        "Metric",
        "A F1",
        "B F1",
        "Δ A−B",
        "95% CI of Δ",
        "P(A>B)",
        "p(raw)",
        "q(BH-FDR)",
        "Result",
    ]
    align = [
        ":---------",
        ":---------",
        ":--------",
        "-----:",
        "-----:",
        "------:",
        ":-----------",
        "------:",
        "------:",
        "--------:",
        ":------",
    ]
    rows = []
    for row in payload["pairwise"]:
        rows.append(
            [
                row["method_a"],
                row["method_b"],
                row["metric_label"],
                fmt_float(row["score_a"], digits),
                fmt_float(row["score_b"], digits),
                delta_cell(row, digits),
                f"[{fmt_float(row['ci_lower'], digits)}, {fmt_float(row['ci_upper'], digits)}]",
                fmt_float(row["prob_a_better"], digits),
                fmt_pvalue(row["p_value_two_sided"], digits),
                fmt_pvalue(row["q_value_bh"], digits),
                result_label(row["observed_delta"], row["q_value_bh"]),
            ]
        )
    parts.append(markdown_table(headers, align, rows))
    parts.extend(["", "`* q < .05`, `** q < .01`, `*** q < .001`."])
    return "\n".join(parts)


def render_per_class_markdown(
    payload: dict,
    methods: Sequence[MethodSpec],
    classes: Sequence[str],
    digits: int,
) -> str:
    parts = [
        "# Per-class Avg-F1 Bootstrap CI and Pairwise Tests",
        "",
        (
            f"Bootstrap units are test instances. n_bootstrap={payload['n_bootstrap']}, "
            f"seed={payload['seed']}, CI={payload['ci']}%, "
            f"soft threshold={payload['similarity_threshold']}. "
            "Pairwise p values are adjusted with BH-FDR; stars use q values."
        ),
        "",
        "## Sources",
        source_note(payload),
        "",
        "## Table 1. Single-system per-class Avg-F1 with 95% CI",
    ]
    headers = ["Method"] + list(classes)
    align = [":----------"] + ["----------------:" for _ in classes]
    rows = []
    for method in methods:
        rows.append(
            [method.name]
            + [
                ci_cell(payload["single_system"][method.name][class_name], digits)
                for class_name in classes
            ]
        )
    parts.append(markdown_table(headers, align, rows))

    parts.extend(["", "## Table 2. Paired bootstrap per-class Avg-F1 tests"])
    headers = [
        "Method A",
        "Method B",
        "Class",
        "A Avg-F1",
        "B Avg-F1",
        "Δ A−B",
        "95% CI of Δ",
        "P(A>B)",
        "p(raw)",
        "q(BH-FDR)",
        "Result",
    ]
    align = [
        ":---------",
        ":---------",
        ":--------",
        "--------:",
        "--------:",
        "------:",
        ":-----------",
        "------:",
        "------:",
        "--------:",
        ":------",
    ]
    rows = []
    for row in payload["pairwise"]:
        rows.append(
            [
                row["method_a"],
                row["method_b"],
                row["class"],
                fmt_float(row["score_a"], digits),
                fmt_float(row["score_b"], digits),
                delta_cell(row, digits),
                f"[{fmt_float(row['ci_lower'], digits)}, {fmt_float(row['ci_upper'], digits)}]",
                fmt_float(row["prob_a_better"], digits),
                fmt_pvalue(row["p_value_two_sided"], digits),
                fmt_pvalue(row["q_value_bh"], digits),
                result_label(row["observed_delta"], row["q_value_bh"]),
            ]
        )
    parts.append(markdown_table(headers, align, rows))
    parts.extend(["", "`* q < .05`, `** q < .01`, `*** q < .001`."])
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    validate_args(args)
    methods = parse_methods(args.method)
    systems = load_methods(methods)
    ids = common_ordered_ids(methods, systems)
    classes = [item.strip() for item in args.classes.split(",") if item.strip()]
    if not classes:
        raise ValueError("--classes must not be empty.")

    output_dir = resolve_path(args.output_dir)
    modes = ["overall", "per-class-avg"] if args.mode == "both" else [args.mode]

    if "overall" in modes:
        markdown, payload = build_overall_report(
            methods=methods,
            systems=systems,
            ids=ids,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            ci=args.ci,
            similarity_threshold=args.similarity_threshold,
            digits=args.digits,
        )
        write_text(output_dir / "overall_ci_pairwise.md", markdown)
        write_json(output_dir / "overall_ci_pairwise.json", payload, args.indent)

    if "per-class-avg" in modes:
        markdown, payload = build_per_class_report(
            methods=methods,
            systems=systems,
            ids=ids,
            classes=classes,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            ci=args.ci,
            similarity_threshold=args.similarity_threshold,
            digits=args.digits,
        )
        write_text(output_dir / "per_class_avg_f1_ci_pairwise.md", markdown)
        write_json(output_dir / "per_class_avg_f1_ci_pairwise.json", payload, args.indent)


if __name__ == "__main__":
    main()
