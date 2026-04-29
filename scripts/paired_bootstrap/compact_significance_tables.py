#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Render compact significance tables from bootstrap report JSON files."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Sequence


DEFAULT_INPUT_DIR = Path("output/paired_bootstrap/seed42_selected")
DEFAULT_OVERALL_JSON = DEFAULT_INPUT_DIR / "overall_ci_pairwise.json"
DEFAULT_PER_CLASS_JSON = DEFAULT_INPUT_DIR / "per_class_avg_f1_ci_pairwise.json"
DEFAULT_OVERALL_MD = DEFAULT_INPUT_DIR / "overall_compact_significance.md"
DEFAULT_PER_CLASS_MD = DEFAULT_INPUT_DIR / "per_class_avg_f1_compact_significance.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create compact letter display Markdown tables from bootstrap JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--overall-json", type=Path, default=DEFAULT_OVERALL_JSON)
    parser.add_argument("--per-class-json", type=Path, default=DEFAULT_PER_CLASS_JSON)
    parser.add_argument("--overall-output", type=Path, default=DEFAULT_OVERALL_MD)
    parser.add_argument("--per-class-output", type=Path, default=DEFAULT_PER_CLASS_MD)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--digits", type=int, default=4)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_float(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


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


def method_order(data: dict) -> list[str]:
    return [item["name"] for item in data["methods"]]


def q_lookup(rows: Sequence[dict], facet_key: str, facet_value: str) -> dict[frozenset[str], float]:
    out: dict[frozenset[str], float] = {}
    for row in rows:
        if row[facet_key] != facet_value:
            continue
        out[frozenset((row["method_a"], row["method_b"]))] = float(row["q_value_bh"])
    return out


def is_significant(
    method_a: str,
    method_b: str,
    q_values: dict[frozenset[str], float],
    alpha: float,
) -> bool:
    if method_a == method_b:
        return False
    return q_values[frozenset((method_a, method_b))] < alpha


def all_subsets(items: Sequence[str]) -> list[frozenset[str]]:
    out: list[frozenset[str]] = []
    n = len(items)
    for mask in range(1, 1 << n):
        subset = frozenset(items[idx] for idx in range(n) if mask & (1 << idx))
        out.append(subset)
    return out


def is_nonsig_clique(
    subset: frozenset[str],
    q_values: dict[frozenset[str], float],
    alpha: float,
) -> bool:
    return all(
        not is_significant(method_a, method_b, q_values, alpha)
        for method_a, method_b in combinations(sorted(subset), 2)
    )


def covers_required_pairs(
    selected: Sequence[frozenset[str]],
    methods: Sequence[str],
    q_values: dict[frozenset[str], float],
    alpha: float,
) -> bool:
    for method in methods:
        if not any(method in subset for subset in selected):
            return False
    for method_a, method_b in combinations(methods, 2):
        if is_significant(method_a, method_b, q_values, alpha):
            continue
        if not any(method_a in subset and method_b in subset for subset in selected):
            return False
    return True


def compact_letters(
    methods: Sequence[str],
    scores: dict[str, float],
    q_values: dict[frozenset[str], float],
    alpha: float,
) -> dict[str, str]:
    """Assign CLD letters.

    Methods that share at least one letter are not significantly different;
    significantly different methods share no letters.
    """
    ranked_methods = sorted(methods, key=lambda method: (-scores[method], methods.index(method)))
    candidates = [
        subset
        for subset in all_subsets(ranked_methods)
        if is_nonsig_clique(subset, q_values, alpha)
    ]
    candidates.sort(
        key=lambda subset: (
            -len(subset),
            min(ranked_methods.index(method) for method in subset),
            [ranked_methods.index(method) for method in sorted(subset, key=ranked_methods.index)],
        )
    )

    best: tuple[frozenset[str], ...] | None = None
    best_key: tuple | None = None
    for size in range(1, len(candidates) + 1):
        for selected in combinations(candidates, size):
            if not covers_required_pairs(selected, ranked_methods, q_values, alpha):
                continue
            selected = tuple(selected)
            score_key = (
                size,
                -sum(len(subset) for subset in selected),
                [
                    min(ranked_methods.index(method) for method in subset)
                    for subset in selected
                ],
            )
            if best_key is None or score_key < best_key:
                best = selected
                best_key = score_key
        if best is not None:
            break

    if best is None:
        raise ValueError("Could not assign compact letters.")

    ordered_groups = sorted(
        best,
        key=lambda subset: (
            min(ranked_methods.index(method) for method in subset),
            -len(subset),
        )
    )
    letters = "abcdefghijklmnopqrstuvwxyz"
    if len(ordered_groups) > len(letters):
        raise ValueError("Too many compact letter groups.")

    method_letters = {method: "" for method in methods}
    for idx, group in enumerate(ordered_groups):
        letter = letters[idx]
        for method in group:
            method_letters[method] += letter
    return method_letters


def format_ci_cell(summary: dict, letters: str, is_top_group: bool, digits: int) -> str:
    cell = (
        f"{fmt_float(summary['observed'], digits)} "
        f"[{fmt_float(summary['ci_lower'], digits)}, {fmt_float(summary['ci_upper'], digits)}]"
        f"<sup>{letters}</sup>"
    )
    return f"**{cell}**" if is_top_group else cell


def source_lines(data: dict) -> str:
    lines = []
    for item in data["methods"]:
        lines.append(
            f"- {item['name']}: `{item['path']}` "
            f"(n={item['n_instances']}, success={item['success']})"
        )
    return "\n".join(lines)


def render_overall(data: dict, alpha: float, digits: int) -> str:
    methods = method_order(data)
    metrics = [(item["key"], item["label"]) for item in data["metrics"]]
    headers = ["Method"] + [label for _, label in metrics]
    align = [":----------"] + ["----------------:" for _ in metrics]
    per_metric_letters: dict[str, dict[str, str]] = {}

    for metric_key, _ in metrics:
        scores = {
            method: data["single_system"][method][metric_key]["observed"]
            for method in methods
        }
        per_metric_letters[metric_key] = compact_letters(
            methods,
            scores,
            q_lookup(data["pairwise"], "metric", metric_key),
            alpha,
        )

    rows = []
    for method in methods:
        row = [method]
        for metric_key, _ in metrics:
            letters = per_metric_letters[metric_key][method]
            row.append(
                format_ci_cell(
                    data["single_system"][method][metric_key],
                    letters,
                    "a" in letters,
                    digits,
                )
            )
        rows.append(row)

    return "\n".join(
        [
            "# Compact Overall Significance Table",
            "",
            (
                f"Values are F1 with {data['ci']}% bootstrap CI. "
                f"Letters are compact significance groups within each metric "
                f"using BH-FDR q < {alpha}; methods sharing a letter are not "
                "significantly different. Bold marks the top significance group."
            ),
            "",
            "## Sources",
            source_lines(data),
            "",
            markdown_table(headers, align, rows),
        ]
    )


def render_per_class(data: dict, alpha: float, digits: int) -> str:
    methods = method_order(data)
    classes = list(data["classes"])
    headers = ["Method"] + classes
    align = [":----------"] + ["----------------:" for _ in classes]
    per_class_letters: dict[str, dict[str, str]] = {}

    for class_name in classes:
        scores = {
            method: data["single_system"][method][class_name]["observed"]
            for method in methods
        }
        per_class_letters[class_name] = compact_letters(
            methods,
            scores,
            q_lookup(data["pairwise"], "class", class_name),
            alpha,
        )

    rows = []
    for method in methods:
        row = [method]
        for class_name in classes:
            letters = per_class_letters[class_name][method]
            row.append(
                format_ci_cell(
                    data["single_system"][method][class_name],
                    letters,
                    "a" in letters,
                    digits,
                )
            )
        rows.append(row)

    return "\n".join(
        [
            "# Compact Per-class Avg-F1 Significance Table",
            "",
            (
                f"Values are per-class Avg-F1 with {data['ci']}% bootstrap CI. "
                f"Letters are compact significance groups within each class "
                f"using BH-FDR q < {alpha}; methods sharing a letter are not "
                "significantly different. Bold marks the top significance group."
            ),
            "",
            "## Sources",
            source_lines(data),
            "",
            markdown_table(headers, align, rows),
        ]
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    overall_data = load_json(args.overall_json)
    per_class_data = load_json(args.per_class_json)
    write_text(args.overall_output, render_overall(overall_data, args.alpha, args.digits))
    write_text(args.per_class_output, render_per_class(per_class_data, args.alpha, args.digits))


if __name__ == "__main__":
    main()
