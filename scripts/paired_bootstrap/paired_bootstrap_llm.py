#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Paired bootstrap over LLM runner test instances.

The evaluation logic mirrors metrics/core.py and metrics/metric_llm.py:
- hard match: exact tuple match after convert_quad
- soft match: same targeted_group and hateful, plus target/argument similarity
- avg F1: arithmetic mean of hard micro-F1 and soft micro-F1
- target F1: targeted_group field F1 after target/argument based alignment
- hate F1: hateful field F1 after target/argument based alignment

Bootstrap units are test instances, not individual predicted tuples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.core import (
    align_elements,
    convert_quad,
    is_hard_match,
    is_soft_match,
    preprocess_quad,
)


METRIC_ALIASES = {
    "all": "all",
    "hard_f1": "f1_hard",
    "f1_hard": "f1_hard",
    "soft_f1": "f1_soft",
    "f1_soft": "f1_soft",
    "avg_f1": "f1_avg",
    "f1_avg": "f1_avg",
    "target_f1": "f1_target",
    "targeted_group_f1": "f1_target",
    "tg_f1": "f1_target",
    "f1_target": "f1_target",
    "hate_f1": "f1_hate",
    "hateful_f1": "f1_hate",
    "f1_hate": "f1_hate",
}

DEFAULT_METRICS = ["f1_hard", "f1_soft", "f1_avg", "f1_target", "f1_hate"]


@dataclass(frozen=True)
class TupleCounts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: "TupleCounts") -> "TupleCounts":
        return TupleCounts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )


@dataclass(frozen=True)
class InstanceCounts:
    hard: TupleCounts
    soft: TupleCounts
    target: TupleCounts
    hate: TupleCounts


@dataclass(frozen=True)
class SystemData:
    path: Path
    results_by_id: dict[str, dict]
    ids: list[str]
    success: int
    total: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap over LLM runner test instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--system-a",
        required=True,
        type=Path,
        help="Runner output JSON for the first system.",
    )
    parser.add_argument(
        "--system-b",
        type=Path,
        default=None,
        help="Optional runner output JSON for the second system. If omitted, run single-system CI.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        help=(
            "Metric(s) to bootstrap. Use 'all' or comma-separated values. "
            "Aliases: f1_hard/hard_f1, f1_soft/soft_f1, f1_avg/avg_f1, "
            "f1_target/target_f1/targeted_group_f1, f1_hate/hate_f1/hateful_f1."
        ),
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
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Soft-match threshold for both target and argument similarity.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Percentile confidence interval width.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON result. The same JSON is printed to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact single-line JSON.",
    )
    return parser.parse_args()


def canonical_metrics(metric_arg: str) -> list[str]:
    raw_metrics = [
        item.strip()
        for item in metric_arg.split(",")
        if item.strip()
    ]
    if not raw_metrics:
        raise ValueError("--metric must not be empty.")

    if any(item == "all" for item in raw_metrics):
        if len(raw_metrics) > 1:
            raise ValueError("--metric all cannot be combined with other metrics.")
        return list(DEFAULT_METRICS)

    metrics: list[str] = []
    for raw_metric in raw_metrics:
        try:
            metric = METRIC_ALIASES[raw_metric]
        except KeyError as exc:
            supported = ", ".join(sorted(METRIC_ALIASES))
            raise ValueError(
                f"Unsupported metric: {raw_metric}. Supported values: {supported}"
            ) from exc
        if metric not in metrics:
            metrics.append(metric)
    return metrics


def validate_args(args: argparse.Namespace) -> None:
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive.")
    if not 0 < args.ci < 100:
        raise ValueError("--ci must be in the open interval (0, 100).")
    if args.similarity_threshold < 0 or args.similarity_threshold > 1:
        raise ValueError("--similarity-threshold must be between 0 and 1.")


def load_system(path: Path) -> SystemData:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path} must contain a top-level list field named 'results'.")

    results_by_id: dict[str, dict] = {}
    ids: list[str] = []
    success = 0

    for idx, row in enumerate(results):
        if "id" not in row:
            raise ValueError(f"{path} results[{idx}] is missing 'id'.")
        instance_id = str(row["id"])
        if instance_id in results_by_id:
            raise ValueError(f"{path} contains duplicate id {instance_id!r}.")
        if "gt_quadruples" not in row:
            raise ValueError(f"{path} results[{idx}] is missing 'gt_quadruples'.")
        if "pred_quadruples" not in row:
            raise ValueError(f"{path} results[{idx}] is missing 'pred_quadruples'.")
        results_by_id[instance_id] = row
        ids.append(instance_id)
        if row.get("status") == "success":
            success += 1

    return SystemData(
        path=path,
        results_by_id=results_by_id,
        ids=ids,
        success=success,
        total=len(ids),
    )


def match_counts(
    pred_quads: Sequence[dict],
    gt_quads: Sequence[dict],
    preprocess: Callable[[dict], object],
    match: Callable[[object, object], bool],
) -> TupleCounts:
    preds = [preprocess(q) for q in pred_quads]
    golds = [preprocess(q) for q in gt_quads]

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()

    for pred_idx, pred_quad in enumerate(preds):
        for gold_idx, gold_quad in enumerate(golds):
            if gold_idx in matched_gold:
                continue
            if match(pred_quad, gold_quad):
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)
                break

    tp = len(matched_pred)
    fp = len(preds) - tp
    fn = len(golds) - len(matched_gold)
    return TupleCounts(tp=tp, fp=fp, fn=fn)


def compute_instance_counts(row: dict, similarity_threshold: float) -> InstanceCounts:
    pred_quads = row["pred_quadruples"]
    gt_quads = row["gt_quadruples"]

    hard = match_counts(
        pred_quads,
        gt_quads,
        preprocess=convert_quad,
        match=is_hard_match,
    )
    soft = match_counts(
        pred_quads,
        gt_quads,
        preprocess=preprocess_quad,
        match=lambda pred, gold: is_soft_match(pred, gold, similarity_threshold),
    )
    target, hate = field_counts(pred_quads, gt_quads)
    return InstanceCounts(hard=hard, soft=soft, target=target, hate=hate)


def field_counts(
    pred_quads: Sequence[dict],
    gt_quads: Sequence[dict],
) -> tuple[TupleCounts, TupleCounts]:
    pred_processed = [preprocess_quad(q) for q in pred_quads]
    gt_processed = [preprocess_quad(q) for q in gt_quads]
    matches = align_elements(pred_processed, gt_processed)

    target_tp = target_fp = target_fn = 0
    hate_tp = hate_fp = hate_fn = 0

    for pred_idx, gt_idx in matches:
        pred_quad = pred_processed[pred_idx]
        gt_quad = gt_processed[gt_idx]

        pred_target_group = ",".join(sorted(pred_quad["targeted_group"]))
        gt_target_group = ",".join(sorted(gt_quad["targeted_group"]))

        if pred_target_group == gt_target_group:
            target_tp += 1
        else:
            target_fp += 1
            target_fn += 1

        if pred_quad["hateful"] == gt_quad["hateful"]:
            hate_tp += 1
        else:
            hate_fp += 1
            hate_fn += 1

    matched_pred = {pred_idx for pred_idx, _ in matches}
    matched_gt = {gt_idx for _, gt_idx in matches}

    target_fp += sum(1 for idx in range(len(pred_processed)) if idx not in matched_pred)
    hate_fp += sum(1 for idx in range(len(pred_processed)) if idx not in matched_pred)
    target_fn += sum(1 for idx in range(len(gt_processed)) if idx not in matched_gt)
    hate_fn += sum(1 for idx in range(len(gt_processed)) if idx not in matched_gt)

    return (
        TupleCounts(tp=target_tp, fp=target_fp, fn=target_fn),
        TupleCounts(tp=hate_tp, fp=hate_fp, fn=hate_fn),
    )


def compute_counts_for_ids(
    system: SystemData,
    ids: Sequence[str],
    similarity_threshold: float,
) -> list[InstanceCounts]:
    return [
        compute_instance_counts(system.results_by_id[instance_id], similarity_threshold)
        for instance_id in ids
    ]


def sum_counts(counts: Iterable[TupleCounts]) -> TupleCounts:
    tp = fp = fn = 0
    for item in counts:
        tp += item.tp
        fp += item.fp
        fn += item.fn
    return TupleCounts(tp=tp, fp=fp, fn=fn)


def prf(counts: TupleCounts) -> dict[str, float]:
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp > 0 else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def observed_metrics(instance_counts: Sequence[InstanceCounts]) -> dict[str, float]:
    hard = prf(sum_counts(item.hard for item in instance_counts))
    soft = prf(sum_counts(item.soft for item in instance_counts))
    target = prf(sum_counts(item.target for item in instance_counts))
    hate = prf(sum_counts(item.hate for item in instance_counts))
    return {
        "hard_precision": hard["precision"],
        "hard_recall": hard["recall"],
        "f1_hard": hard["f1"],
        "soft_precision": soft["precision"],
        "soft_recall": soft["recall"],
        "f1_soft": soft["f1"],
        "f1_avg": (hard["f1"] + soft["f1"]) / 2,
        "target_precision": target["precision"],
        "target_recall": target["recall"],
        "f1_target": target["f1"],
        "hate_precision": hate["precision"],
        "hate_recall": hate["recall"],
        "f1_hate": hate["f1"],
    }


def scores_from_selected_counts(
    instance_counts: Sequence[InstanceCounts],
    sample_indices: Sequence[int],
) -> dict[str, float]:
    hard = prf(sum_counts(instance_counts[i].hard for i in sample_indices))
    soft = prf(sum_counts(instance_counts[i].soft for i in sample_indices))
    target = prf(sum_counts(instance_counts[i].target for i in sample_indices))
    hate = prf(sum_counts(instance_counts[i].hate for i in sample_indices))
    return {
        "f1_hard": hard["f1"],
        "f1_soft": soft["f1"],
        "f1_avg": (hard["f1"] + soft["f1"]) / 2,
        "f1_target": target["f1"],
        "f1_hate": hate["f1"],
    }


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_bootstrap(scores: Sequence[float], observed: float, ci: float) -> dict[str, float]:
    sorted_scores = sorted(scores)
    alpha = (100.0 - ci) / 2.0
    return {
        "observed": observed,
        "mean": mean(scores),
        "std": pstdev(scores) if len(scores) > 1 else 0.0,
        "ci": ci,
        "ci_lower": percentile(sorted_scores, alpha),
        "ci_upper": percentile(sorted_scores, 100.0 - alpha),
    }


def bootstrap_single(
    instance_counts: Sequence[InstanceCounts],
    metrics: Sequence[str],
    n_bootstrap: int,
    rng: random.Random,
) -> dict[str, list[float]]:
    n = len(instance_counts)
    scores = {metric: [] for metric in metrics}
    for _ in range(n_bootstrap):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        sampled_scores = scores_from_selected_counts(instance_counts, sample_indices)
        for metric in metrics:
            scores[metric].append(sampled_scores[metric])
    return scores


def bootstrap_paired(
    counts_a: Sequence[InstanceCounts],
    counts_b: Sequence[InstanceCounts],
    metrics: Sequence[str],
    n_bootstrap: int,
    rng: random.Random,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    if len(counts_a) != len(counts_b):
        raise ValueError("Paired bootstrap requires equal numbers of instances.")

    n = len(counts_a)
    scores_a = {metric: [] for metric in metrics}
    scores_b = {metric: [] for metric in metrics}
    deltas = {metric: [] for metric in metrics}

    for _ in range(n_bootstrap):
        sample_indices = [rng.randrange(n) for _ in range(n)]
        sampled_scores_a = scores_from_selected_counts(counts_a, sample_indices)
        sampled_scores_b = scores_from_selected_counts(counts_b, sample_indices)
        for metric in metrics:
            score_a = sampled_scores_a[metric]
            score_b = sampled_scores_b[metric]
            scores_a[metric].append(score_a)
            scores_b[metric].append(score_b)
            deltas[metric].append(score_a - score_b)

    return scores_a, scores_b, deltas


def system_summary(
    system: SystemData,
    instance_counts: Sequence[InstanceCounts],
) -> dict:
    return {
        "path": str(system.path),
        "n_instances": system.total,
        "success": system.success,
        "success_rate": system.success / system.total if system.total else 0.0,
        "observed": observed_metrics(instance_counts),
    }


def compare_ids(system_a: SystemData, system_b: SystemData) -> list[str]:
    ids_a = set(system_a.ids)
    ids_b = set(system_b.ids)

    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)[:10]
        only_b = sorted(ids_b - ids_a)[:10]
        raise ValueError(
            "System A and B must contain the same test instance ids. "
            f"Only in A (first 10): {only_a}; only in B (first 10): {only_b}."
        )

    return list(system_a.ids)


def paired_test_summary(
    deltas: Sequence[float],
    observed_delta: float,
    ci: float,
) -> dict[str, float]:
    bootstrap_summary = summarize_bootstrap(deltas, observed_delta, ci)
    prob_delta_gt_0 = sum(delta > 0 for delta in deltas) / len(deltas)
    prob_delta_lt_0 = sum(delta < 0 for delta in deltas) / len(deltas)
    prob_delta_ge_0 = sum(delta >= 0 for delta in deltas) / len(deltas)
    prob_delta_le_0 = sum(delta <= 0 for delta in deltas) / len(deltas)
    p_value_two_sided = min(1.0, 2.0 * min(prob_delta_le_0, prob_delta_ge_0))

    return {
        "observed_delta": bootstrap_summary["observed"],
        "mean_delta": bootstrap_summary["mean"],
        "std_delta": bootstrap_summary["std"],
        "ci": bootstrap_summary["ci"],
        "ci_lower": bootstrap_summary["ci_lower"],
        "ci_upper": bootstrap_summary["ci_upper"],
        "prob_a_better": prob_delta_gt_0,
        "prob_b_better": prob_delta_lt_0,
        "p_value_two_sided": p_value_two_sided,
    }


def build_result(args: argparse.Namespace) -> dict:
    validate_args(args)
    metrics = canonical_metrics(args.metric)
    metric_selection = "all" if metrics == DEFAULT_METRICS else ",".join(metrics)
    system_a = load_system(args.system_a)
    rng = random.Random(args.seed)

    if args.system_b is None:
        ids = system_a.ids
        counts_a = compute_counts_for_ids(system_a, ids, args.similarity_threshold)
        observed_a = observed_metrics(counts_a)
        scores_a = bootstrap_single(counts_a, metrics, args.n_bootstrap, rng)

        summary_a = system_summary(system_a, counts_a)
        summary_a["bootstrap"] = {
            metric: summarize_bootstrap(scores_a[metric], observed_a[metric], args.ci)
            for metric in metrics
        }

        return {
            "mode": "single",
            "metric": metric_selection,
            "metrics": metrics,
            "n_instances": len(ids),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "similarity_threshold": args.similarity_threshold,
            "ci_method": "percentile",
            "system_a": summary_a,
        }

    system_b = load_system(args.system_b)
    ids = compare_ids(system_a, system_b)
    counts_a = compute_counts_for_ids(system_a, ids, args.similarity_threshold)
    counts_b = compute_counts_for_ids(system_b, ids, args.similarity_threshold)
    observed_a = observed_metrics(counts_a)
    observed_b = observed_metrics(counts_b)
    scores_a, scores_b, deltas = bootstrap_paired(
        counts_a,
        counts_b,
        metrics,
        args.n_bootstrap,
        rng,
    )

    summary_a = system_summary(system_a, counts_a)
    summary_b = system_summary(system_b, counts_b)
    summary_a["bootstrap"] = {
        metric: summarize_bootstrap(scores_a[metric], observed_a[metric], args.ci)
        for metric in metrics
    }
    summary_b["bootstrap"] = {
        metric: summarize_bootstrap(scores_b[metric], observed_b[metric], args.ci)
        for metric in metrics
    }

    return {
        "mode": "paired",
        "metric": metric_selection,
        "metrics": metrics,
        "n_instances": len(ids),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "similarity_threshold": args.similarity_threshold,
        "ci_method": "percentile",
        "system_a": summary_a,
        "system_b": summary_b,
        "comparison": {
            "delta_label": "system_a_minus_system_b",
            **{
                metric: paired_test_summary(
                    deltas[metric],
                    observed_a[metric] - observed_b[metric],
                    args.ci,
                )
                for metric in metrics
            },
        },
    }


def write_result(result: dict, output: Optional[Path], indent: int) -> None:
    json_indent = None if indent == 0 else indent
    text = json.dumps(result, ensure_ascii=False, indent=json_indent)
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = build_result(args)
    write_result(result, args.output, args.indent)


if __name__ == "__main__":
    main()
