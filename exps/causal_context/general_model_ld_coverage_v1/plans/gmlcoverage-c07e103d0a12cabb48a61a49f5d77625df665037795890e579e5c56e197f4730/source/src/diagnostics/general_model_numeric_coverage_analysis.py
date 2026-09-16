"""Eight-condition coverage-extension analysis over unchanged frozen math.

Candidate scores, all block readouts, summaries and joint bootstrap are imported
from the immutable original module. Only condition/contrast orchestration and
resource interpretation differ. Query gold remains an explicit analysis input.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from diagnostics import general_model_numeric_analysis as _frozen


ANALYSIS_VERSION = "general-model-numeric-coverage-analysis/v1"
GROUP_LABELS = _frozen.GROUP_LABELS
TASKS = _frozen.TASKS
SCORE_MODES = _frozen.SCORE_MODES
SCORE_FIELDS = _frozen.SCORE_FIELDS
CONDITIONS = ("C0", "CLnew", "CD", "CLDnew", "PLnew", "PD", "CLq", "CLqD")
PRIMARY_CONTRASTS = ("L", "D", "LD", "LxD")
REFERENCE_CONTRASTS = ("CLnew_minus_CLq", "CLDnew_minus_CLqD")
CONTRASTS = {
    "L": {"CLnew": 1, "C0": -1},
    "D": {"CD": 1, "C0": -1},
    "LD": {"CLDnew": 1, "C0": -1},
    "LxD": {"CLDnew": 1, "CLnew": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLDnew": 1, "CD": -1},
    "D_given_L": {"CLDnew": 1, "CLnew": -1},
    "CL_minus_PL": {"CLnew": 1, "PLnew": -1},
    "CD_minus_PD": {"CD": 1, "PD": -1},
    "CLnew_minus_CLq": {"CLnew": 1, "CLq": -1},
    "CLDnew_minus_CLqD": {"CLDnew": 1, "CLqD": -1},
}

candidate_catalog = _frozen.candidate_catalog
candidate_scores = _frozen.candidate_scores
block_readouts = _frozen.block_readouts
block_analysis = _frozen.block_analysis
logsumexp = _frozen.logsumexp
joint_bootstrap_mean_ci = _frozen.joint_bootstrap_mean_ci
_finite = _frozen._finite
_gold_ordinal = _frozen._gold_ordinal
_summary = _frozen._summary


def paired_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(CONDITIONS):
        raise ValueError("paired contrasts require exactly eight registered conditions")
    finite = {condition: _finite(value, condition) for condition, value in values.items()}
    return {name: math.fsum(weight * finite[condition] for condition, weight in weights.items())
            for name, weights in CONTRASTS.items()}


def numerical_direction(value: float, *, epsilon: float, contrast: str) -> dict[str, Any]:
    value = _finite(value, "effect")
    epsilon = _finite(epsilon, "epsilon")
    if epsilon < 0 or contrast not in CONTRASTS:
        raise ValueError("invalid epsilon or contrast")
    bound = epsilon * sum(abs(weight) for weight in CONTRASTS[contrast].values())
    raw_sign = 1 if value > 0 else -1 if value < 0 else 0
    direction = "numerically_unresolved" if abs(value) <= bound else "positive" if value > 0 else "negative"
    return {"raw_sign": raw_sign, "direction": direction, "error_bound": bound}


def analyze_blocks(
    blocks: Iterable[Mapping[str, Any]], *, frame: Sequence[Mapping[str, Any]],
    gold_by_query: Mapping[str, Mapping[str, Any]], epsilon: float,
    expected_query_count: int = 643, bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Analyze complete eight-condition paired raw blocks after raw sealing.

    CI endpoints are the four score modes' hate/group margins in the ten
    registered contrasts. Other diagnostics retain paired query values and
    descriptive summaries, without tests or confirmatory p-values.
    """
    import numpy as np

    epsilon = _finite(epsilon, "epsilon")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if len(frame) != expected_query_count or not frame:
        raise ValueError("analysis requires the complete registered query frame")
    query_ids = [row["query_id"] for row in frame]
    if len(set(query_ids)) != len(query_ids) or any(not isinstance(qid, str) for qid in query_ids):
        raise ValueError("query frame requires unique string query IDs")
    if any(type(row.get("lex_hit")) is not bool for row in frame):
        raise ValueError("query frame requires a Boolean lex_hit (query-only Lq) for every query")
    if set(gold_by_query) != set(query_ids):
        raise ValueError("analysis gold frame must exactly match registered query frame")
    gold_sizes = []
    for query_id in query_ids:
        gold = gold_by_query[query_id]
        for task in TASKS:
            _gold_ordinal(task, gold[task])
        gold_sizes.append(len(gold["group"]))
    query_set = set(query_ids)
    indexed = {}
    for block in blocks:
        key = (block["query_id"], block["task"], block["condition"])
        if key[0] not in query_set or key[1] not in TASKS or key[2] not in CONDITIONS:
            raise ValueError(f"unexpected analysis block: {key}")
        if key in indexed:
            raise ValueError(f"duplicate analysis block: {key}")
        indexed[key] = block_analysis(key[1], block["candidates"], gold=gold_by_query[key[0]][key[1]])
    expected_blocks = len(frame) * len(TASKS) * len(CONDITIONS)
    if len(indexed) != expected_blocks:
        raise ValueError(f"incomplete paired analysis: expected {expected_blocks} blocks, got {len(indexed)}")
    strata = {
        "all": [True] * len(frame),
        "Lq_hit": [row["lex_hit"] for row in frame],
        "Lq_no_hit": [not row["lex_hit"] for row in frame],
        "gold_size_0": [size == 0 for size in gold_sizes],
        "gold_size_1": [size == 1 for size in gold_sizes],
        "gold_size_2plus": [size >= 2 for size in gold_sizes],
    }
    descriptive_strata = {
        **strata,
        "gold_size_2_appendix": [size == 2 for size in gold_sizes],
        "gold_size_3_appendix": [size == 3 for size in gold_sizes],
    }
    endpoint_names = []
    for task in TASKS:
        labels = ("hate",) if task == "hate" else GROUP_LABELS
        for mode in SCORE_MODES:
            for label in labels:
                for contrast in CONTRASTS:
                    endpoint_names.append((task, f"{mode}/margin/{label}", contrast))
    endpoint_positions = {key: index for index, key in enumerate(endpoint_names)}
    bootstrap_matrix = np.empty((len(frame), len(endpoint_names)), dtype=np.float64)
    condition_values: dict[tuple[str, str, str], list[float | None]] = {}
    contrast_values: dict[tuple[str, str, str], list[float | None]] = {}
    per_query = []
    for index, query_id in enumerate(query_ids):
        query_result = {"query_id": query_id, "lex_hit": frame[index]["lex_hit"],
                        "gold_group_cardinality": gold_sizes[index], "tasks": {}}
        for task in TASKS:
            cells = {condition: indexed[(query_id, task, condition)] for condition in CONDITIONS}
            keys = tuple(cells["C0"])
            if any(tuple(values) != keys for values in cells.values()):
                raise ValueError("paired condition readout schemas differ")
            effects = {contrast: {} for contrast in CONTRASTS}
            for metric in keys:
                values = {condition: cells[condition][metric] for condition in CONDITIONS}
                for condition, value in values.items():
                    condition_values.setdefault((task, metric, condition), []).append(value)
                if all(value is None for value in values.values()):
                    contrasts = {name: None for name in CONTRASTS}
                elif any(value is None for value in values.values()):
                    raise ValueError("paired numeric metric is undefined in only some conditions")
                else:
                    contrasts = paired_contrasts(values)
                for contrast, value in contrasts.items():
                    effects[contrast][metric] = value
                    key = (task, metric, contrast)
                    contrast_values.setdefault(key, []).append(value)
                    if key in endpoint_positions:
                        bootstrap_matrix[index, endpoint_positions[key]] = value
            query_result["tasks"][task] = {"conditions": cells, "contrasts": effects}
        per_query.append(query_result)
    intervals = joint_bootstrap_mean_ci(bootstrap_matrix, strata=strata,
                                        replicates=bootstrap_replicates, seed=bootstrap_seed)
    condition_summaries = []
    contrast_summaries = []
    for family, source, target in (("condition", condition_values, condition_summaries),
                                   ("contrast", contrast_values, contrast_summaries)):
        for (task, metric, cell), values in source.items():
            for stratum, mask in descriptive_strata.items():
                selected = [value for value, include in zip(values, mask) if include]
                summary = {"task": task, "metric": metric, family: cell, "stratum": stratum,
                           **_summary(selected)}
                key = (task, metric, cell)
                if family == "contrast" and key in endpoint_positions:
                    if stratum in intervals:
                        summary["descriptive_ci95"] = intervals[stratum][endpoint_positions[key]]
                    if summary["mean"] is not None:
                        summary.update(numerical_direction(summary["mean"], epsilon=epsilon, contrast=cell))
                        bound = summary["error_bound"]
                        summary["positive_query_count"] = sum(value > bound for value in selected)
                        summary["negative_query_count"] = sum(value < -bound for value in selected)
                        summary["unresolved_query_count"] = sum(abs(value) <= bound for value in selected)
                        for direction in ("positive", "negative", "unresolved"):
                            summary[f"{direction}_query_fraction"] = summary[f"{direction}_query_count"] / len(selected)
                target.append(summary)
    return {
        "schema_version": ANALYSIS_VERSION, "query_count": len(frame), "block_count": len(indexed),
        "epsilon": epsilon, "conditions": list(CONDITIONS),
        "contrasts": {name: dict(weights) for name, weights in CONTRASTS.items()},
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "reference_contrasts": list(REFERENCE_CONTRASTS),
        "score_modes": list(SCORE_MODES), "group_order": list(GROUP_LABELS),
        "bootstrap": {"unit": "query", "seed": bootstrap_seed, "replicates": bootstrap_replicates,
                      "interval": "percentile_95", "scope": "descriptive",
                      "shared_draw_across_tasks_conditions_endpoints_within_stratum": True,
                      "resampling_population": "fixed stratum queries, stratum-sized draws",
                      "rng": "NumPy PCG64, SHA256(seed + NUL + stratum) integer stream seed",
                      "endpoint_count": len(endpoint_names), "resampled_model_seeds": False,
                      "empty_stratum_policy": "undefined interval"},
        "interpretation": {
            "lexicon_resource": "Lnew is the source-independent globally deduplicated Lq union Ld",
            "lexicon_hit_strata": "frame/per_query lex_hit and Lq_hit/Lq_no_hit refer only to query Lq",
            "reference_differences": "coverage-expansion input differences, not isolated semantic or accuracy gains",
            "demo_ablation": "D absent means demonstrations are hidden; Lnew retains demo-derived lexical coverage",
            "probability": "total-logprob scores yield probability restricted to the fixed canonical candidate space",
            "score_normalization": {
                "answer_sum": "restricted answer-sequence probability mass",
                "total_with_eos": "restricted answer-plus-EOS sequence probability mass",
                "answer_mean": "normalized mean-score preference weight, not sequence probability",
                "mean_with_eos": "normalized mean-score preference weight, not sequence probability",
            },
            "primary_score": "answer_sum", "primary_endpoint": "paired marginal log-odds change",
            "interaction": "four-cell nonadditivity, not an independent causal mechanism claim",
            "canonical_order_dependence": True,
            "gold_rank": "one plus the number of strictly better candidates; ties also counted",
            "equal_k": "logmeanexp within each cardinality before normalization over cardinalities",
            "within_k_endpoints": "k=0 and k=5 undefined because one label side is empty",
            "gold_size_confounding": "descriptive only; gold size covaries with hate label and query-only Lq hit",
            "auxiliary_summary": "all auxiliary metrics retain paired values and descriptive distributions",
        },
        "stratum_counts": {name: sum(mask) for name, mask in descriptive_strata.items()},
        "condition_summaries": condition_summaries, "contrast_summaries": contrast_summaries,
        "per_query": per_query,
    }

