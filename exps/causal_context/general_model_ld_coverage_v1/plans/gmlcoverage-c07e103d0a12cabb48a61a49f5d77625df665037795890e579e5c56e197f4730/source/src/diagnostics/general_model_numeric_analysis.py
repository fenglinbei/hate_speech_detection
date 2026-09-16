"""Gold-independent candidate math and descriptive L/D numerical analysis.

All probabilities are restricted to the frozen canonical candidate space.
Gold is accepted only by the explicit, post-scoring analysis entry points.
"""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ANALYSIS_VERSION = "general-model-numeric-analysis/v1"
GROUP_LABELS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
TASKS = ("hate", "group")
CONDITIONS = ("C0", "CL", "CD", "CLD", "PL", "PD")
SCORE_MODES = ("answer_sum", "answer_mean", "total_with_eos", "mean_with_eos")
SCORE_FIELDS = ("answer_sum", "answer_mean", "eos_logprob", "total_with_eos", "mean_with_eos")
CONTRASTS = {
    "L": {"CL": 1, "C0": -1},
    "D": {"CD": 1, "C0": -1},
    "LD": {"CLD": 1, "C0": -1},
    "LxD": {"CLD": 1, "CL": -1, "CD": -1, "C0": 1},
    "L_given_D": {"CLD": 1, "CD": -1},
    "D_given_L": {"CLD": 1, "CL": -1},
    "CL_minus_PL": {"CL": 1, "PL": -1},
    "CD_minus_PD": {"CD": 1, "PD": -1},
}


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def candidate_catalog() -> dict[str, list[dict[str, Any]]]:
    """Return fresh canonical candidates; group ordinal is its inclusion mask."""
    hate = [
        {"candidate_id": label, "ordinal": ordinal, "labels": [label],
         "cardinality": None, "canonical_answer": json.dumps(label), "label_bitmap": None}
        for ordinal, label in enumerate(("hate", "non-hate"))
    ]
    group = []
    for mask in range(1 << len(GROUP_LABELS)):
        labels = [label for bit, label in enumerate(GROUP_LABELS) if mask & (1 << bit)]
        group.append({
            "candidate_id": f"group-{mask:02d}", "ordinal": mask, "labels": labels,
            "cardinality": len(labels), "label_bitmap": mask,
            "canonical_answer": json.dumps(labels, separators=(",", ":")),
        })
    return {"hate": hate, "group": group}


def candidate_scores(token_logprobs: Sequence[float], eos_logprob: float) -> dict[str, float]:
    if not token_logprobs:
        raise ValueError("candidate answer must contain at least one token")
    values = [_finite(value, "token logprob") for value in token_logprobs]
    eos = _finite(eos_logprob, "EOS logprob")
    answer_sum = math.fsum(values)
    total = math.fsum([*values, eos])
    return {
        "answer_sum": answer_sum, "answer_mean": answer_sum / len(values),
        "eos_logprob": eos, "total_with_eos": total,
        "mean_with_eos": total / (len(values) + 1),
    }


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires a nonempty collection")
    finite = [_finite(value, "logsumexp input") for value in values]
    maximum = max(finite)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in finite))


def _validated_candidates(task: str, candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"unsupported numeric task: {task}")
    expected = candidate_catalog()[task]
    if len(candidates) != len(expected):
        raise ValueError(f"{task} requires exactly {len(expected)} candidates")
    for candidate, canonical in zip(candidates, expected):
        for field, value in canonical.items():
            if candidate.get(field) != value:
                raise ValueError(f"candidate identity/order mismatch: {field}")
        token_ids = candidate.get("answer_token_ids")
        logprobs = candidate.get("token_logprobs")
        if (not isinstance(token_ids, list) or not token_ids
                or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
                or not isinstance(logprobs, list) or len(logprobs) != len(token_ids)):
            raise ValueError("candidate answer token/logprob boundary mismatch")
        scores = candidate.get("scores", {})
        if set(scores) != set(SCORE_FIELDS):
            raise ValueError("candidate score fields are incomplete or unexpected")
        recomputed = candidate_scores(logprobs, scores["eos_logprob"])
        for field, value in recomputed.items():
            if not math.isclose(_finite(scores[field], field), value, abs_tol=1e-10, rel_tol=1e-12):
                raise ValueError(f"stored candidate score disagrees with tokens: {field}")
    return list(candidates)


def _margins(task: str, candidates: Sequence[Mapping[str, Any]], mode: str) -> dict[str, float]:
    scores = [float(candidate["scores"][mode]) for candidate in candidates]
    if task == "hate":
        return {"hate": scores[0] - scores[1]}
    return {
        label: logsumexp([score for mask, score in enumerate(scores) if mask & (1 << bit)])
        - logsumexp([score for mask, score in enumerate(scores) if not mask & (1 << bit)])
        for bit, label in enumerate(GROUP_LABELS)
    }


def block_readouts(task: str, candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """All finite registered numerical gate values, with no gold input."""
    candidates = _validated_candidates(task, candidates)
    result = {}
    for candidate in candidates:
        prefix = f"candidate/{candidate['candidate_id']}"
        for index, value in enumerate(candidate["token_logprobs"]):
            result[f"{prefix}/token/{index}"] = float(value)
        for field in SCORE_FIELDS:
            result[f"{prefix}/{field}"] = float(candidate["scores"][field])
    for mode in SCORE_MODES:
        for label, margin in _margins(task, candidates, mode).items():
            result[f"margin/{mode}/{label}"] = margin
    return result


def paired_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    if set(values) != set(CONDITIONS):
        raise ValueError("paired contrasts require exactly six core conditions")
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


def _probabilities(scores: Sequence[float]) -> list[float]:
    normalizer = logsumexp(scores)
    return [math.exp(score - normalizer) for score in scores]


def _entropy(probabilities: Sequence[float]) -> float:
    return -math.fsum(probability * math.log(probability)
                      for probability in probabilities if probability > 0)


def _gold_ordinal(task: str, gold: Any) -> int:
    if task == "hate":
        if gold not in ("hate", "non-hate"):
            raise ValueError("gold hate must be an independent binary label")
        return 0 if gold == "hate" else 1
    if (not isinstance(gold, list) or any(label not in GROUP_LABELS for label in gold)
            or len(set(gold)) != len(gold)):
        raise ValueError("gold group must be a unique canonical label list")
    return sum(1 << GROUP_LABELS.index(label) for label in gold)


def block_analysis(
    task: str, candidates: Sequence[Mapping[str, Any]], *, gold: Any = None,
) -> dict[str, float | None]:
    """Derive distribution, cardinality, and optional sealed-raw gold diagnostics."""
    candidates = _validated_candidates(task, candidates)
    gold_index = None if gold is None else _gold_ordinal(task, gold)
    result: dict[str, float | None] = {}
    for mode in SCORE_MODES:
        scores = [float(candidate["scores"][mode]) for candidate in candidates]
        probabilities = _probabilities(scores)
        prefix = f"{mode}/"
        result[prefix + "candidate_entropy"] = _entropy(probabilities)
        for label, value in _margins(task, candidates, mode).items():
            result[prefix + "margin/" + label] = value
            if task == "hate":
                inclusion = probabilities[0]
            else:
                bit = GROUP_LABELS.index(label)
                inclusion = math.fsum(probability for mask, probability in enumerate(probabilities)
                                      if mask & (1 << bit))
            result[prefix + "inclusion_mass/" + label] = inclusion
        if task == "group":
            cardinality_scores = [
                [scores[mask] for mask in range(32) if mask.bit_count() == size]
                for size in range(6)
            ]
            cardinality_totals = [logsumexp(values) for values in cardinality_scores]
            cardinality_means = [value - math.log(len(values))
                                 for value, values in zip(cardinality_totals, cardinality_scores)]
            for weighting, values in (("natural", cardinality_totals), ("equal_k", cardinality_means)):
                size_probabilities = _probabilities(values)
                size_prefix = prefix + "cardinality/" + weighting + "/"
                for size, probability in enumerate(size_probabilities):
                    result[size_prefix + f"p{size}"] = probability
                result[size_prefix + "expected_size"] = math.fsum(
                    size * probability for size, probability in enumerate(size_probabilities))
                result[size_prefix + "entropy"] = _entropy(size_probabilities)
            for bit, label in enumerate(GROUP_LABELS):
                for size in range(6):
                    inside = [score for mask, score in enumerate(scores)
                              if mask.bit_count() == size and mask & (1 << bit)]
                    outside = [score for mask, score in enumerate(scores)
                               if mask.bit_count() == size and not mask & (1 << bit)]
                    natural = logsumexp(inside) - logsumexp(outside) if inside and outside else None
                    corrected = (natural - math.log(len(inside)) + math.log(len(outside))
                                 if natural is not None else None)
                    result[prefix + f"within_k/natural/{label}/{size}"] = natural
                    result[prefix + f"within_k/count_corrected/{label}/{size}"] = corrected
        if gold_index is not None:
            gold_score = scores[gold_index]
            gold_prefix = prefix + "gold/"
            result[gold_prefix + "rank"] = float(1 + sum(score > gold_score for score in scores))
            result[gold_prefix + "tie_count"] = float(sum(score == gold_score for score in scores))
            result[gold_prefix + "mass"] = probabilities[gold_index]
            result[gold_prefix + "nll"] = logsumexp(scores) - gold_score
            result[gold_prefix + "best_nongold_margin"] = gold_score - max(
                score for index, score in enumerate(scores) if index != gold_index)
            if task == "group":
                for bit, label in enumerate(GROUP_LABELS):
                    result[gold_prefix + "toggle_margin/" + label] = gold_score - scores[gold_index ^ (1 << bit)]
    return result


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    import numpy as np

    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(finite):
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None,
                "q25": None, "q75": None, "iqr": None,
                "min": None, "max": None, "status": "undefined"}
    if not np.isfinite(finite).all():
        raise ValueError("analysis summary contains a nonfinite value")
    q25, q75 = np.quantile(finite, [0.25, 0.75], method="linear")
    return {
        "n": int(len(finite)), "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "p05": float(np.quantile(finite, 0.05)), "p95": float(np.quantile(finite, 0.95)),
        "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25),
        "min": float(np.min(finite)), "max": float(np.max(finite)), "status": "defined",
    }


def joint_bootstrap_mean_ci(
    values: Any, *, strata: Mapping[str, Sequence[bool]], replicates: int = 10_000,
    seed: int = 42, batch_size: int = 256,
) -> dict[str, list[list[float] | None]]:
    """Within each fixed stratum, share each query draw across all endpoints."""
    import numpy as np

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("bootstrap values must be a nonempty finite query-by-endpoint matrix")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("bootstrap replicates and batch size must be positive")
    n, endpoints = matrix.shape
    masks = {}
    for name, raw_mask in strata.items():
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError("bootstrap stratum does not match query frame")
        masks[name] = mask
    result = {}
    for name, mask in masks.items():
        selected = matrix[mask]
        size = len(selected)
        if not size:
            result[name] = [None] * endpoints
            continue
        stream = int.from_bytes(hashlib.sha256(f"{seed}\0{name}".encode()).digest(), "big")
        rng = np.random.default_rng(stream)
        distribution = np.empty((replicates, endpoints), dtype=np.float64)
        for start in range(0, replicates, batch_size):
            count = min(batch_size, replicates - start)
            indices = rng.integers(0, size, size=(count, size))
            weights = np.zeros((count, size), dtype=np.float64)
            np.add.at(weights, (np.arange(count)[:, None], indices), 1)
            distribution[start:start + count] = (weights @ selected) / size
        result[name] = np.quantile(distribution, [0.025, 0.975], axis=0, method="linear").T.tolist()
    return result


def analyze_blocks(
    blocks: Iterable[Mapping[str, Any]], *, frame: Sequence[Mapping[str, Any]],
    gold_by_query: Mapping[str, Mapping[str, Any]], epsilon: float,
    expected_query_count: int = 643, bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Analyze complete six-condition paired raw blocks after raw sealing.

    CI endpoints are the four score modes' hate/group margins in the eight
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
        raise ValueError("query frame requires a Boolean lex_hit for every query")
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
        "lex_hit": [row["lex_hit"] for row in frame],
        "lex_no_hit": [not row["lex_hit"] for row in frame],
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
        "epsilon": epsilon, "conditions": list(CONDITIONS), "contrasts": CONTRASTS,
        "score_modes": list(SCORE_MODES), "group_order": list(GROUP_LABELS),
        "bootstrap": {"unit": "query", "seed": bootstrap_seed, "replicates": bootstrap_replicates,
                      "interval": "percentile_95", "scope": "descriptive",
                      "shared_draw_across_tasks_conditions_endpoints_within_stratum": True,
                      "resampling_population": "fixed stratum queries, stratum-sized draws",
                      "rng": "NumPy PCG64, SHA256(seed + NUL + stratum) integer stream seed",
                      "endpoint_count": len(endpoint_names), "resampled_model_seeds": False,
                      "empty_stratum_policy": "undefined interval"},
        "interpretation": {
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
            "gold_size_confounding": "descriptive only; gold size covaries with hate label and lexicon hit",
            "auxiliary_summary": "all auxiliary metrics retain paired values and descriptive distributions",
        },
        "stratum_counts": {name: sum(mask) for name, mask in descriptive_strata.items()},
        "condition_summaries": condition_summaries, "contrast_summaries": contrast_summaries,
        "per_query": per_query,
    }
