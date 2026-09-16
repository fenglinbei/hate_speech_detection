"""Strict, deterministic Stage-1 free-generation metrics.

The implementation in this module is intentionally independent from the legacy
``metrics.core`` helpers.  Stage 1 has a frozen JSON wire contract and three
different evaluation namespaces (unbound fields, span-bound labels, and whole
tuples); collapsing those namespaces would change the estimand.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

from utils.quadruple import (
    HATEFUL_LABELS,
    Quadruple,
    canonicalize_quadruples,
    parse_quadruples,
    serialize_quadruples,
)


METRIC_SCHEMA_VERSION = "stage1-query-metrics/v1"
ASSIGNMENT_VERSION = "stage1-optimal-tuple-assignment/v1"
SIMILARITY_VERSION = "similarity-v1"
DEFAULT_SOFT_THRESHOLD = 0.5
DEFAULT_MAX_TUPLES = 6


@dataclass(frozen=True)
class Assignment:
    """A deterministic bipartite assignment."""

    pairs: tuple[tuple[int, int], ...]
    weights: tuple[float, ...]
    total_weight: float

    @property
    def count(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": ASSIGNMENT_VERSION,
            "pairs": [list(pair) for pair in self.pairs],
            "weights": list(self.weights),
            "total_weight": self.total_weight,
        }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def similarity_v1(pred: str | None, gold: str | None) -> float:
    """Frozen similarity over canonical Unicode code-point sequences."""

    if pred is None and gold is None:
        return 1.0
    if pred is None or gold is None:
        return 0.0
    return SequenceMatcher(None, pred, gold, autojunk=False).ratio()


def _choose_better(
    left: tuple[int, float, tuple[tuple[int, int], ...], tuple[float, ...]],
    right: tuple[int, float, tuple[tuple[int, int], ...], tuple[float, ...]],
) -> tuple[int, float, tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Maximise cardinality, then rounded weight, then minimise pair list."""

    if left[0] != right[0]:
        return left if left[0] > right[0] else right
    if left[1] != right[1]:
        return left if left[1] > right[1] else right
    return left if left[2] <= right[2] else right


def optimal_assignment(
    pred_count: int,
    gold_count: int,
    edge_weight: Callable[[int, int], float | None],
) -> Assignment:
    """Return the frozen lexicographically-tied maximum assignment.

    ``None`` means an edge is forbidden.  Every admitted edge weight is rounded
    to 12 decimal places before optimisation, as required by Stage 1.
    """

    if pred_count < 0 or gold_count < 0:
        raise ValueError("assignment sizes must be non-negative")

    weights: dict[tuple[int, int], float] = {}
    for pred_index in range(pred_count):
        for gold_index in range(gold_count):
            value = edge_weight(pred_index, gold_index)
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("assignment weights must be finite")
            weights[(pred_index, gold_index)] = round(number, 12)

    @lru_cache(maxsize=None)
    def solve(
        pred_index: int,
        used_gold_mask: int,
    ) -> tuple[int, float, tuple[tuple[int, int], ...], tuple[float, ...]]:
        if pred_index == pred_count:
            return (0, 0.0, (), ())

        best = solve(pred_index + 1, used_gold_mask)
        for gold_index in range(gold_count):
            bit = 1 << gold_index
            weight = weights.get((pred_index, gold_index))
            if weight is None or used_gold_mask & bit:
                continue
            child = solve(pred_index + 1, used_gold_mask | bit)
            candidate = (
                child[0] + 1,
                round(child[1] + weight, 12),
                ((pred_index, gold_index),) + child[2],
                (weight,) + child[3],
            )
            best = _choose_better(best, candidate)
        return best

    count, total, pairs, pair_weights = solve(0, 0)
    if count != len(pairs):  # defensive invariant
        raise AssertionError("assignment cardinality invariant failed")
    return Assignment(pairs=pairs, weights=pair_weights, total_weight=total)


def _quad_key(quad: Quadruple) -> tuple[Any, ...]:
    return (quad.target, quad.argument, quad.targeted_group, quad.hateful)


def _span_weight(pred: Quadruple, gold: Quadruple) -> tuple[float, float, float]:
    target = similarity_v1(pred.target, gold.target)
    argument = similarity_v1(pred.argument, gold.argument)
    return target, argument, round((target + argument) / 2.0, 12)


def _assignment_counts(
    assignment: Assignment,
    pred_count: int,
    gold_count: int,
) -> dict[str, int]:
    return {
        "tp": assignment.count,
        "fp": pred_count - assignment.count,
        "fn": gold_count - assignment.count,
    }


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _field_value(quad: Quadruple, field: str) -> Any:
    return getattr(quad, field)


def _unbound_field(
    pred: Sequence[Quadruple],
    gold: Sequence[Quadruple],
    field: str,
    *,
    scoring_valid: bool,
) -> dict[str, Any]:
    exact = optimal_assignment(
        len(pred),
        len(gold),
        lambda i, j: 1.0 if _field_value(pred[i], field) == _field_value(gold[j], field) else None,
    )
    counts = _assignment_counts(exact, len(pred), len(gold))
    result: dict[str, Any] = {
        "exact_tp": counts["tp"],
        "exact_fp": counts["fp"],
        "exact_fn": counts["fn"],
        "exact_correct": bool(
            scoring_valid
            and counts["fp"] == 0
            and counts["fn"] == 0
        ),
        "exact_assignment": exact.as_dict(),
    }
    if field in {"target", "argument"}:
        similarity = optimal_assignment(
            len(pred),
            len(gold),
            lambda i, j: similarity_v1(
                _field_value(pred[i], field),
                _field_value(gold[j], field),
            ),
        )
        denominator = max(len(pred), len(gold), 1)
        result.update(
            {
                "similarity": similarity.total_weight / denominator,
                "similarity_denominator": denominator,
                "similarity_assignment": similarity.as_dict(),
            }
        )
    return result


def _label_counts_for_bound_field(
    pred: Sequence[Quadruple],
    gold: Sequence[Quadruple],
    assignment: Assignment,
    field: str,
    *,
    scoring_valid: bool,
) -> dict[str, Any]:
    matched_pred = {pair[0] for pair in assignment.pairs}
    matched_gold = {pair[1] for pair in assignment.pairs}
    tp = fp = fn = 0
    all_exact = True
    for pred_index, gold_index in assignment.pairs:
        if _field_value(pred[pred_index], field) == _field_value(gold[gold_index], field):
            tp += 1
        else:
            fp += 1
            fn += 1
            all_exact = False
    fp += len(pred) - len(matched_pred)
    fn += len(gold) - len(matched_gold)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "correct": bool(
            scoring_valid
            and all_exact
            and len(matched_pred) == len(pred)
            and len(matched_gold) == len(gold)
        ),
    }


def _group_atom_counts(
    pred: Sequence[Quadruple],
    gold: Sequence[Quadruple],
    assignment: Assignment,
) -> dict[str, int]:
    matched_pred = {pair[0] for pair in assignment.pairs}
    matched_gold = {pair[1] for pair in assignment.pairs}
    tp = fp = fn = 0
    for pred_index, gold_index in assignment.pairs:
        pred_atoms = set(pred[pred_index].targeted_group)
        gold_atoms = set(gold[gold_index].targeted_group)
        tp += len(pred_atoms & gold_atoms)
        fp += len(pred_atoms - gold_atoms)
        fn += len(gold_atoms - pred_atoms)
    for index, quad in enumerate(pred):
        if index not in matched_pred:
            fp += len(quad.targeted_group)
    for index, quad in enumerate(gold):
        if index not in matched_gold:
            fn += len(quad.targeted_group)
    return {"tp": tp, "fp": fp, "fn": fn}


def _hateful_class_counts(
    pred: Sequence[Quadruple],
    gold: Sequence[Quadruple],
    assignment: Assignment,
) -> dict[str, dict[str, int]]:
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in HATEFUL_LABELS}
    matched_pred = {pair[0] for pair in assignment.pairs}
    matched_gold = {pair[1] for pair in assignment.pairs}
    for pred_index, gold_index in assignment.pairs:
        pred_label = pred[pred_index].hateful
        gold_label = gold[gold_index].hateful
        if pred_label == gold_label:
            counts[pred_label]["tp"] += 1
        else:
            counts[pred_label]["fp"] += 1
            counts[gold_label]["fn"] += 1
    for index, quad in enumerate(pred):
        if index not in matched_pred:
            counts[quad.hateful]["fp"] += 1
    for index, quad in enumerate(gold):
        if index not in matched_gold:
            counts[quad.hateful]["fn"] += 1
    return counts


def _issue_dict(issue: Any, *, phase: str) -> dict[str, str]:
    return {
        "phase": phase,
        "code": str(issue.code),
        "message": str(issue.message),
        "path": str(issue.path),
    }


def evaluate_query(
    *,
    query_id: str,
    condition: str,
    raw_output: str,
    gold: Sequence[Quadruple | Mapping[str, Any]],
    runner_status: str = "ok",
    content_sha256: str | None = None,
    gold_sha256: str | None = None,
    prompt_sha256: str | None = None,
    context_record_sha256: str | None = None,
    soft_threshold: float = DEFAULT_SOFT_THRESHOLD,
    max_tuples: int = DEFAULT_MAX_TUPLES,
) -> dict[str, Any]:
    """Evaluate one output while retaining every failure in the denominator."""

    if not (0.0 <= soft_threshold < 1.0):
        raise ValueError("soft_threshold must be in [0, 1)")
    gold_quads = canonicalize_quadruples(gold)
    if len(gold_quads) > max_tuples:
        raise ValueError(f"gold tuple count exceeds frozen maximum {max_tuples}")

    strict_result = parse_quadruples(raw_output, mode="strict")
    recovery_result = parse_quadruples(raw_output, mode="recover")
    runner_ok = runner_status == "ok"
    tuple_limit_ok = len(strict_result.quadruples) <= max_tuples
    scoring_valid = runner_ok and strict_result.strict_format_valid and tuple_limit_ok
    pred_quads = strict_result.quadruples if scoring_valid else []

    hard_assignment = optimal_assignment(
        len(pred_quads),
        len(gold_quads),
        lambda i, j: 1.0 if _quad_key(pred_quads[i]) == _quad_key(gold_quads[j]) else None,
    )
    soft_assignment = optimal_assignment(
        len(pred_quads),
        len(gold_quads),
        lambda i, j: (
            _span_weight(pred_quads[i], gold_quads[j])[2]
            if pred_quads[i].targeted_group == gold_quads[j].targeted_group
            and pred_quads[i].hateful == gold_quads[j].hateful
            and _span_weight(pred_quads[i], gold_quads[j])[0] > soft_threshold
            and _span_weight(pred_quads[i], gold_quads[j])[1] > soft_threshold
            else None
        ),
    )
    bound_assignment = optimal_assignment(
        len(pred_quads),
        len(gold_quads),
        lambda i, j: (
            _span_weight(pred_quads[i], gold_quads[j])[2]
            if _span_weight(pred_quads[i], gold_quads[j])[0] > soft_threshold
            and _span_weight(pred_quads[i], gold_quads[j])[1] > soft_threshold
            else None
        ),
    )

    hard_counts = _assignment_counts(hard_assignment, len(pred_quads), len(gold_quads))
    soft_counts = _assignment_counts(soft_assignment, len(pred_quads), len(gold_quads))
    hard_correct = bool(scoring_valid and hard_counts["fp"] == 0 and hard_counts["fn"] == 0)
    soft_correct = bool(scoring_valid and soft_counts["fp"] == 0 and soft_counts["fn"] == 0)

    field_unbound = {
        field: _unbound_field(pred_quads, gold_quads, field, scoring_valid=scoring_valid)
        for field in ("target", "argument", "targeted_group", "hateful")
    }
    targeted_group = _label_counts_for_bound_field(
        pred_quads,
        gold_quads,
        bound_assignment,
        "targeted_group",
        scoring_valid=scoring_valid,
    )
    targeted_group["atom"] = _group_atom_counts(pred_quads, gold_quads, bound_assignment)
    hateful = _label_counts_for_bound_field(
        pred_quads,
        gold_quads,
        bound_assignment,
        "hateful",
        scoring_valid=scoring_valid,
    )
    hateful["class_counts"] = _hateful_class_counts(pred_quads, gold_quads, bound_assignment)
    joint = _label_counts_for_bound_field(
        pred_quads,
        gold_quads,
        bound_assignment,
        "targeted_group",
        scoring_valid=scoring_valid,
    )
    # Replace the group-only result with the joint equality result.
    joint_tp = joint_fp = joint_fn = 0
    joint_exact = True
    matched_pred = {pair[0] for pair in bound_assignment.pairs}
    matched_gold = {pair[1] for pair in bound_assignment.pairs}
    for pred_index, gold_index in bound_assignment.pairs:
        pred_key = (pred_quads[pred_index].targeted_group, pred_quads[pred_index].hateful)
        gold_key = (gold_quads[gold_index].targeted_group, gold_quads[gold_index].hateful)
        if pred_key == gold_key:
            joint_tp += 1
        else:
            joint_fp += 1
            joint_fn += 1
            joint_exact = False
    joint_fp += len(pred_quads) - len(matched_pred)
    joint_fn += len(gold_quads) - len(matched_gold)
    joint = {
        "tp": joint_tp,
        "fp": joint_fp,
        "fn": joint_fn,
        "correct": bool(
            scoring_valid
            and joint_exact
            and len(matched_pred) == len(pred_quads)
            and len(matched_gold) == len(gold_quads)
        ),
    }

    canonical_gold = serialize_quadruples(gold_quads)
    errors = [_issue_dict(issue, phase="strict") for issue in strict_result.errors]
    warnings = [_issue_dict(issue, phase="strict") for issue in strict_result.warnings]
    if not runner_ok:
        errors.append({"phase": "runner", "code": "runner_status_not_ok", "message": runner_status, "path": "$"})
    if not tuple_limit_ok:
        errors.append(
            {
                "phase": "evaluator",
                "code": "tuple_limit_exceeded",
                "message": f"predicted more than {max_tuples} tuples",
                "path": "$",
            }
        )

    strict_format_valid = bool(runner_ok and strict_result.strict_format_valid and tuple_limit_ok)
    pred_tuple_count_observed = len(strict_result.quadruples) if strict_result.strict_format_valid else 0
    tuple_count_correct = bool(
        strict_format_valid and pred_tuple_count_observed == len(gold_quads)
    )
    result = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "id": str(query_id),
        "condition": condition,
        "content_sha256": content_sha256,
        "gold_sha256": gold_sha256 or _sha256_text(canonical_gold),
        "prompt_sha256": prompt_sha256,
        "context_record_sha256": context_record_sha256,
        "raw_output": raw_output,
        "raw_output_sha256": _sha256_text(raw_output),
        "runner_status": runner_status,
        "syntax_valid": bool(runner_ok and strict_result.syntax_valid),
        "schema_valid": bool(runner_ok and strict_result.schema_valid and tuple_limit_ok),
        "strict_format_valid": strict_format_valid,
        "recoverable_parse_valid": bool(runner_ok and recovery_result.recoverable_parse_valid),
        "canonical_wire_equal": bool(runner_ok and tuple_limit_ok and strict_result.canonical_wire_equal),
        "pred_tuple_count": pred_tuple_count_observed,
        "scored_pred_tuple_count": len(pred_quads),
        "gold_tuple_count": len(gold_quads),
        "tuple_count_correct": tuple_count_correct,
        "tuple": {
            "hard": {**hard_counts, "correct": hard_correct, "assignment": hard_assignment.as_dict()},
            "soft": {**soft_counts, "correct": soft_correct, "assignment": soft_assignment.as_dict()},
        },
        "field_unbound": field_unbound,
        "field_bound": {
            "alignment": bound_assignment.as_dict(),
            "targeted_group": targeted_group,
            "hateful": hateful,
            "group_hate_joint": joint,
        },
        "correctness": {
            "tuple/hard": hard_correct,
            "tuple/soft": soft_correct,
            "field_unbound/target": field_unbound["target"]["exact_correct"],
            "field_unbound/argument": field_unbound["argument"]["exact_correct"],
            "field_bound/targeted_group": targeted_group["correct"],
            "field_bound/hateful": hateful["correct"],
            "field_bound/group_hate_joint": joint["correct"],
            "format/strict": strict_format_valid,
            "tuple_count": tuple_count_correct,
        },
        "errors": errors,
        "warnings": warnings,
        "metric_contract": {
            "similarity": SIMILARITY_VERSION,
            "assignment": ASSIGNMENT_VERSION,
            "soft_threshold": soft_threshold,
            "invalid_prediction_policy": "empty-prediction-retain-denominator/v1",
        },
    }
    return result


def _sum_path(records: Sequence[Mapping[str, Any]], *path: str) -> int:
    total = 0
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        total += int(value)
    return total


def aggregate_query_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate sufficient statistics without averaging per-query F1."""

    if not records:
        raise ValueError("cannot aggregate an empty expected query frame")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate query id in evaluation frame")
    n = len(records)

    tuple_summary: dict[str, Any] = {}
    for mode in ("hard", "soft"):
        tuple_summary[mode] = _prf(
            _sum_path(records, "tuple", mode, "tp"),
            _sum_path(records, "tuple", mode, "fp"),
            _sum_path(records, "tuple", mode, "fn"),
        )
        tuple_summary[mode]["query_exact_rate"] = (
            sum(bool(record["tuple"][mode]["correct"]) for record in records) / n
        )
    tuple_summary["f1_avg"] = (tuple_summary["hard"]["f1"] + tuple_summary["soft"]["f1"]) / 2.0

    field_unbound: dict[str, Any] = {}
    for field in ("target", "argument", "targeted_group", "hateful"):
        exact = _prf(
            _sum_path(records, "field_unbound", field, "exact_tp"),
            _sum_path(records, "field_unbound", field, "exact_fp"),
            _sum_path(records, "field_unbound", field, "exact_fn"),
        )
        exact["query_exact_rate"] = (
            sum(bool(record["field_unbound"][field]["exact_correct"]) for record in records) / n
        )
        if field in {"target", "argument"}:
            exact["similarity"] = fmean(
                float(record["field_unbound"][field]["similarity"]) for record in records
            )
        field_unbound[field] = exact

    field_bound: dict[str, Any] = {}
    for field in ("targeted_group", "hateful", "group_hate_joint"):
        summary = _prf(
            _sum_path(records, "field_bound", field, "tp"),
            _sum_path(records, "field_bound", field, "fp"),
            _sum_path(records, "field_bound", field, "fn"),
        )
        denominator = summary["tp"] + summary["fp"] + summary["fn"]
        summary["bound_tuple_accuracy"] = summary["tp"] / denominator if denominator else 0.0
        summary["bound_query_exact_rate"] = (
            sum(bool(record["field_bound"][field]["correct"]) for record in records) / n
        )
        field_bound[field] = summary

    atom = _prf(
        _sum_path(records, "field_bound", "targeted_group", "atom", "tp"),
        _sum_path(records, "field_bound", "targeted_group", "atom", "fp"),
        _sum_path(records, "field_bound", "targeted_group", "atom", "fn"),
    )
    field_bound["targeted_group"]["atom"] = atom

    class_summaries: dict[str, Any] = {}
    for label in HATEFUL_LABELS:
        class_summaries[label] = _prf(
            _sum_path(records, "field_bound", "hateful", "class_counts", label, "tp"),
            _sum_path(records, "field_bound", "hateful", "class_counts", label, "fp"),
            _sum_path(records, "field_bound", "hateful", "class_counts", label, "fn"),
        )
    field_bound["hateful"]["classes"] = class_summaries
    field_bound["hateful"]["macro_f1"] = fmean(
        float(class_summaries[label]["f1"]) for label in HATEFUL_LABELS
    )

    error_codes: Counter[str] = Counter()
    for record in records:
        error_codes.update(str(error["code"]) for error in record.get("errors", []))
    multi = [record for record in records if int(record["gold_tuple_count"]) > 1]
    return {
        "schema_version": "stage1-evaluation-summary/v1",
        "query_count": n,
        "tuple": tuple_summary,
        "field_unbound": field_unbound,
        "field_bound": field_bound,
        "format": {
            "strict_format_rate": sum(bool(record["strict_format_valid"]) for record in records) / n,
            "recoverable_parse_rate": sum(bool(record["recoverable_parse_valid"]) for record in records) / n,
            "canonical_wire_rate": sum(bool(record["canonical_wire_equal"]) for record in records) / n,
            "error_code_distribution": dict(sorted(error_codes.items())),
        },
        "tuple_count": {
            "accuracy": sum(bool(record["tuple_count_correct"]) for record in records) / n,
            "mae": fmean(
                abs(int(record["pred_tuple_count"]) - int(record["gold_tuple_count"]))
                for record in records
            ),
            "multi_tuple_query_count": len(multi),
            "multi_tuple_all_gold_recovered_rate": (
                sum(int(record["tuple"]["soft"]["fn"]) == 0 for record in multi) / len(multi)
                if multi
                else None
            ),
            "multi_tuple_exact_structure_rate": (
                sum(bool(record["tuple"]["hard"]["correct"]) for record in multi) / len(multi)
                if multi
                else None
            ),
        },
    }


def exact_mcnemar_p(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value (binomial with p=0.5)."""

    if n01 < 0 or n10 < 0:
        raise ValueError("McNemar counts must be non-negative")
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(n01, n10) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def validate_paired_records(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed instead of taking a silent ID/hash intersection."""

    if len(records_a) != len(records_b):
        raise ValueError("paired frames have different lengths")
    for ordinal, (left, right) in enumerate(zip(records_a, records_b, strict=True)):
        if str(left.get("id")) != str(right.get("id")):
            raise ValueError(f"paired id mismatch at ordinal {ordinal}")
        for field in ("content_sha256", "gold_sha256"):
            if left.get(field) != right.get(field):
                raise ValueError(f"paired {field} mismatch for id {left.get('id')}")


def flip_table(
    records_a: Sequence[Mapping[str, Any]],
    records_b: Sequence[Mapping[str, Any]],
    endpoint: str,
) -> dict[str, Any]:
    """Compute one endpoint-specific paired flip table."""

    validate_paired_records(records_a, records_b)
    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for left, right in zip(records_a, records_b, strict=True):
        a = bool(left["correctness"][endpoint])
        b = bool(right["correctness"][endpoint])
        counts[f"n{int(a)}{int(b)}"] += 1
    n = len(records_a)
    if n == 0:
        raise ValueError("cannot compute flips on an empty frame")
    wrong_a = counts["n00"] + counts["n01"]
    correct_a = counts["n10"] + counts["n11"]
    return {
        "schema_version": "stage1-flip-table/v1",
        "endpoint": endpoint,
        **counts,
        "n": n,
        "wrong_to_correct": counts["n01"] / n,
        "correct_to_wrong": counts["n10"] / n,
        "conditional_recovery": counts["n01"] / wrong_a if wrong_a else None,
        "conditional_harm": counts["n10"] / correct_a if correct_a else None,
        "net_flip": (counts["n01"] - counts["n10"]) / n,
        "exact_mcnemar_p": exact_mcnemar_p(counts["n01"], counts["n10"]),
    }


def canonical_json_sha256(value: Any) -> str:
    """Hash helper used by immutable evaluation targets."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_text(payload)


__all__ = [
    "ASSIGNMENT_VERSION",
    "DEFAULT_MAX_TUPLES",
    "DEFAULT_SOFT_THRESHOLD",
    "METRIC_SCHEMA_VERSION",
    "SIMILARITY_VERSION",
    "Assignment",
    "aggregate_query_metrics",
    "canonical_json_sha256",
    "evaluate_query",
    "exact_mcnemar_p",
    "flip_table",
    "optimal_assignment",
    "similarity_v1",
    "validate_paired_records",
]
