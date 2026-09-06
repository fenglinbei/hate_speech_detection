"""Task projections and field-independent scoring for general-model L/D studies.

Extraction scores are unbound field diagnostics, not relation/tuple scores.
All query-level rates keep the supplied frame, including missing predictions.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any

from metrics.stage1_metrics import optimal_assignment, similarity_v1
from utils.quadruple import GROUP_ORDER, HATEFUL_LABELS, QUAD_KEYS, canonicalize_quadruples


TASKS = ("hate", "group", "extraction")
GROUP_LABELS = tuple(label for label in GROUP_ORDER if label != "non-hate")
POLICY_VERSION = "general-model-task-scoring/v2"
MAX_EXTRACTION_ITEMS = 128


def project_gold(quadruples: list[dict]) -> dict[str, Any]:
    """Project complete canonical gold without deriving hate from group labels."""
    if not isinstance(quadruples, list) or not quadruples:
        raise ValueError("gold must be a nonempty list of canonical quadruples")
    quads = canonicalize_quadruples(quadruples)
    groups = {label for quad in quads for label in quad.targeted_group}
    return {
        "hate": "hate" if any(quad.hateful == "hate" for quad in quads) else "non-hate",
        "group": [label for label in GROUP_LABELS if label in groups],
        "extraction": [
            {
                "target": quad.target,
                "argument": quad.argument,
                "targeted_group": list(quad.targeted_group),
                "hateful": quad.hateful,
            }
            for quad in quads
        ],
    }


def _error(code: str, path: str = "$", message: str = "") -> dict[str, str]:
    return {"code": code, "path": path, "message": message or code}


def _result(value: Any, valid: bool, status: str, errors: list[dict], **extra: Any) -> dict:
    return {"value": value, "valid": valid, "status": status, "errors": errors, **extra}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("nonfinite_json_constant")


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _decode_json(text: str, *, allow_embedded: bool = True) -> tuple[Any, str, list[dict]]:
    try:
        return _DECODER.decode(text), "json", []
    except ValueError as exc:
        if str(exc) in {"duplicate_json_key", "nonfinite_json_constant"}:
            return None, "invalid", [_error(str(exc))]
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        try:
            return _DECODER.decode(fence.group(1)), "json-fence", []
        except ValueError:
            return None, "invalid", [_error("invalid_json_fence")]
    if not allow_embedded:
        return None, "invalid", [_error("invalid_complete_json")]
    if text.startswith('"'):
        return None, "invalid", [_error("invalid_json")]
    if text[:1] in {"[", "{"}:
        try:
            _DECODER.raw_decode(text)
        except ValueError:
            return None, "invalid", [_error("invalid_json")]

    # Consume each complete container as a unit so its nested objects are not
    # mistaken for competing answers. Multiple top-level candidates are refused.
    candidates = []
    index = 0
    while index < len(text):
        if text[index] not in "[{":
            index += 1
            continue
        try:
            value, end = _DECODER.raw_decode(text, index)
        except ValueError as exc:
            if str(exc) in {"duplicate_json_key", "nonfinite_json_constant"}:
                return None, "invalid", [_error(str(exc))]
            index += 1
            continue
        candidates.append(value)
        index = end
    if len(candidates) == 1:
        return candidates[0], "embedded-json", []
    code = "multiple_json_answers" if candidates else "no_json_answer"
    return None, "invalid", [_error(code)]


def _group_value(value: Any, *, extraction: bool = False) -> list[str]:
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    allowed = GROUP_ORDER if extraction else GROUP_LABELS
    if not isinstance(value, list) or any(not isinstance(label, str) for label in value):
        raise ValueError("group_must_be_label_array")
    labels = [label.strip() for label in value]
    if any(label not in allowed for label in labels):
        raise ValueError("unknown_group_label")
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate_group_label")
    if extraction and (not labels or ("non-hate" in labels and len(labels) > 1)):
        raise ValueError("invalid_extraction_group")
    return [label for label in allowed if label in labels]


def _unwrap_classification(task: str, value: Any) -> tuple[Any, str | None]:
    if not isinstance(value, dict):
        return value, None
    allowed = ({"hate", "label", "hateful", "result"} if task == "hate" else
               {"group", "groups", "labels", "targeted_group", "result"})
    if len(value) != 1 or next(iter(value)) not in allowed:
        raise ValueError("ambiguous_classification_object")
    key = next(iter(value))
    return value[key], f"object-field:{key}"


def _classification_result(value: Any, valid: bool, status: str, errors: list[dict],
                           *, recovery_rule: str | None = None, **extra: Any) -> dict:
    strict_valid = valid and status == "ok"
    return _result(value, valid, status, errors, strict_valid=strict_valid,
                   format_status="canonical" if strict_valid else status,
                   recovery_rule=recovery_rule, **extra)


def _parse_classification(task: str, text: str) -> dict:
    if task == "hate" and text in HATEFUL_LABELS:
        return _classification_result(text, True, "recovered", [], format="bare-label",
                                      recovery_rule="bare-label")
    if task == "group" and all(part.strip() in GROUP_LABELS for part in text.split(",")):
        try:
            value = _group_value(text)
        except ValueError as exc:
            return _classification_result(None, False, "invalid", [_error(str(exc))])
        return _classification_result(value, True, "recovered", [], format="bare-labels",
                                      recovery_rule="bare-labels")

    # This is a complete-output grammar, not a label search inside prose or JSON.
    brace_label = re.fullmatch(r'\{\s*"(hate|non-hate)"\s*\}', text) if task == "hate" else None
    if brace_label:
        return _classification_result(brace_label.group(1), True, "recovered", [],
                                      format="brace-single-label", recovery_rule="brace-single-label")
    value, format_status, errors = _decode_json(text, allow_embedded=False)
    if errors:
        return _classification_result(None, False, "invalid", errors)
    recovery_rules = [] if format_status == "json" else [format_status]
    try:
        value, object_rule = _unwrap_classification(task, value)
        if object_rule:
            recovery_rules.append(object_rule)
        if task == "hate":
            if not isinstance(value, str) or value not in HATEFUL_LABELS:
                raise ValueError("invalid_hate_label")
        else:
            if isinstance(value, str):
                recovery_rules.append("group-string")
            elif isinstance(value, list) and any(isinstance(label, str) and label != label.strip()
                                                for label in value):
                recovery_rules.append("label-whitespace")
            value = _group_value(value)
    except ValueError as exc:
        return _classification_result(None, False, "invalid", [_error(str(exc))])
    return _classification_result(value, True, "recovered" if recovery_rules else "ok", [],
                                  format=format_status, recovery_rule="+".join(recovery_rules) or None)


def _extraction_field(field: str, value: Any) -> Any:
    if field in {"target", "argument"}:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("span_must_be_text_or_null")
        normalized = unicodedata.normalize("NFC", value).strip()
        if not normalized or normalized == "NULL":
            raise ValueError("empty_or_legacy_null_span")
        return normalized
    if field == "targeted_group":
        return _group_value(value, extraction=True)
    if value not in HATEFUL_LABELS:
        raise ValueError("invalid_hate_label")
    return value


def _parse_extraction(value: Any, format_status: str) -> dict:
    if isinstance(value, dict):
        value = [value]
        format_status += "+single-object"
    if not isinstance(value, list):
        return _result(None, False, "invalid", [_error("extraction_must_be_array_or_object")])
    if len(value) > MAX_EXTRACTION_ITEMS:
        return _result(None, False, "invalid", [_error("extraction_item_limit")])
    parsed, field_statuses, errors = [], [], []
    any_field = False
    for index, item in enumerate(value):
        fields = {}
        statuses = {}
        if not isinstance(item, dict):
            errors.append(_error("tuple_must_be_object", f"$[{index}]"))
            item = {}
        for extra in sorted(set(item) - set(QUAD_KEYS)):
            errors.append(_error("unknown_field", f"$[{index}].{extra}"))
        for field in QUAD_KEYS:
            if field not in item:
                statuses[field] = "missing"
                continue
            try:
                fields[field] = _extraction_field(field, item[field])
                statuses[field] = "valid"
                any_field = True
            except (ValueError, TypeError) as exc:
                statuses[field] = "invalid"
                errors.append(_error(str(exc), f"$[{index}].{field}"))
        parsed.append(fields)
        field_statuses.append(statuses)
    valid = any_field or not value
    incomplete = any(status != "valid" for item in field_statuses for status in item.values())
    status = "partial" if incomplete or errors else ("ok" if format_status == "json" else "recovered")
    if not valid:
        status = "invalid"
        errors.append(_error("no_scorable_fields"))
    return _result(parsed, valid, status, errors, format=format_status, field_statuses=field_statuses)


def parse_prediction(task: str, text: str, *, termination: str | None = None) -> dict[str, Any]:
    """Parse an answer without consulting gold or inferring absent labels.

    Extraction ``valid`` means at least one scorable field (or a valid empty
    array), not a complete tuple. Field statuses distinguish null from missing.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if task != "extraction" and termination == "length":
        return _classification_result(None, False, "invalid", [_error("truncated_output")])
    result = _result if task == "extraction" else _classification_result
    if text is None or (isinstance(text, str) and not text.strip()):
        return result(None, False, "missing", [_error("missing_output")])
    if not isinstance(text, str):
        return result(None, False, "invalid", [_error("output_must_be_text")])
    text = text.strip()
    if task != "extraction":
        return _parse_classification(task, text)
    value, format_status, errors = _decode_json(text)
    if errors:
        return _result(None, False, "invalid", errors)
    return _parse_extraction(value, format_status)


def _divide(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": _divide(tp, tp + fp),
            "recall": _divide(tp, tp + fn), "f1": _divide(2 * tp, 2 * tp + fp + fn)}


def _classification_metrics(task: str, records: list[dict]) -> dict:
    labels = HATEFUL_LABELS if task == "hate" else GROUP_LABELS
    counts = {label: Counter() for label in labels}
    correct = 0
    sample_f1 = 0.0
    invalid_empty_gold = 0
    for row in records:
        parsed = row["parsed"]
        gold = {row["gold"]} if task == "hate" else set(row["gold"])
        pred = ({parsed["value"]} if task == "hate" else set(parsed["value"])) if parsed["valid"] else set()
        row["correct"] = bool(parsed["valid"] and gold == pred)
        correct += row["correct"]
        for label in labels:
            counts[label]["tp"] += label in gold and label in pred
            counts[label]["fp"] += label not in gold and label in pred
            counts[label]["fn"] += label in gold and label not in pred
        score = _divide(2 * len(gold & pred), len(gold) + len(pred))
        if not gold and not pred and parsed["valid"]:
            score = 1.0
        row["sample_f1"] = score
        sample_f1 += score
        invalid_empty_gold += not gold and not parsed["valid"]
    per_label = {label: _prf(**{key: counts[label][key] for key in ("tp", "fp", "fn")}) for label in labels}
    totals = {key: sum(value[key] for value in counts.values()) for key in ("tp", "fp", "fn")}
    return {
        "accuracy" if task == "hate" else "exact_match": _divide(correct, len(records)),
        "correct_count": correct,
        "micro_f1": _prf(**totals)["f1"],
        "macro_f1": _divide(sum(value["f1"] for value in per_label.values()), len(labels)),
        "sample_f1": _divide(sample_f1, len(records)),
        "per_label": per_label,
        "invalid_empty_gold_count": invalid_empty_gold,
    }


def _value_key(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _extraction_metrics(records: list[dict]) -> dict:
    fields = {field: Counter() for field in QUAD_KEYS}
    for row in records:
        gold = row["gold"]
        predicted = row["parsed"]["value"] or []
        statuses = row["parsed"].get("field_statuses", [])
        row["field_unbound"] = {}
        for field in QUAD_KEYS:
            gold_values = [item[field] for item in gold]
            pred_values = [item[field] for item in predicted if field in item]
            matches = sum((Counter(map(_value_key, gold_values)) & Counter(map(_value_key, pred_values))).values())
            # Missing fields consume no prediction label. Surplus empty tuple
            # slots still count as excess output rather than disappearing.
            extra_slots = max(0, len(predicted) - max(len(gold), len(pred_values)))
            fp = len(pred_values) - matches + extra_slots
            fn = len(gold_values) - matches
            complete = len(pred_values) == len(predicted) == len(gold)
            result = {
                **_prf(matches, fp, fn),
                "exact_correct": bool(complete and matches == len(gold)),
                "gold_field_count": len(gold),
                "available_prediction_fields": len(pred_values),
                "missing_prediction_fields": sum(item[field] == "missing" for item in statuses),
                "invalid_prediction_fields": sum(item[field] == "invalid" for item in statuses),
                "answer_coverage": _divide(min(len(pred_values), len(gold)), len(gold)),
            }
            if field in {"target", "argument"}:
                assignment = optimal_assignment(len(pred_values), len(gold_values),
                                                lambda i, j: similarity_v1(pred_values[i], gold_values[j]))
                result["similarity"] = assignment.total_weight / max(len(predicted), len(gold), 1)
            row["field_unbound"][field] = result
            for key in ("tp", "fp", "fn", "gold_field_count", "available_prediction_fields",
                        "missing_prediction_fields", "invalid_prediction_fields", "answer_coverage"):
                fields[field][key] += result[key]
            fields[field]["exact_correct"] += result["exact_correct"]
            fields[field]["similarity"] += result.get("similarity", 0.0)
    summary = {}
    for field, values in fields.items():
        summary[field] = {
            **_prf(values["tp"], values["fp"], values["fn"]),
            "exact_match": _divide(values["exact_correct"], len(records)),
            "answer_coverage": _divide(values["answer_coverage"], len(records)),
            **{key: values[key] for key in ("gold_field_count", "available_prediction_fields",
                                          "missing_prediction_fields", "invalid_prediction_fields")},
        }
        if field in {"target", "argument"}:
            summary[field]["similarity"] = _divide(values["similarity"], len(records))
    return {"field_unbound": summary, "binding_scored": False, "tuple_f1_scored": False}


def evaluate_predictions(task: str, rows: list[dict]) -> dict[str, Any]:
    """Score exactly the supplied unique-query frame and preserve parse states."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    records = []
    seen = set()
    for row in rows:
        if row["query_id"] is None:
            raise ValueError("query IDs must be nonempty and unique")
        query_id = str(row["query_id"])
        if not query_id or query_id in seen:
            raise ValueError("query IDs must be nonempty and unique")
        seen.add(query_id)
        gold = row["gold"]
        if task == "hate":
            if not isinstance(gold, str) or gold not in HATEFUL_LABELS:
                raise ValueError("invalid projected hate gold")
        elif task == "group":
            if not isinstance(gold, list):
                raise ValueError("projected group gold must be a list")
            gold = _group_value(gold)
        else:
            gold = project_gold(gold)["extraction"]
        records.append({"query_id": query_id, "gold": gold, "prediction": row.get("prediction"),
                        "termination": row.get("termination"),
                        "parsed": parse_prediction(task, row.get("prediction"),
                                                   termination=row.get("termination"))})
    metrics = _extraction_metrics(records) if task == "extraction" else _classification_metrics(task, records)
    invalid = sum(not row["parsed"]["valid"] for row in records)
    metrics.update({"invalid_count": invalid, "invalid_rate": _divide(invalid, len(records)),
                    "valid_rate": _divide(len(records) - invalid, len(records)),
                    "status_counts": dict(Counter(row["parsed"]["status"] for row in records))})
    if task != "extraction":
        canonical = sum(row["parsed"]["strict_valid"] for row in records)
        recovered = sum(row["parsed"]["format_status"] == "recovered" for row in records)
        missing = sum(row["parsed"]["format_status"] == "missing" for row in records)
        strict_records = [{**row, "parsed": {**row["parsed"], "valid": row["parsed"]["strict_valid"]}}
                          for row in records]
        strict_metrics = _classification_metrics(task, strict_records)
        strict_metrics.update({"invalid_count": len(records) - canonical,
                               "invalid_rate": _divide(len(records) - canonical, len(records)),
                               "valid_rate": _divide(canonical, len(records))})
        for row, strict_row in zip(records, strict_records):
            row["strict_correct"] = strict_row["correct"]
            row["strict_sample_f1"] = strict_row["sample_f1"]
        metrics.update({"canonical_count": canonical, "canonical_rate": _divide(canonical, len(records)),
                        "recovered_count": recovered, "recovered_rate": _divide(recovered, len(records)),
                        "missing_count": missing, "missing_rate": _divide(missing, len(records)),
                        "format_status_counts": dict(Counter(row["parsed"]["format_status"] for row in records)),
                        "strict_metrics": strict_metrics})
    return {
        "schema_version": POLICY_VERSION,
        "task": task,
        "query_count": len(records),
        "metrics": metrics,
        "records": records,
        "policy": {
            "fixed_query_denominator": True,
            "macro_label_universe": list(HATEFUL_LABELS if task == "hate" else GROUP_LABELS) if task != "extraction" else None,
            "zero_division": 0,
            "group_empty_gold": "valid [] scores 1 in exact/sample-F1; invalid scores 0; positive-label F1 has no negative-label term",
            "extraction": "field-unbound exact multisets; target/argument similarity-v1 optimal assignment divided by max(gold tuple count, predicted tuple count, 1)",
            "max_extraction_items": MAX_EXTRACTION_ITEMS,
            "classification": (None if task == "extraction" else {
                "canonical_output": "JSON string label" if task == "hate" else "JSON array of exact labels",
                "strict_metrics": "same full frame; recovered, invalid and missing outputs are unscorable",
                "recovery_rate_denominator": "all supplied queries",
                "embedded_json_allowed": False,
                "recovery_consults_gold": False,
            }),
        },
    }
