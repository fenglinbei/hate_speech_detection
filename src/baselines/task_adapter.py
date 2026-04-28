from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from tools.convert import output2triple
from utils.parser import (
    extract_triplets,
    parse_binary_label,
    parse_hatexplain_output,
    parse_llm_output_quad,
    parse_llm_output_trip,
    validate_quadruples,
)


STRUCTURED_GROUPS = ["Racism", "Region", "LGBTQ", "Sexism", "others", "non-hate"]
HATEXPLAIN_LABELS = ("hatespeech", "offensive", "normal")


def normalize_task_type(task_type: str | None) -> str:
    value = str(task_type or "structured").strip().lower()
    if value in {"cold", "cold_binary", "binary"}:
        return "cold_binary"
    if value in {"hatexplain", "hate_xplain"}:
        return "hatexplain"
    return "structured"


def task_type_from_configs(build_config: dict | None = None, runner_config: dict | None = None) -> str:
    build_config = build_config or {}
    runner_config = runner_config or {}
    baseline = runner_config.get("baseline") or build_config.get("baseline") or {}
    task_type = (
        baseline.get("task_type")
        or runner_config.get("tester", {}).get("task_type")
        or build_config.get("task_type")
        or "structured"
    )
    return normalize_task_type(task_type)


def stable_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_binary_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return "hate" if value else "non-hate"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "hate" if int(value) == 1 else "non-hate"
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"1", "hate", "hateful", "toxic", "offensive", "abusive"}:
        return "hate"
    if text in {"0", "non-hate", "nonhate", "not-hate", "normal", "clean"}:
        return "non-hate"
    return None


def binary_label_from_quadruples(quadruples: Any) -> str:
    if isinstance(quadruples, dict):
        quadruples = [quadruples]
    if not isinstance(quadruples, list):
        return binary_label_from_text(quadruples)
    for quad in quadruples:
        if not isinstance(quad, dict):
            continue
        label = normalize_binary_label(quad.get("hateful"))
        if label == "hate":
            return "hate"
        group = str(quad.get("targeted_group", "")).strip().lower().replace("_", "-")
        if group and group != "non-hate":
            return "hate"
    return "non-hate"


def binary_label_from_text(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    label = normalize_binary_label(text)
    if label:
        return label
    if re.search(r"\b(?:non|not)\s*-?\s*hate(?:ful)?\b", text):
        return "non-hate"
    parts = [part.strip() for part in text.replace("[end]", "").split("|")]
    if len(parts) >= 4:
        label = normalize_binary_label(parts[3])
        if label:
            return label
    if len(parts) >= 3 and parts[2] and parts[2] != "non-hate":
        return "hate"
    return "non-hate" if "non-hate" in text else "hate"


def binary_label_from_record(record: dict) -> str:
    for key in ("gt_label", "pred_label", "hateful", "label", "output"):
        if key in record:
            label = parse_binary_label(str(record.get(key)))
            if label:
                return label
    return binary_label_from_quadruples(record.get("quadruples", record.get("gt_quadruples", [])))


def hatexplain_annotation_from_record(record: dict) -> dict[str, Any]:
    annotation = record.get("annotation") or record.get("gt_annotation") or record.get("pred_annotation") or {}
    label = str(annotation.get("label", "")).strip().lower()
    target_groups = annotation.get("target_groups", []) or []
    rationales = annotation.get("rationales", []) or []
    rationale_texts = []
    for rationale in rationales:
        if isinstance(rationale, dict):
            text = str(rationale.get("text", "")).strip()
        else:
            text = str(rationale).strip()
        if text:
            rationale_texts.append(text)
    return {
        "label": label,
        "target_groups": [str(group).strip() for group in target_groups if str(group).strip()],
        "rationales": rationale_texts,
    }


def canonical_hatexplain_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    normalized = hatexplain_annotation_from_record({"annotation": annotation})
    return {
        "label": normalized["label"],
        "target_groups": sorted(set(normalized["target_groups"])),
        "rationales": sorted(set(normalized["rationales"])),
    }


def structured_output_to_triples(value: Any) -> str:
    triples = []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        for quad in value:
            if isinstance(quad, dict):
                triples.append(f"{quad.get('target', '')} | {quad.get('argument', '')} | {quad.get('targeted_group', '')}")
            else:
                triples.append(str(quad).replace("[END]", "").strip())
    elif isinstance(value, str):
        raw_text = value.replace("[END]", "").strip()
        for seq in raw_text.split("[SEP]"):
            parts = [part.strip() for part in seq.split("|")]
            if len(parts) >= 3:
                triples.append(f"{parts[0]} | {parts[1]} | {parts[2]}")
    if triples:
        return (" [SEP] ".join(triple for triple in triples if triple) + " [END]").strip()
    try:
        return output2triple(value)
    except Exception:
        return "[END]"


def structured_output_from_record(record: dict) -> str:
    if isinstance(record.get("output"), str):
        return structured_output_to_triples(record["output"])
    triples_source = record.get("quadruples", record.get("gt_quadruples", []))
    return structured_output_to_triples(triples_source)


def gold_output(record: dict, task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    if task_type == "cold_binary":
        return binary_label_from_record(record)
    if task_type == "hatexplain":
        return json.dumps(hatexplain_annotation_from_record(record), ensure_ascii=False, separators=(",", ":"))
    return structured_output_from_record(record)


def render_demo(record: dict, example_template: str, task_type: str) -> str:
    return (
        example_template.replace("{retrieve_content}", str(record.get("content", "")))
        .replace("{retrieve_output}", gold_output(record, task_type))
    )


def record_classes(record: dict, task_type: str) -> list[str]:
    task_type = normalize_task_type(task_type)
    if task_type == "cold_binary":
        return [binary_label_from_record(record)]
    if task_type == "hatexplain":
        label = hatexplain_annotation_from_record(record).get("label", "")
        return [label] if label else []
    groups = []
    for quad in record.get("quadruples", record.get("gt_quadruples", [])) or []:
        raw = str(quad.get("targeted_group", "")).replace("|", ",").replace("/", ",").replace(";", ",")
        for part in raw.split(","):
            group = part.strip()
            if group and group not in groups:
                groups.append(group)
    return groups


def reason_only_prompt(text: str, task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    if task_type == "hatexplain":
        return (
            "You are a hate speech analyst. Read the text and write one short Reason: line "
            "that identifies the key cues needed to classify it. Do not output JSON.\n\n"
            f"Text:\n{text}\nReason:"
        )
    return (
        "你是一个内容审查专家。请阅读句子，并给出以“Reason:”开头的一段简短推理"
        "（不超过120字），用于指导之后选择类似示例。不要输出最终答案。\n\n"
        f"### 句子：\n{text}\n### 分析："
    )


def reason_and_answer_prompt(text: str, examples: str, lexicons: str, task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    if task_type == "cold_binary":
        return (
            "请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。\n\n"
            "背景知识：\n"
            f"{lexicons}\n\n"
            "示例：\n"
            f"{examples}\n\n"
            "请先输出一行以 Reason: 开头的简短推理，然后在“最终标签：”后只输出 hate 或 non-hate。\n\n"
            f"文本：{text}\n\n"
            "Reason:\n最终标签："
        )
    if task_type == "hatexplain":
        return (
            "Classify the text for the HateXplain task and identify target groups and rationales.\n\n"
            "Background lexicon:\n"
            f"{lexicons}\n\n"
            "Examples:\n"
            f"{examples}\n\n"
            "First output one short Reason: line. Then write FINAL_JSON: followed by one valid JSON object "
            "with exactly the keys label, target_groups, rationales.\n\n"
            f"Text:\n{text}\n\n"
            "Reason:\nFINAL_JSON:"
        )
    return (
        "你是一个内容审查专家，请你分析我的句子并且从中提取出一个或者多个三元组。\n"
        "targeted_group 必须且只能从 {Racism, Sexism, LGBTQ, Region, others, non-hate} 中选择。\n\n"
        "背景知识：\n"
        f"{lexicons}\n\n"
        "示例：\n"
        f"{examples}\n\n"
        f"### 句子：\n{text}\n\n"
        "请先输出一行以 Reason: 开头的简短推理，然后在“### 最终三元组：”后只输出三元组。\n"
        "### 最终三元组："
    )


def build_scoring_prompt_no_demo(text: str, task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    if task_type == "cold_binary":
        return (
            "请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。\n"
            "只输出一个标签：hate 或 non-hate。\n\n"
            f"文本：{text}\n标签："
        )
    if task_type == "hatexplain":
        return (
            "Classify the text for the HateXplain task. Return one valid JSON object with keys "
            '"label", "target_groups", "rationales".\n\n'
            f"Text:\n{text}\nJSON:"
        )
    return (
        "给定一句中文社交媒体文本，请抽取其中所有三元组：target | argument | targeted_group。\n"
        "多个三元组用 [SEP] 分隔，最后用 [END] 结束。\n\n"
        f"文本：{text}\n三元组："
    )


def build_scoring_prompt_with_one_demo(demo: dict, text: str, task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    demo_x = str(demo.get("content", ""))
    demo_y = gold_output(demo, task_type)
    if task_type == "cold_binary":
        return (
            "请判断下面文本是否属于仇恨言论二分类任务中的 hate 或 non-hate。\n"
            "只输出一个标签：hate 或 non-hate。\n\n"
            f"示例文本：{demo_x}\n示例标签：{demo_y}\n\n"
            f"文本：{text}\n标签："
        )
    if task_type == "hatexplain":
        return (
            "Classify the text for the HateXplain task. Return one valid JSON object with keys "
            '"label", "target_groups", "rationales".\n\n'
            f"Example text:\n{demo_x}\nExample JSON:\n{demo_y}\n\n"
            f"Text:\n{text}\nJSON:"
        )
    return (
        "给定一句中文社交媒体文本，请抽取其中所有三元组：target | argument | targeted_group。\n"
        "多个三元组用 [SEP] 分隔，最后用 [END] 结束。\n\n"
        f"示例文本：{demo_x}\n示例三元组：{demo_y}\n\n"
        f"文本：{text}\n三元组："
    )


def extract_reason(text: str) -> str:
    if not text:
        return ""
    for key in ("Reason:", "Reason："):
        idx = text.find(key)
        if idx >= 0:
            return text[idx:].splitlines()[0].strip()
    lines = str(text).strip().splitlines()
    return lines[0].strip() if lines else ""


def normalize_triple_key(triple: str) -> str:
    return " ".join(str(triple).strip().split())


def split_triples(answer: str) -> list[str]:
    triples_text = extract_triplets(answer) or str(answer or "")
    triples_text = triples_text.replace("[END]", "").strip()
    return [part.strip() for part in triples_text.split("[SEP]") if part.strip()]


def join_triples(triples: list[str]) -> str:
    return (" [SEP] ".join(triples) + " [END]") if triples else "[END]"


def vote_answers(answers: list[str], task_type: str, q: int | None = None) -> str:
    task_type = normalize_task_type(task_type)
    need = ((q if q is not None else len(answers)) // 2) + 1
    if task_type == "cold_binary":
        labels = [parse_binary_label(answer) for answer in answers]
        labels = [label for label in labels if label]
        if not labels:
            return ""
        label, count = Counter(labels).most_common(1)[0]
        return label if count >= need else labels[-1]
    if task_type == "hatexplain":
        key_to_json = {}
        keys = []
        for answer in answers:
            annotation = parse_hatexplain_output(answer)
            if not annotation:
                continue
            canonical = canonical_hatexplain_annotation(annotation)
            key = stable_dumps(canonical)
            key_to_json.setdefault(key, json.dumps(canonical, ensure_ascii=False, separators=(",", ":")))
            keys.append(key)
        if not keys:
            return ""
        key, count = Counter(keys).most_common(1)[0]
        return key_to_json[key] if count >= need else key_to_json[keys[-1]]

    counts: dict[str, int] = {}
    exemplar: dict[str, str] = {}
    last_valid: list[str] = []
    for answer in answers:
        triples = split_triples(answer)
        if triples:
            last_valid = triples
        seen = set()
        for triple in triples:
            key = normalize_triple_key(triple)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
            exemplar.setdefault(key, triple)
    voted = [exemplar[key] for key, count in counts.items() if count >= need]
    voted.sort(key=lambda item: (-counts[normalize_triple_key(item)], item))
    if not voted:
        voted = last_valid
    return join_triples(voted)


def parse_answer(answer: str, task_type: str) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    if task_type == "cold_binary":
        label = parse_binary_label(answer)
        return {"valid": bool(label), "answer": label or "", "pred_label": label}
    if task_type == "hatexplain":
        annotation = parse_hatexplain_output(answer)
        return {"valid": bool(annotation), "answer": json.dumps(annotation, ensure_ascii=False) if annotation else "", "pred_annotation": annotation}

    triples_text = extract_triplets(answer) or answer
    quadruples = parse_llm_output_quad(answer)
    if not quadruples:
        quadruples = parse_llm_output_trip(triples_text)
    return {
        "valid": validate_quadruples(quadruples),
        "answer": triples_text,
        "pred_quadruples": quadruples,
    }


def result_record(base_item: dict, answer: str, raw_outputs: list[str], task_type: str, attempts: int) -> dict[str, Any]:
    task_type = normalize_task_type(task_type)
    parsed = parse_answer(answer, task_type)
    record = {
        **base_item,
        "llm_output": answer if answer else None,
        "ids_raw_outputs": raw_outputs,
        "status": "success" if parsed["valid"] else "invalid",
        "attempts": attempts,
    }
    if task_type == "cold_binary":
        record["pred_label"] = parsed.get("pred_label")
    elif task_type == "hatexplain":
        record["pred_annotation"] = parsed.get("pred_annotation")
        record["pred_quadruples"] = []
    else:
        record["pred_quadruples"] = parsed.get("pred_quadruples", [])
    if not parsed["valid"]:
        record["error"] = "Output validation failed"
    return record
