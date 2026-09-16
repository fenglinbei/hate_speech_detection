"""Choice-first evidence review fields, independent of AI and human provenance."""

from __future__ import annotations

import copy
from typing import Any

from build_lex.annotated_lexicon_repair import LexiconRepairError as ReviewError


GROUPS = ("Racism", "Region", "LGBTQ", "Sexism", "others")
KINDS = ("query", "demo", "relation", "definition", "hit")
CHOICES = {
    "hate": {"hate": "仇恨", "non-hate": "非仇恨"},
    "group": dict(zip(GROUPS, ("种族", "地域", "性少数", "性别", "其他群体"))),
    "reason": {
        "agreed_attack": "作者认可攻击", "no_attack": "无攻击命题",
        "behavior_criticism": "仅批评行为", "opposes_attack": "反对攻击",
        "neutral_mention": "中性提及", "identity_target": "明确身份对象",
        "group_evaluation": "群体评价", "no_group_target": "无群体评价目标",
        "individual_scope": "个体范围未定义", "institution_scope": "机构范围未定义",
        "category_boundary": "类别边界不明确", "unclear_target": "指代对象不明",
        "unclear_stance": "立场不明", "unclear_sense": "词义不明",
        "missing_context": "缺少判别语境", "other": "其他原因",
    },
    "stance": {"supports": "赞同", "opposes": "反对", "reports": "转述", "sarcasm": "反讽", "self_reference": "自述", "neutral": "中性", "unclear": "不明"},
    "target_types": {"group": "群体", "individual": "个人", "institution": "机构", "behavior": "行为或观点", "none": "无明确对象", "unclear": "不明"},
    "expression_types": {"group_attack": "群体攻击", "identity_attack": "身份贬损", "individual_insult": "个体辱骂", "behavior_criticism": "行为批评", "quoted_rebuttal": "引用反驳", "neutral_or_laughter": "普通提及或笑声", "other": "其他"},
    "original_status": {"accepted": "认可原标注", "suspected_error": "同口径疑错", "policy_changed": "新规则下已裁决", "policy_ambiguous": "规则歧义", "context_insufficient": "缺少判别语境"},
    "relation": {"direct": "直接", "partial": "部分", "none": "无", "unclear": "不明"},
    "lexicon_risk": {"yes": "可能引入", "no": "未发现", "unclear": "不明"},
    "definition_verdict": {"reasonable": "合理", "too_narrow": "过窄", "other_problem": "其他问题", "uncertain": "未决"},
    "source_fit": {"valid_sense": "适配", "wrong_sense": "错义", "substring_mismatch": "子串误命中", "uncertain": "不明", "not_rendered": "未进入提示"},
    "query_fit": {"applicable": "适配", "inapplicable": "不适配", "uncertain": "不明"},
    "issues": {"valid_sense": "合理义项", "substring_mismatch": "子串误命中", "wrong_sense": "错义", "overly_narrow_definition": "定义过窄", "uncertain": "不明", "not_rendered": "未进入提示", "provenance_mismatch": "来源待核验"},
    "use": {"reference_analysis": "参考标签分析", "input_control": "输入对照候选", "behavior_only": "仅无 Gold 行为观察", "verify_first": "先核验", "defer": "暂缓"},
    "explanation_choice": {"defer": "暂不采纳解释", "adopt": "采纳提案", "modified": "修改后采纳"},
    "reopen_reason": {"misclick": "误选", "new_evidence": "新证据", "policy_update": "规则更新", "source_issue": "来源问题", "context_missing": "关键语境缺失", "source_refresh": "重新核对更新的资源", "other": "其他"},
    "exposure": {"not_seen": "未见过", "seen": "见过", "unsure": "不确定"},
}


def empty_values(kind: str) -> dict[str, Any]:
    common: dict[str, Any] = {"evidence": [], "note": ""}
    if kind in {"query", "demo"}:
        common.update(hate=None, group=None, hate_reason="", group_reason="", stance="unclear", target_types=[], expression_types=[])
        if kind == "demo":
            common.update(hate_original_status="", group_original_status="")
    elif kind == "relation":
        common.update(topic_hate="", topic_group="", rule_hate="", rule_group="", lexicon_risk="unclear")
    elif kind == "definition":
        common.update(definition_verdict="", issues=[])
    elif kind == "hit":
        common.update(source_fit="", query_fit="", issues=[])
    else:
        raise ReviewError("未知的审核对象类型。")
    return common


def empty_assessment() -> dict[str, Any]:
    return {"hate": None, "group": None, "hate_original_status": "", "group_original_status": "", "hate_use": "", "group_use": "", "explanation_choice": "defer", "note": ""}


def _choice(value: Any, name: str, *, required: bool) -> None:
    if not isinstance(value, str) or (value not in CHOICES[name] and (required or value != "")):
        raise ReviewError("请选择有效的选项：" + name)


def _labels(values: dict[str, Any]) -> None:
    if values["hate"] is not None and values["hate"] not in ("hate", "non-hate"):
        raise ReviewError("hate 只能为 hate、non-hate 或未决 null。")
    group = values["group"]
    if group is not None:
        if not isinstance(group, list) or any(not isinstance(x, str) or x not in GROUPS for x in group) or len(set(group)) != len(group):
            raise ReviewError("group 必须为五类的无重复集合、空集合 [] 或未决 null。")
        values["group"] = [x for x in GROUPS if x in group]


def _shape(value: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value) <= set(baseline):
        raise ReviewError("审核字段与当前模板不一致。")
    result = {**copy.deepcopy(baseline), **copy.deepcopy(value)}
    if not isinstance(result["note"], str) or len(result["note"]) > 2000:
        raise ReviewError("补充说明格式不正确或超过 2000 字符。")
    return result


def normalize_values(kind: str, value: Any, source: dict[str, Any], *, required: bool = False,
                     allow_policy_changed: bool = False, final_label_only: bool = False) -> dict[str, Any]:
    result = _shape(value, empty_values(kind))
    scalar: dict[str, str] = {}
    arrays: dict[str, str] = {}
    if kind in {"query", "demo"}:
        _labels(result)
        scalar.update(hate_reason="reason", group_reason="reason", stance="stance")
        arrays.update(target_types="target_types", expression_types="expression_types")
        if kind == "demo":
            scalar.update(hate_original_status="original_status", group_original_status="original_status")
    elif kind == "relation":
        scalar.update({x: "relation" for x in ("topic_hate", "topic_group", "rule_hate", "rule_group")})
        scalar["lexicon_risk"] = "lexicon_risk"
    elif kind == "definition":
        scalar["definition_verdict"] = "definition_verdict"
        arrays["issues"] = "issues"
    elif kind == "hit":
        scalar.update(source_fit="source_fit", query_fit="query_fit")
        arrays["issues"] = "issues"
    for field, choices in scalar.items():
        _choice(result[field], choices, required=required and not (final_label_only and field.endswith("_original_status")))
    if kind == "demo":
        _policy_changed_status(result, allow_policy_changed=allow_policy_changed)
    for field, choices in arrays.items():
        vals = result[field]
        if not isinstance(vals, list) or any(not isinstance(x, str) or x not in CHOICES[choices] for x in vals) or len(vals) != len(set(vals)):
            raise ReviewError("多选字段包含无效或重复值：" + field)
    if "target_types" in result and "none" in result["target_types"] and len(result["target_types"]) > 1:
        raise ReviewError("无明确对象不能与其他对象类型同时选择。")
    evidence = result["evidence"]
    texts = {"text": source.get("text", ""), **source.get("texts", {})}
    if not isinstance(evidence, list) or len(evidence) > 50:
        raise ReviewError("证据片段格式不正确。")
    for span in evidence:
        if not isinstance(span, dict) or set(span) != {"source", "start", "end", "text"}:
            raise ReviewError("证据必须包含原文来源和 Unicode 字符区间。")
        raw = texts.get(span["source"]) if isinstance(span["source"], str) else None
        start, end = span["start"], span["end"]
        if not isinstance(raw, str) or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(raw) or raw[start:end] != span["text"]:
            raise ReviewError("证据片段与原文 Unicode 字符区间不匹配。")
    if required and kind in {"query", "demo"}:
        if "other" in (result["hate_reason"], result["group_reason"]) and not result["note"].strip():
            raise ReviewError("其他原因需要一句简短说明；也可采用已有 AI 说明。")
        if result["hate"] == "hate" and result["hate_reason"] in {"no_attack", "opposes_attack", "neutral_mention", "behavior_criticism"}:
            raise ReviewError("hate 标签与当前依据相冲突，请改选依据或标签。")
        if result["group"] and result["group_reason"] == "no_group_target":
            raise ReviewError("group 标签与无群体评价目标的依据相冲突。")
        if kind == "demo" and not final_label_only:
            original = source.get("original_answer", source.get("answer", {}))
            validate_original_status(result, original, allow_policy_changed=allow_policy_changed)
    return result


def _policy_changed_status(values: dict[str, Any], *, allow_policy_changed: bool) -> None:
    if values.get("hate_original_status") == "policy_changed":
        raise ReviewError("本次规则变化仅适用于 group，不能用于 hate。")
    if values.get("group_original_status") == "policy_changed" and (not allow_policy_changed or values["group"] is None):
        raise ReviewError("新规则下已裁决需要已生效的 group 修订和非 null 的完整标签集合。")


def validate_original_status(values: dict[str, Any], original: dict[str, Any], *, allow_policy_changed: bool = False) -> None:
    _policy_changed_status(values, allow_policy_changed=allow_policy_changed)
    for task in ("hate", "group"):
        status = values[task + "_original_status"]
        if status in {"policy_ambiguous", "context_insufficient"} and values[task] is not None:
            raise ReviewError("规则或关键语境仍未解决时，该任务的正式标签须保留未决。")
        if status == "accepted" and (values[task] is None or task not in original or values[task] != original[task]):
            raise ReviewError("认可原标注需要该任务已裁决且与原标签一致。")


def normalize_assessment(value: Any, original: dict[str, Any], *, required: bool = False,
                         allow_policy_changed: bool = False) -> dict[str, Any]:
    result = _shape(value, empty_assessment())
    _labels(result)
    _policy_changed_status(result, allow_policy_changed=allow_policy_changed)
    for task in ("hate", "group"):
        _choice(result[task + "_original_status"], "original_status", required=required)
        _choice(result[task + "_use"], "use", required=required)
        if required and result[task + "_use"] == "reference_analysis" and result[task] is None:
            raise ReviewError("未决任务暂不能用作参考标签分析，请选择先核验、行为观察或暂缓。")
    _choice(result["explanation_choice"], "explanation_choice", required=True)
    if required:
        validate_original_status(result, original, allow_policy_changed=allow_policy_changed)
    return result
