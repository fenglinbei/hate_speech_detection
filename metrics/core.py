import json
from difflib import SequenceMatcher

def normalize_groups(groups_str: str):
    """标准化群体类别表示：去除空格、排序、去重"""
    if groups_str == "non_hate":
        return groups_str
    groups = [g.strip() for g in groups_str.split(",")]
    return ",".join(sorted(set(groups)))

def string_similarity(a, b):
    """处理 None 值的字符串相似度计算"""
    if a is None and b is None:
        return 1.0  # 双方均为 None 时视为完全匹配
    if a is None or b is None:
        return 0.0  # 单方为 None 时视为不匹配
    return SequenceMatcher(None, a, b).ratio()

def is_hard_match(pred, gold):
    """改进的硬匹配：处理多类别情况"""
    # 基础字段检查
    if pred['hateful'] != gold['hateful']:
        return False
    if pred['target'] != gold['target'] or pred['argument'] != gold['argument']:
        return False
    
    # 特殊处理仇恨类别
    pred_groups = normalize_groups(pred['targeted_group'])
    gold_groups = normalize_groups(gold['targeted_group'])
    return pred_groups == gold_groups

def is_soft_match(pred, gold):
    """改进的软匹配：处理多类别情况"""
    # 基础字段检查
    if pred['hateful'] != gold['hateful']:
        return False
    
    # 标准化群体类别
    pred_groups = set(normalize_groups(pred['targeted_group']).split(","))
    gold_groups = set(normalize_groups(gold['targeted_group']).split(","))
    
    # 仇恨场景需要完全匹配群体类别
    if pred['hateful'] == 'hate' and pred_groups != gold_groups:
        return False
    # 非仇恨场景检查字符串
    if pred['hateful'] == 'non_hate' and pred['targeted_group'] != gold['targeted_group']:
        return False
    
    # 文本相似度检查
    target_sim = string_similarity(pred['target'], gold['target'])
    arg_sim = string_similarity(pred['argument'], gold['argument'])
    return target_sim > 0.5 and arg_sim > 0.5

def compute_metrics(preds, golds, match_func):
    """统计 TP/FP/FN"""
    tp = 0
    matched_golds = set()
    matched_preds = set()
    # 遍历预测结果与标准答案匹配
    for i, pred in enumerate(preds):
        for j, gold in enumerate(golds):
            if j not in matched_golds and match_func(pred, gold):
                tp += 1
                matched_golds.add(j)
                matched_preds.add(i)
                break
    fp = len(preds) - len(matched_preds)
    fn = len(golds) - len(matched_golds)
    return tp, fp, fn

def calculate_f1(tp, fp, fn):
    """计算 F1 分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def get_score(pred_list: list[dict[str, str]], gold_list: list[dict[str, str]]) -> tuple[float, float]:
    
    # 计算硬匹配指标
    tp_hard, fp_hard, fn_hard = compute_metrics(pred_list, gold_list, is_hard_match)
    f1_hard = calculate_f1(tp_hard, fp_hard, fn_hard)
    
    # 计算软匹配指标
    tp_soft, fp_soft, fn_soft = compute_metrics(pred_list, gold_list, is_soft_match)
    f1_soft = calculate_f1(tp_soft, fp_soft, fn_soft)
    
    # 计算平均分
    avg_f1 = (f1_hard + f1_soft) / 2

    return f1_hard, f1_soft