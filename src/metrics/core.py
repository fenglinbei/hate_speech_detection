import json
from difflib import SequenceMatcher
from typing import Optional, Tuple

def normalize_groups(groups_str: str):
    """标准化群体类别表示：去除空格、排序、去重"""
    if groups_str == "non_hate":
        return groups_str
    groups = [g.strip() for g in groups_str.split(",")]
    return ",".join(sorted(set(groups)))

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

def get_score(pred_list: list[dict[str, str]], gt_list: list[dict[str, str]]) -> tuple[float, float]:
    
    # 计算硬匹配指标
    tp_hard, fp_hard, fn_hard = compute_metrics(pred_list, gt_list, is_hard_match)
    f1_hard = calculate_f1(tp_hard, fp_hard, fn_hard)
    
    # 计算软匹配指标
    tp_soft, fp_soft, fn_soft = compute_metrics(pred_list, gt_list, is_soft_match)
    f1_soft = calculate_f1(tp_soft, fp_soft, fn_soft)
    
    # 计算平均分
    avg_f1 = (f1_hard + f1_soft) / 2

    return f1_hard, f1_soft


def string_similarity(a: Optional[str], b: Optional[str]):
    """处理含 None 值的字符串相似度计算"""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def align_elements(
        pred: list[dict[str, str]], 
        gt: list[dict[str, str]]
        ) -> list[Tuple[int, int]]:
    """通过贪心算法建立元素间最优匹配关系"""

    gt_used = [False] * len(gt)
    matches = []
    
    for p_idx, p_elem in enumerate(pred):
        best_score = -1
        best_g_idx = -1
        # 计算元素综合相似度（取target与argument均值）
        for g_idx, g_elem in enumerate(gt):
            if not gt_used[g_idx]:
                target_sim = string_similarity(p_elem['target'], g_elem['target'])
                arg_sim = string_similarity(p_elem['argument'], g_elem['argument'])
                total_sim = (target_sim + arg_sim) / 2
                if total_sim > best_score:
                    best_score = total_sim
                    best_g_idx = g_idx
        if best_g_idx != -1:
            matches.append( (p_idx, best_g_idx) )
            gt_used[best_g_idx] = True
    return matches

def get_similarity(
        pred: list[dict[str, str]], 
        gt: list[dict[str, str]]
        ) -> tuple[float, float, list[Tuple[int, int]]]:
    """改进后的相似度计算函数"""

    # 空值处理
    pred = pred or []
    gt = gt or []
    
    matches = align_elements(pred, gt)
    max_len = max(len(pred), len(gt)) or 1  # 防零除
    
    # 计算目标项相似度
    target_score = sum(
        string_similarity(pred[p]['target'], gt[g]['target'])
        for p, g in matches
    ) / max_len
    
    # 计算参数相似度
    arg_score = sum(
        string_similarity(pred[p]['argument'], gt[g]['argument'])
        for p, g in matches
    ) / max_len
    
    return round(target_score, 4), round(arg_score, 4), matches

def convert_quad(quad):
    return (
        str(quad.get("target", "")).strip(),
        str(quad.get("argument", "")).strip(),
        str(quad.get("targeted_group", "")).strip().lower(),
        str(quad.get("hateful", "")).strip().lower()
    )

def is_hard_match(pred_quad, gt_quad):
    """
    判断预测四元组和标准答案是否硬匹配
    硬匹配：四元组的每个元素完全一致
    """
    return (pred_quad[0] == gt_quad[0] and
            pred_quad[1] == gt_quad[1] and
            pred_quad[2] == gt_quad[2] and
            pred_quad[3] == gt_quad[3])
import difflib
def calculate_similarity(pred_text, gt_text):
    """
    使用difflib.SequenceMatcher计算两个字符串的相似度
    """
    seq_matcher = difflib.SequenceMatcher(None, pred_text, gt_text)
    similarity = seq_matcher.ratio()
    return similarity


def is_soft_match(pred_quad, gt_quad, similarity_threshold: float = 0.5):
    """
    判断预测四元组和标准答案是否软匹配
    软匹配：Targeted_Group和Hateful完全一致，Target和Argument相似度>0.5
    """
    # 必须完全匹配的元素
    if (pred_quad["targeted_group"] != gt_quad["targeted_group"] or
            pred_quad["hateful"] != gt_quad["hateful"]):
        return False
    
    # 计算Target的相似度
    target_similarity = calculate_similarity(pred_quad["target"], gt_quad["target"])
    
    # 计算Argument的相似度
    argument_similarity = calculate_similarity(pred_quad["argument"], gt_quad["argument"])
    
    # 如果相似度都超过0.5则匹配成功
    return target_similarity > similarity_threshold and argument_similarity > similarity_threshold

def sort_by_hard_example(ids: list[str], pred_data_dict: dict, gt_data_dict: dict, alpha: float = 0.5, beta: float = 0.5) -> tuple[list[int], list[float], list[float]]:
    import numpy as np

    match_scores:list[float] = []
    label_scores:list[float] = []

    for idx, id in enumerate(ids):
        gt_quads = gt_data_dict[id]
        pred_quads = pred_data_dict[id]

        gt_quads = [convert_quad(q) for q in gt_quads]
        pred_quads = [convert_quad(q) for q in pred_quads]

        matched_ids = []

        for i, gt_quad in enumerate(gt_quads):
            best_match: int = 0
            best_match_score: float = 0.0
            for j, pred_quad in enumerate(pred_quads):
                if j in matched_ids:
                    continue
                sim_score = (calculate_similarity(pred_quad[0], gt_quad[0]) + calculate_similarity(pred_quad[1], gt_quad[1])) / 2
                if sim_score > best_match_score:
                    best_match_score = sim_score
                    best_match = j
            matched_ids.append(best_match)

        match_scores.append(sum([calculate_similarity(pred_quads[j][0], gt_quads[i][0]) + calculate_similarity(pred_quads[j][1], gt_quads[i][1]) for i, j in enumerate(matched_ids)]) / (2 * len(gt_quads)) if len(gt_quads) > 0 else 0.0)
        
        label_score = 0
        for i, j in enumerate(matched_ids):
            label_score += 1 if pred_quads[j][2] == gt_quads[i][2] and pred_quads[j][3] == gt_quads[i][3] else 0
        label_scores.append(label_score / len(gt_quads) if len(gt_quads) > 0 else 0.0)

    return list(np.argsort(alpha * np.array(match_scores) + beta * np.array(label_scores))), match_scores, label_scores

def calculate_hard_metrics(ids: list[str], pred_data_dict: dict, gt_data_dict: dict):

    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    # 收集每个示例的硬匹配和软匹配结果，用于计算总体F1
    all_hard_tp, all_hard_fp, all_hard_fn = 0, 0, 0

    for idx, id in enumerate(ids):

        # 提取真实和预测的四元组列表
        gt_quads = gt_data_dict[id]
        pred_quads = pred_data_dict[id]

        # 预处理四元组列表
        gt_quads = [convert_quad(q) for q in gt_quads]
        pred_quads = [convert_quad(q) for q in pred_quads]

        # 硬匹配评估
        hard_matched_pred = set()
        hard_matched_gold = set()

        for i, pred_quad in enumerate(pred_quads):
            for j, gt_quad in enumerate(gt_quads):
                if j in hard_matched_gold:
                    continue
                if is_hard_match(pred_quad, gt_quad):
                    hard_matched_pred.add(i)
                    hard_matched_gold.add(j)
                    break

        hard_tp = len(hard_matched_pred)
        hard_fp = len(pred_quads) - hard_tp
        hard_fn = len(gt_quads) - len(hard_matched_gold)
        
        all_hard_tp += hard_tp
        all_hard_fp += hard_fp
        all_hard_fn += hard_fn

    # 计算硬匹配的精确率、召回率和F1分数
    precision = all_hard_tp / (all_hard_tp + all_hard_fp) if (all_hard_tp + all_hard_fp) > 0 else 0
    recall = all_hard_tp / (all_hard_tp + all_hard_fn) if (all_hard_tp + all_hard_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "predicted_positives": predicted_positives,
        "actual_positives": actual_positives
    }

def preprocess_quad(quad):
    """统一处理四元组格式并标准化"""
    return {
        "target": str(quad.get("target", "")).strip(),
        "argument": str(quad.get("argument", "")).strip(),
        "targeted_group": sorted(str(quad.get("targeted_group", "")).lower().split(', ')),
        "hateful": str(quad.get("hateful", "")).lower().strip()
    }

def calculate_soft_metrics(ids: list[str], pred_data_dict: dict, gt_data_dict: dict, similarity_threshold: float = 0.5):
    """带软匹配的结构化四元组评估指标计算"""
    
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    all_soft_tp, all_soft_fp, all_soft_fn = 0, 0, 0

    for idx, id in enumerate(ids):

        # 提取真实和预测的四元组列表
        gt_quads = gt_data_dict[id]
        pred_quads = pred_data_dict[id]

        # 获取预处理后的四元组列表
        gt_quads = [preprocess_quad(q) for q in gt_data_dict[id]]
        pred_quads = [preprocess_quad(q) for q in pred_data_dict[id]]

        soft_matched_pred = set()
        soft_matched_gold = set()

        for i, pred_quad in enumerate(pred_quads):
            for j, gt_quad in enumerate(gt_quads):
                if j in soft_matched_gold:
                    continue
                if is_soft_match(pred_quad, gt_quad, similarity_threshold):
                    soft_matched_pred.add(i)
                    soft_matched_gold.add(j)
                    break
                # else:
                #     print(pred_quad, gt_quad)
        
        soft_tp = len(soft_matched_pred)
        soft_fp = len(pred_quads) - soft_tp
        soft_fn = len(gt_quads) - len(soft_matched_gold)
        
        all_soft_tp += soft_tp
        all_soft_fp += soft_fp
        all_soft_fn += soft_fn

    soft_precision = all_soft_tp / (all_soft_tp + all_soft_fp) if (all_soft_tp + all_soft_fp) > 0 else 0
    soft_recall = all_soft_tp / (all_soft_tp + all_soft_fn) if (all_soft_tp + all_soft_fn) > 0 else 0
    soft_f1 = 2 * soft_precision * soft_recall / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0

    return {
        "precision": round(soft_precision, 4),
        "recall": round(soft_recall, 4),
        "f1": round(soft_f1, 4),
        "true_positives": true_positives,
        "predicted_positives": predicted_positives,
        "actual_positives": actual_positives
    }
