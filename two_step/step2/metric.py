import os
import json

from loguru import logger

import json
from fuzzywuzzy import process
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

from difflib import SequenceMatcher

def normalize_groups(groups_str: str):
    """标准化群体类别表示：去除空格、排序、去重"""
    if groups_str == "non_hate":
        return groups_str
    groups = [g.strip() for g in groups_str.split(",")]
    return ",".join(sorted(set(groups)))

def is_match(pred, gold):
    """改进的硬匹配：处理多类别情况"""
    # 基础字段检查
    if pred['hateful'] != gold['hateful']:
        return False
    
    # 特殊处理仇恨类别
    pred_groups = normalize_groups(pred['targeted_group'])
    gold_groups = normalize_groups(gold['targeted_group'])
    return pred_groups == gold_groups


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

    # 计算匹配指标
    tp, fp, fn = compute_metrics(pred_list, gold_list, is_match)
    f1 = calculate_f1(tp, fp, fn)

    return f1


class Metrics:

    def __init__(self):

        self.total_f1 = 0

    def _load_data(self, datas):
        logger.info("正在读取数据...")
        
        results = datas

        self.metric_datas = []

        for result in results:
            if result["status"] != "success":
                continue

            gold_list = [
                {
                    "targeted_group": result["gt_targeted_group"],
                    "hateful": result["gt_hateful"],
                }
            ]
            pred_list = [
                {
                    "targeted_group": result["pred_targeted_group"],
                    "hateful": result["pred_hateful"],
                }
            ]
            self.metric_datas.append((pred_list, gold_list))

        self.data_len = len(self.metric_datas)
    

    def _calculate_score(self, pred_list: list[dict[str, str]], gold_list: list[dict[str, str]]):

        f1 = get_score(pred_list, gold_list)

        self.total_f1 += f1

    def run(self, datas) -> dict:
        logger.info('开始计算分数')
        self._load_data(datas)
        try:
            for idx, (pred_list, gold_list) in enumerate(self.metric_datas, start=1):
                self._calculate_score(pred_list, gold_list)

                if idx % 10 == 0:
                    logger.info(
                                f"进度: {idx}/{self.data_len} "
                                f"({idx/self.data_len:.1%})"
                            )
        except Exception as e:
            logger.exception(e)
            logger.error(f"运行时异常: {str(e)}", exc_info=True)
            exit()
        
        self.avg_f1 = self.total_f1 / self.data_len

        score_dict = {"f1": self.avg_f1}
        
        return score_dict

        
    
    

