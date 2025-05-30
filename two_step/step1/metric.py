import os
import json

from loguru import logger

import json
from fuzzywuzzy import process
from difflib import SequenceMatcher
from typing import List, Tuple, Optional

from difflib import SequenceMatcher

def string_similarity(a, b):
    """处理含 None 值的字符串相似度计算"""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def align_elements(pred, gt):
    """通过贪心算法建立元素间最优匹配关系"""
    gt__used = [False] * len(gt)
    matches = []
    
    for p_idx, p_elem in enumerate(pred):
        best_score = -1
        best_g_idx = -1
        # 计算元素综合相似度（取target与argument均值）
        for g_idx, g_elem in enumerate(gt):
            if not gt__used[g_idx]:
                target_sim = string_similarity(p_elem['target'], g_elem['target'])
                arg_sim = string_similarity(p_elem['argument'], g_elem['argument'])
                total_sim = (target_sim + arg_sim) / 2
                if total_sim > best_score:
                    best_score = total_sim
                    best_g_idx = g_idx
        if best_g_idx != -1:
            matches.append( (p_idx, best_g_idx) )
            gt__used[best_g_idx] = True
    return matches

def get_similarity(pred, gt) -> tuple[float, float]:
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
    
    return round(target_score, 4), round(arg_score, 4)

def label_match(pred: list[str], gt: list[str]) -> bool:
    gt_len = len(gt)
    pred_len = len(pred)

    if gt_len != pred_len:
        return False
    
    match_len = 0
    for label in gt:
        if label in pred:
            match_len += 1
    
    return match_len == gt_len

def hard


class Metrics:

    def __init__(self, data_path: Optional[str] = None):

        # 类参数
        self.data_path = data_path

        # 统计参数
        self.total_target_sim = 0
        self.total_arg_sim = 0
        self.avg_sim = 0
        self.total = 0
        self.success = 0

    def _load_data(self, datas: Optional[list[dict]] = None, data_path: Optional[str] = None):
        logger.info("正在读取数据...")
        
        assert isinstance(data_path, str) or isinstance(datas, list), "未提供合法参数"

        if isinstance(data_path, str):
            with open(data_path, "r") as f:
                results = json.load(f)["results"]
        elif isinstance(datas, list):
            results = datas
        else:
            raise ValueError("未提供合法参数")

        self.metric_datas = []

        for result in results:
            self.total += 1
            if result["status"] != "success":
                continue
            self.success += 1

            print(result)
            gt__quads = result["quadruples"]
            gt__list = [
                {
                    "target": q["target"],
                    "argument": q["argument"],
                }
                for q in gt__quads
            ]
            pred_list = result["parsed_quadruples"]

            self.metric_datas.append((pred_list, gt__list))

        self.data_len = len(self.metric_datas)
    

    def _calculate_score(self, pred_list: list[dict[str, str]], gt__list: list[dict[str, str]]):

        target_sim, arg_sim = get_similarity(pred_list, gt__list)

        self.total_target_sim += target_sim
        self.total_arg_sim += arg_sim

    def run(self, datas: Optional[list[dict]] = None) -> dict:
        logger.info('开始计算分数')
        self._load_data(datas) if datas else self._load_data(data_path=self.data_path)
        try:
            for idx, (pred_list, gt__list) in enumerate(self.metric_datas, start=1):
                self._calculate_score(pred_list, gt__list)

                if idx % 10 == 0:
                    logger.info(
                                f"进度: {idx}/{self.data_len} "
                                f"({idx/self.data_len:.1%})"
                            )
        except Exception as e:
            logger.exception(e)
            logger.error(f"运行时异常: {str(e)}", exc_info=True)
            exit()
        
        self.avg_target_sim = self.total_target_sim / self.data_len
        self.avg_arg_sim = self.total_arg_sim / self.data_len
        self.avg = (self.total_target_sim + self.total_arg_sim) / 2 / self.data_len

        score_dict = {"target_sim": self.avg_target_sim,
                      "arg_sim": self.avg_arg_sim,
                      "avg": self.avg,
                      "success": self.success,
                      "total": self.total,
                      "success_rate": self.success / self.total}
        
        return score_dict
    
    def save_metric(self, score_dict: dict):
        if isinstance(self.data_path, str):
            with open(self.data_path, "r") as f:
                data = json.load(f)
            
            data["metric"] = score_dict

            with open(self.data_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    METRIC = Metrics(data_path="/workspace/two_step/step1/result/output_qwen2.5-7b-instruct_5_23333333_20250517_090431.json")
    score_dict = METRIC.run()
    METRIC.save_metric(score_dict)
        
    
    

