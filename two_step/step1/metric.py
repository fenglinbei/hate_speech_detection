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

def align_elements(pred, gold):
    """通过贪心算法建立元素间最优匹配关系"""
    gold_used = [False] * len(gold)
    matches = []
    
    for p_idx, p_elem in enumerate(pred):
        best_score = -1
        best_g_idx = -1
        # 计算元素综合相似度（取target与argument均值）
        for g_idx, g_elem in enumerate(gold):
            if not gold_used[g_idx]:
                target_sim = string_similarity(p_elem['target'], g_elem['target'])
                arg_sim = string_similarity(p_elem['argument'], g_elem['argument'])
                total_sim = (target_sim + arg_sim) / 2
                if total_sim > best_score:
                    best_score = total_sim
                    best_g_idx = g_idx
        if best_g_idx != -1:
            matches.append( (p_idx, best_g_idx) )
            gold_used[best_g_idx] = True
    return matches

def get_similarity(pred, gold) -> tuple[float, float]:
    """改进后的相似度计算函数"""
    # 空值处理
    pred = pred or []
    gold = gold or []
    
    matches = align_elements(pred, gold)
    max_len = max(len(pred), len(gold)) or 1  # 防零除
    
    # 计算目标项相似度
    target_score = sum(
        string_similarity(pred[p]['target'], gold[g]['target'])
        for p, g in matches
    ) / max_len
    
    # 计算参数相似度
    arg_score = sum(
        string_similarity(pred[p]['argument'], gold[g]['argument'])
        for p, g in matches
    ) / max_len
    
    return round(target_score, 4), round(arg_score, 4)



class Metrics:

    def __init__(self, data_path: str):

        # 类参数
        self.data_path = data_path

        # 统计参数
        self.total_target_sim = 0
        self.total_arg_sim = 0
        self.avg_sim = 0
        self.total = 0
        self.success = 0

    def _load_data(self, data_path: str):
        logger.info("正在读取数据...")
        
        with open(data_path, "r") as f:
            datas = json.load(f)

        results = datas["results"]

        self.metric_datas = []

        for result in results:
            self.total += 1
            if result["status"] != "success":
                continue
            self.success += 1

            print(result)
            gold_quads = result["quadruples"]
            gold_list = [
                {
                    "target": q["target"],
                    "argument": q["argument"],
                }
                for q in gold_quads
            ]
            pred_list = result["parsed_quadruples"]

            self.metric_datas.append((pred_list, gold_list))

        self.data_len = len(self.metric_datas)
    

    def _calculate_score(self, pred_list: list[dict[str, str]], gold_list: list[dict[str, str]]):

        target_sim, arg_sim = get_similarity(pred_list, gold_list)

        self.total_target_sim += target_sim
        self.total_arg_sim += arg_sim

    def run(self) -> dict:
        logger.info('开始计算分数')
        self._load_data(self.data_path)
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

        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        data["etric"] = score_dict

        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    METRIC = Metrics(data_path="/workspace/two_step/step1/result/output_qwen2-7b-instruct-ft-202504250045-215b_0_23333333_20250425_101409.json")
    score_dict = METRIC.run()
    METRIC.save_metric(score_dict)
        
    
    

