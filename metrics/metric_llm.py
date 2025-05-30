import os
import json
import time

from tqdm import tqdm
from loguru import logger
from difflib import SequenceMatcher
from typing import Optional, Tuple

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


def calculate_hard_metrics(ids: list[str], pred_data_dict: dict, gt_data_dict: dict):

    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    for idx, id in tqdm(enumerate(ids), desc="Calculate Hard Metric"):

        # 提取真实和预测的四元组列表
        gt_quads = gt_data_dict[id]
        pred_quads = pred_data_dict[id]

        # 预处理四元组列表
        target_quads = [convert_quad(q) for q in gt_quads]
        prediction_quads = [convert_quad(q) for q in pred_quads]

        actual_positives += len(target_quads)
        predicted_positives += len(prediction_quads)

        # 创建可变的副本用于匹配
        remaining_targets = target_quads.copy()
        
        # 进行四元组匹配
        for pred in prediction_quads:
            for i, targ in enumerate(remaining_targets):
                # 完整匹配四个属性
                if pred == targ:
                    true_positives += 1
                    del remaining_targets[i]  # 避免重复匹配
                    break
    
    # 计算指标
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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

    for idx, id in tqdm(enumerate(ids), desc="Calculate Soft Metric"):

        # 提取真实和预测的四元组列表
        gt_quads = gt_data_dict[id]
        pred_quads = pred_data_dict[id]

        # 获取预处理后的四元组列表
        gt_quads = [preprocess_quad(q) for q in gt_data_dict[id]]
        pred_quads = [preprocess_quad(q) for q in pred_data_dict[id]]

        actual_positives += len(gt_quads)
        predicted_positives += len(pred_quads)

        # 创建可变副本用于匹配
        remaining_targets = gt_quads.copy()

        # 遍历预测四元组
        for pred in pred_quads:
            for i, targ in enumerate(remaining_targets):
                # 相似度计算
                target_sim = SequenceMatcher(
                    None, 
                    targ["target"], 
                    pred["target"]
                ).ratio()
                
                argument_sim = SequenceMatcher(
                    None,
                    targ["argument"],
                    pred["argument"]
                ).ratio()

                # 多标签匹配（排序后比较）
                group_match = targ["targeted_group"] == pred["targeted_group"]
                
                # 关键匹配条件
                if (target_sim >= similarity_threshold and 
                    argument_sim >= similarity_threshold and 
                    group_match and 
                    targ["hateful"] == pred["hateful"]):
                    
                    true_positives += 1
                    del remaining_targets[i]  # 移除已匹配项
                    break  # 跳出当前循环

    # 计算指标
    precision = true_positives / predicted_positives if predicted_positives else 0
    recall = true_positives / actual_positives if actual_positives else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": true_positives,
        "predicted_positives": predicted_positives,
        "actual_positives": actual_positives
    }


class LLMmetrics:

    def __init__(self, output_dir: str = "./metrics/llm/"):
        self.output_dir = output_dir

        self.init_metric()

    def init_metric(self):
        self.total_f1_hard = 0
        self.total_f1_soft = 0
        self.avg_f1 = 0

        self.success = 0
        self.total = 0

    def _load_data_from_path(self, data_path: str) -> Tuple[dict, dict]:
        logger.info("========Reading data========")
        with open(data_path, 'r') as f:
            json_datas = json.load(f)
        
        datas = json_datas["results"]
        return self._load_data(datas)

    def _load_data_from_dict(self, data_list: list[dict]) -> Tuple[dict, dict]:
        logger.info("========Reading data========")
        return self._load_data(data_list)

    def _load_data(self, datas: list[dict]) -> Tuple[dict, dict]:

        gt_data_dict = {}
        pred_data_dict = {}
        for data in datas:
            
            self.total += 1
            if data["status"] != "success":
                continue

            self.success += 1

            content_id = str(data["id"])
            gt_data_dict[content_id] = data["gt_quadruples"]
            pred_data_dict[content_id] = data["pred_quadruples"]

        return pred_data_dict, gt_data_dict
    
    def _save_result(self, info_data: dict, score_dict: dict):

        logger.info("========Saving data========")
        os.makedirs(self.output_dir, exist_ok=True)
        
        model_name = info_data["model"]
        shot_num = info_data["shot_num"]
        seed = info_data["seed"]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"metric_{model_name}_{shot_num}_{seed}_{timestamp}.json")

        with open(file_path, 'w') as f:
            f.write(json.dumps({"info": info_data, "metrics": score_dict}, ensure_ascii=False, indent=2))
        
        logger.info("========Data Saved========")

    def run(
            self, 
            datas_list: Optional[list[dict]] = None, 
            data_path: Optional[str] = None,
            info_data: Optional[dict] = None,
            save_data: bool = False,
            similarity_threshold: float = 0.5
            ) -> Optional[dict]:
        
        self.init_metric()

        if isinstance(datas_list, list):
            pred_data_dict, gt_data_dict = self._load_data_from_dict(datas_list)
        elif isinstance(data_path, str):
            pred_data_dict, gt_data_dict = self._load_data_from_path(data_path)
        else:
            raise ValueError(f"Invaild Input, datas_list: {type(datas_list)} expected: list[dict], data_path: {type(data_path)} expected: str")
        
        logger.info('========Start Calculating The Score========')
        try:
            
            ids = [k for k in gt_data_dict.keys()]
            hard_metrics = calculate_hard_metrics(ids, pred_data_dict, gt_data_dict)
            soft_metrics = calculate_soft_metrics(ids, pred_data_dict, gt_data_dict, similarity_threshold)

        except Exception as e:
            logger.exception(e)
            logger.error(f"Runtime Error: {str(e)}", exc_info=True)
            exit()

        metric_dict = {"f1_hard": hard_metrics["f1"],
                      "f1_soft": soft_metrics["f1"],
                      "f1_avg": round((hard_metrics["f1"] + soft_metrics["f1"]) / 2, 4),
                      "success": self.success,
                      "total": self.total,
                      "success_rate": round(self.success / self.total, 4),
                      "hard": hard_metrics,
                      "soft": soft_metrics
                      }

        if save_data:
            if isinstance(info_data, dict):
                self._save_result(info_data, metric_dict)
            else:
                raise ValueError(f"Invaild Input, info_data: {type(info_data)} expected: dict")
        else:
            return metric_dict

if __name__ == "__main__":
    METRIC = LLMmetrics()
    metric = METRIC.run(data_path="few_shot/output/output_qwen3-8b_0_23333333.json")
    print(metric)
        
    
    

