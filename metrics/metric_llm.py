import os
import json
import time

from tqdm import tqdm
from loguru import logger
from difflib import SequenceMatcher
from typing import Optional, Tuple
from collections import defaultdict

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
        
        # 新增的字段级指标统计
        self.field_metrics = {
            'target_sim': 0.0,      # Target文本平均相似度
            'argument_sim': 0.0,    # Argument文本平均相似度
            'targeted_group': {     # 目标群体分类指标
                'tp': 0, 'fp': 0, 'fn': 0, 
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            },
            'hateful': {            # 仇恨性分类指标
                'tp': 0, 'fp': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0
            }
        }
        self.valid_pairs = 0  # 有效匹配对计数

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
            if data["status"] == "success":
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

    def calculate_field_metrics(self, pred_data_dict: dict, gt_data_dict: dict):
        """计算字段级别的评估指标"""
        tg_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        hate_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        total_target_sim = 0.0
        total_arg_sim = 0.0
        valid_pairs = 0
        
        # 遍历所有数据
        for id_, pred_quads in pred_data_dict.items():
            gt_quads = gt_data_dict.get(id_, [])
            
            # 预处理四元组
            pred_processed = [preprocess_quad(q) for q in pred_quads]
            gt_processed = [preprocess_quad(q) for q in gt_quads]
            
            # 获取对齐匹配
            matches = align_elements(pred_processed, gt_processed)
            
            # 处理每个匹配对
            for p_idx, g_idx in matches:
                pred_quad = pred_processed[p_idx]
                gt_quad = gt_processed[g_idx]
                
                # 1. 计算文本相似度
                target_sim = string_similarity(pred_quad['target'], gt_quad['target'])
                arg_sim = string_similarity(pred_quad['argument'], gt_quad['argument'])
                total_target_sim += target_sim
                total_arg_sim += arg_sim
                
                # 2. 计算目标群体分类指标
                pred_tg = ",".join(sorted(pred_quad['targeted_group']))
                gt_tg = ",".join(sorted(gt_quad['targeted_group']))
                
                if pred_tg == gt_tg:
                    tg_metrics['all']['tp'] += 1
                else:
                    tg_metrics['all']['fp'] += 1
                    tg_metrics['all']['fn'] += 1
                
                # 3. 计算仇恨性分类指标
                if pred_quad['hateful'] == gt_quad['hateful']:
                    hate_metrics['all']['tp'] += 1
                else:
                    hate_metrics['all']['fp'] += 1
                    hate_metrics['all']['fn'] += 1
                
                valid_pairs += 1
            
            # 处理未匹配的预测四元组（FP）
            matched_pred = {p_idx for p_idx, _ in matches}
            for p_idx in range(len(pred_processed)):
                if p_idx not in matched_pred:
                    tg_metrics['all']['fp'] += 1
                    hate_metrics['all']['fp'] += 1
            
            # 处理未匹配的真实四元组（FN）
            matched_gt = {g_idx for _, g_idx in matches}
            for g_idx in range(len(gt_processed)):
                if g_idx not in matched_gt:
                    tg_metrics['all']['fn'] += 1
                    hate_metrics['all']['fn'] += 1
        
        # 计算总体指标
        self.valid_pairs = valid_pairs if valid_pairs > 0 else 1
        self.field_metrics['target_sim'] = total_target_sim / self.valid_pairs
        self.field_metrics['argument_sim'] = total_arg_sim / self.valid_pairs
        
        # 计算目标群体指标
        tg = tg_metrics['all']
        precision = tg['tp'] / (tg['tp'] + tg['fp']) if (tg['tp'] + tg['fp']) > 0 else 0
        recall = tg['tp'] / (tg['tp'] + tg['fn']) if (tg['tp'] + tg['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.field_metrics['targeted_group'].update({
            'tp': tg['tp'], 'fp': tg['fp'], 'fn': tg['fn'],
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        })
        
        # 计算仇恨性指标
        hate = hate_metrics['all']
        precision = hate['tp'] / (hate['tp'] + hate['fp']) if (hate['tp'] + hate['fp']) > 0 else 0
        recall = hate['tp'] / (hate['tp'] + hate['fn']) if (hate['tp'] + hate['fn']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        self.field_metrics['hateful'].update({
            'tp': hate['tp'], 'fp': hate['fp'], 'fn': hate['fn'],
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        })
        
        return self.field_metrics


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
        
        pbar = tqdm(
            total=3,
            desc=f"Calculating Score",
            unit="item",
            dynamic_ncols=True,
            leave=True
        )
        try:
            ids = [k for k in gt_data_dict.keys()]
            hard_metrics = calculate_hard_metrics(ids, pred_data_dict, gt_data_dict)
            pbar.update(1)
            soft_metrics = calculate_soft_metrics(ids, pred_data_dict, gt_data_dict, similarity_threshold)
            pbar.update(1)
            field_metrics = self.calculate_field_metrics(pred_data_dict, gt_data_dict)
            pbar.update(1)

        except Exception as e:
            logger.exception(e)
            logger.error(f"Runtime Error: {str(e)}", exc_info=True)
            exit()

        metric_dict = {
            # 基础指标
            "f1_hard": hard_metrics["f1"],
            "f1_soft": soft_metrics["f1"],
            "f1_avg": round((hard_metrics["f1"] + soft_metrics["f1"]) / 2, 4),
            "success": self.success,
            "total": self.total,
            "success_rate": round(self.success / self.total, 4),
            "hard_precision": hard_metrics["precision"],
            "hard_recall": hard_metrics["recall"],
            "soft_precision": soft_metrics["precision"],
            "soft_recall": soft_metrics["recall"],
            
            # 新增字段级指标
            "field_metrics": {
                "text_similarity": {
                    "target_avg_sim": round(field_metrics['target_sim'], 4),
                    "argument_avg_sim": round(field_metrics['argument_sim'], 4)
                },
                "targeted_group": dict(field_metrics['targeted_group']),
                "hateful": dict(field_metrics['hateful']),
                "matched_pairs": self.valid_pairs
            }
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
    metric = METRIC.run(data_path="runner/output/qwen2.5-7b-instruct/rag-lex-1-vllm-720-mav-1.json", similarity_threshold=0.25)
    print(json.dumps(metric, indent=2, ensure_ascii=False))
        
    
    

