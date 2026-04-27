import os
import json
import time

from tqdm import tqdm
from difflib import SequenceMatcher
from typing import Optional, Tuple
from collections import defaultdict

from metrics.core import *
from utils.log import init_logger

logger = init_logger(level="INFO", show_console=True)

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
                return metric_dict
            else:
                self._save_result({}, metric_dict)
                return metric_dict
                # raise ValueError(f"Invaild Input, info_data: {type(info_data)} expected: dict")
        else:
            return metric_dict
        

class BinaryClassificationMetrics:
    LABELS = ("hate", "non-hate")

    def __init__(self, output_dir: str = "./metrics/llm/"):
        self.output_dir = output_dir

    @staticmethod
    def normalize_label(value) -> Optional[str]:
        if isinstance(value, bool):
            return "hate" if value else "non-hate"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "hate" if int(value) == 1 else "non-hate"
        text = str(value or "").strip().lower().replace("_", "-")
        if text in {"1", "hate", "hateful"}:
            return "hate"
        if text in {"0", "non-hate", "nonhate", "not-hate"}:
            return "non-hate"
        return None

    def _extract_gt_label(self, row: dict) -> Optional[str]:
        label = self.normalize_label(row.get("gt_label"))
        if label:
            return label
        for quad in row.get("gt_quadruples", []) or []:
            label = self.normalize_label(quad.get("hateful"))
            if label == "hate":
                return "hate"
        return "non-hate" if row.get("gt_quadruples") else None

    def _load_pairs(self, datas: list[dict]) -> list[tuple[Optional[str], Optional[str], str]]:
        pairs = []
        for row in datas:
            gt = self._extract_gt_label(row)
            pred = self.normalize_label(row.get("pred_label"))
            status = str(row.get("status", ""))
            pairs.append((gt, pred, status))
        return pairs

    @staticmethod
    def _safe_div(num: int, den: int) -> float:
        return num / den if den else 0.0

    def run(
            self,
            datas_list: Optional[list[dict]] = None,
            data_path: Optional[str] = None,
            info_data: Optional[dict] = None,
            save_data: bool = False,
            **_,
            ) -> dict:
        if isinstance(datas_list, list):
            datas = datas_list
        elif isinstance(data_path, str):
            with open(data_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            datas = payload.get("results", []) if isinstance(payload, dict) else payload
        else:
            raise ValueError("BinaryClassificationMetrics requires datas_list or data_path.")

        pairs = self._load_pairs(datas)
        total = len(pairs)
        success = sum(1 for _, pred, status in pairs if status == "success" and pred in self.LABELS)
        correct = sum(1 for gt, pred, _ in pairs if gt in self.LABELS and gt == pred)
        invalid = sum(1 for _, pred, _ in pairs if pred not in self.LABELS)

        per_label = {}
        for label in self.LABELS:
            tp = sum(1 for gt, pred, _ in pairs if gt == label and pred == label)
            fp = sum(1 for gt, pred, _ in pairs if gt != label and pred == label)
            fn = sum(1 for gt, pred, _ in pairs if gt == label and pred != label)
            precision = self._safe_div(tp, tp + fp)
            recall = self._safe_div(tp, tp + fn)
            f1 = self._safe_div(2 * precision * recall, precision + recall)
            per_label[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": sum(1 for gt, _, _ in pairs if gt == label),
            }

        confusion_labels = ["hate", "non-hate", "invalid"]
        confusion = {gt_label: {pred_label: 0 for pred_label in confusion_labels} for gt_label in self.LABELS}
        for gt, pred, _ in pairs:
            if gt not in self.LABELS:
                continue
            pred_key = pred if pred in self.LABELS else "invalid"
            confusion[gt][pred_key] += 1

        macro_precision = sum(
            self._safe_div(per_label[label]["tp"], per_label[label]["tp"] + per_label[label]["fp"])
            for label in self.LABELS
        ) / len(self.LABELS)
        macro_recall = sum(
            self._safe_div(per_label[label]["tp"], per_label[label]["tp"] + per_label[label]["fn"])
            for label in self.LABELS
        ) / len(self.LABELS)
        macro_f1 = sum(per_label[label]["f1"] for label in self.LABELS) / len(self.LABELS)
        metric_dict = {
            "task_type": "cold_binary",
            "accuracy": round(self._safe_div(correct, total), 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "f1_macro": round(macro_f1, 4),
            "hate_precision": per_label["hate"]["precision"],
            "hate_recall": per_label["hate"]["recall"],
            "hate_f1": per_label["hate"]["f1"],
            "non_hate_precision": per_label["non-hate"]["precision"],
            "non_hate_recall": per_label["non-hate"]["recall"],
            "non_hate_f1": per_label["non-hate"]["f1"],
            "success": success,
            "total": total,
            "success_rate": round(self._safe_div(success, total), 4),
            "invalid": invalid,
            "per_label": per_label,
            "confusion_matrix": confusion,
        }

        if save_data:
            self._save_result(info_data or {}, metric_dict)
        return metric_dict

    def _save_result(self, info_data: dict, score_dict: dict):
        os.makedirs(self.output_dir, exist_ok=True)
        model_name = info_data.get("model", "model")
        shot_num = info_data.get("shot_num", 0)
        seed = info_data.get("seed", 0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"metric_{model_name}_{shot_num}_{seed}_{timestamp}.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump({"info": info_data, "metrics": score_dict}, file, ensure_ascii=False, indent=2)


class StepOneMetrics:

    def __init__(self, output_dir: str = "./metrics/llm/"):
        self.output_dir = output_dir

        self.init_metric()

    def init_metric(self):
        self.success = 0
        self.total = 0
        
        # 新增的字段级指标统计
        self.field_metrics = {
            'target_sim': 0.0,      # Target文本平均相似度
            'argument_sim': 0.0,    # Argument文本平均相似度
            'match': 0,
            'soft_match': 0,
        }

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
            pred_data_dict[content_id] = data["pred_tuples"]

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

    def calculate_field_metrics(self, pred_data_dict: dict, gt_data_dict: dict, similarity_threshold: float = 0.5):
        """计算字段级别的评估指标"""

        total_target_sim = 0.0
        total_arg_sim = 0.0
        hard_match = 0
        soft_match = 0
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
                
                if target_sim == 1 and arg_sim == 1:
                    hard_match += 1
                if target_sim > similarity_threshold and arg_sim > similarity_threshold:
                    soft_match += 1

                valid_pairs += 1
        
        # 计算总体指标
        self.valid_pairs = valid_pairs if valid_pairs > 0 else 1
        self.field_metrics['target_sim'] = total_target_sim / self.valid_pairs
        self.field_metrics['argument_sim'] = total_arg_sim / self.valid_pairs
        self.field_metrics['match'] = hard_match
        self.field_metrics['soft_match'] = soft_match
        
        return self.field_metrics

    def run(
            self, 
            datas_list: Optional[list[dict]] = None, 
            data_path: Optional[str] = None,
            info_data: Optional[dict] = None,
            save_data: bool = False,
            similarity_threshold: float = 0.5
            ) -> dict:
        
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
            field_metrics = self.calculate_field_metrics(pred_data_dict, gt_data_dict, similarity_threshold)
            pbar.update(1)

        except Exception as e:
            logger.exception(e)
            logger.error(f"Runtime Error: {str(e)}", exc_info=True)
            exit()

        metric_dict = {
            # 基础指标
            "success": self.success,
            "total": self.total,
            "success_rate": round(self.success / self.total, 4),
            "match": field_metrics['match'],
            "soft_match": field_metrics['soft_match'],
            "match_rate": round(field_metrics['match'] / self.valid_pairs, 4),
            "soft_match_rate": round(field_metrics['soft_match'] / self.valid_pairs, 4),
            "field_metrics": {
                "text_similarity": {
                    "target_avg_sim": round(field_metrics['target_sim'], 4),
                    "argument_avg_sim": round(field_metrics['argument_sim'], 4)
                },
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
        
    
    

