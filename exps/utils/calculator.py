import os
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

from exps.utils.data import *

class F1Calculator:
    """F1分数计算工具类"""
    
    @staticmethod
    def calculate_f1(precision, recall):
        """计算F1分数（F1是精确率和召回率的调和平均数）[6](@ref)"""
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_precision(tp, fp):
        """计算精确率：TP / (TP + FP) [7](@ref)"""
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)
    
    @staticmethod
    def calculate_recall(tp, fn):
        """计算召回率：TP / (TP + FN) [7](@ref)"""
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_from_confusion_matrix(tp, fp, fn):
        """从混淆矩阵的基本元素计算F1分数[6](@ref)"""
        precision = F1Calculator.calculate_precision(tp, fp)
        recall = F1Calculator.calculate_recall(tp, fn)
        return F1Calculator.calculate_f1(precision, recall)
    
    def get_f1(self, predictions: list[dict], true_labels: list[int], average='macro', binary_label=False):

        pred_labels = [pred['predicted_label'] for pred in predictions]

        print("Sample predictions vs true labels:")
        for i in range(10):
            print(f"Text: {predictions[i]['text'][:50]}... | Predicted: {predictions[i]['predicted_label']} ({predictions[i]['label_description']}) | True: {true_labels[i]}")
        
        # 计算各项指标
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average=average)
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true_labels, pred_labels, average=None)
        
        # 生成详细分类报告
        class_report = classification_report(true_labels, pred_labels, 
                                           target_names=[(label_dict_inv if not binary_label else binary_label_dict_inv).get(i, f'Class_{i}') for i in sorted(set(true_labels))], digits=4, output_dict=True)
        
        # 计算混淆矩阵相关指标
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # 计算每个类别的精确率和召回率
        precision_per_class = []
        recall_per_class = []
        
        for i in range(len(cm)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'f1_per_class': dict(zip(sorted(set(true_labels)), f1_per_class)),
            'precision_per_class': dict(zip(sorted(set(true_labels)), precision_per_class)),
            'recall_per_class': dict(zip(sorted(set(true_labels)), recall_per_class)),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def save_results_to_json(self, evaluation_results, output_path="evaluation_results.json"):
        """
        将评估结果保存到JSON文件
        
        Args:
            evaluation_results: evaluate_with_f1函数返回的结果
            output_path: 输出JSON文件路径
        """
        # 创建结果字典
        results_dict = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(evaluation_results['true_labels'])
            },
            "metrics": {
                "accuracy": evaluation_results['accuracy'],
                "f1_score": evaluation_results['f1_score'],
                "f1_per_class": evaluation_results['f1_per_class'],
                "precision_per_class": evaluation_results['precision_per_class'],
                "recall_per_class": evaluation_results['recall_per_class']
            },
            "detailed_classification_report": evaluation_results['classification_report'],
            "confusion_matrix": evaluation_results['confusion_matrix'],
            "predictions": []
        }
        
        # 添加详细的预测结果
        for i, (pred, true_label) in enumerate(zip(evaluation_results['predictions'], evaluation_results['true_labels'])):
            results_dict["predictions"].append({
                "sample_id": i,
                "text": pred["text"],
                "predicted_label": pred["predicted_label"],
                "predicted_label_description": pred["label_description"],
                "true_label": int(true_label),
                "true_label_description": label_dict_inv.get(int(true_label), f"Class_{true_label}"),
                "confidence": pred["confidence"],
                # "all_probabilities": pred["all_probabilities"],
                "correct": pred["predicted_label"] == true_label
            })
        
        # 计算并添加总体统计信息
        correct_predictions = sum(1 for pred in results_dict["predictions"] if pred["correct"])
        results_dict["summary"] = {
            "total_correct": correct_predictions,
            "total_incorrect": len(results_dict["predictions"]) - correct_predictions,
            "overall_accuracy": correct_predictions / len(results_dict["predictions"]),
            "per_class_stats": self._calculate_per_class_stats(results_dict["predictions"])
        }
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 保存到JSON文件[6,7](@ref)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=4, ensure_ascii=False)
            print(f"结果已成功保存到: {output_path}")
        except Exception as e:
            print(f"保存JSON文件时出错: {str(e)}")
        
        return results_dict
    
    def _calculate_per_class_stats(self, predictions):
        """计算每个类别的统计信息"""
        class_stats = {}
        
        for pred in predictions:
            true_label = pred["true_label"]
            pred_label = pred["predicted_label"]
            
            if true_label not in class_stats:
                class_stats[true_label] = {
                    "class_name": label_dict_inv.get(true_label, f"Class_{true_label}"),
                    "total_samples": 0,
                    "correct_predictions": 0,
                    "incorrect_predictions": 0
                }
            
            class_stats[true_label]["total_samples"] += 1
            if pred["correct"]:
                class_stats[true_label]["correct_predictions"] += 1
            else:
                class_stats[true_label]["incorrect_predictions"] += 1
        
        # 计算每个类别的准确率
        for stats in class_stats.values():
            if stats["total_samples"] > 0:
                stats["accuracy"] = stats["correct_predictions"] / stats["total_samples"]
            else:
                stats["accuracy"] = 0.0
        
        return class_stats