import os
import json
import torch
import random
from datetime import datetime
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

random.seed(23333333)
torch.manual_seed(23333333)

label_dict = {
    "Racism": 0,
    "Region": 1,
    "LGBTQ": 2,
    "Sexism": 3,
    "others": 4,
    "non-hate": 5
}

label_dict_inv = {v: k for k, v in label_dict.items()}

def get_data(data_path):
    texts = []
    datas = json.load(open(data_path, "r", encoding="utf-8"))
    labels = []
    for data in datas:
        text = data['content']
        texts.append(text)
        quadruples: list[dict[str, str]] = data['quadruples']
        all_target_groups = set()
        for quadruple in quadruples:
            target, argument, target_groups, is_hate = quadruple.values()
            target_groups = [i.strip() for i in target_groups.split(',')]
            all_target_groups.update(target_groups)
        
        if 'non-hate' not in all_target_groups:
            text_label = all_target_groups.pop()
        else:
            text_label = 'non-hate'
            for tg in all_target_groups:
                if tg != 'non-hate':
                    text_label = tg
                    break
        
        labels.append(label_dict[text_label])
    return texts, labels


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

# F1计算示例
if __name__ == "__main__":
    # F1计算工具使用示例
    calculator = F1Calculator()
    
    # 示例数据
    true_positive = 80
    false_positive = 20
    false_negative = 10
    
    f1_value = calculator.f1_from_confusion_matrix(true_positive, false_positive, false_negative)
    print(f"\nF1计算示例:")
    print(f"TP={true_positive}, FP={false_positive}, FN={false_negative}")
    print(f"F1分数: {f1_value:.4f}")


class BertInference:
    def __init__(self, model_path, max_length=512):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, text):
        """对单条文本进行预测"""
        # 文本预处理
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            "text": text,
            "predicted_label": predicted_class,
            "label_description": label_dict_inv.get(predicted_class, "未知"),
            "confidence": confidence,
            # "all_probabilities": probabilities.cpu().numpy()[0]
        }
    
    def predict_batch(self, texts, batch_size=32):
        """批量预测"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 批量编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
                confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
            
            # 处理批量结果
            for j, text in enumerate(batch_texts):
                results.append({
                    "text": text,
                    "predicted_label": int(predicted_classes[j]),
                    "label_description": label_dict_inv.get(int(predicted_classes[j]), "未知"),
                    "confidence": float(confidences[j]),
                    # "all_probabilities": probabilities[j].cpu().numpy()
                })
        
        return results
    
    def evaluate_with_f1(self, test_texts, true_labels, average='weighted'):
        """
        评估模型并计算F1分数
        Args:
            test_texts: 测试文本列表
            true_labels: 真实标签列表
            average: F1计算方式 ('micro', 'macro', 'weighted', 'binary')
        Returns:
            dict: 包含各项评估指标的结果
        """
        # 批量预测
        predictions = self.predict_batch(test_texts)
        pred_labels = [pred['predicted_label'] for pred in predictions]
        
        # 计算各项指标
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average=average)
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(true_labels, pred_labels, average=None)
        
        # 生成详细分类报告
        class_report = classification_report(true_labels, pred_labels, 
                                           target_names=[label_dict_inv.get(i, f'Class_{i}') for i in sorted(set(true_labels))], digits=4, output_dict=True)
        
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
                "model_path": str(self.model.name_or_path) if hasattr(self.model, 'name_or_path') else "fine_tuned_bert",
                "device": str(self.device),
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

# 使用示例
if __name__ == "__main__":
    # 初始化推理器
    inference = BertInference('./models/bert-finetuned/checkpoint-2409')
    
    # 单条文本推理
    text = "这个产品质量非常好，强烈推荐！"
    result = inference.predict_single(text)
    print("单条推理结果:")
    print(f"文本: {result['text']}")
    print(f"预测标签: {result['predicted_label']} ({result['label_description']})")
    print(f"置信度: {result['confidence']:.4f}")
    # print(f"所有类别概率: {result['all_probabilities']}")
    print("-" * 50)
    
    # 批量推理
    texts, labels = get_data("data/full/std/test.json")
    
    # batch_results = inference.predict_batch(texts[:10])
    # print("批量推理结果:")
    # for i, res in enumerate(batch_results):
    #     print(f"{i+1}. {res['text']} -> {res['label_description']} (置信度: {res['confidence']:.4f})")

    # 进行评估并计算F1值
    evaluation_results = inference.evaluate_with_f1(texts, labels)
    saved_results = inference.save_results_to_json(evaluation_results, output_path="exps/bert/results.json")
    
    print("=== 模型评估结果 ===")
    print(f"准确率 (Accuracy): {evaluation_results['accuracy']:.4f}")
    print(f"加权F1分数 (Weighted F1): {evaluation_results['f1_score']:.4f}")
    print()
    
    print("=== 各类别详细指标 ===")
    for class_id in sorted(evaluation_results['f1_per_class'].keys()):
        class_name = label_dict_inv.get(class_id, f'Class_{class_id}')
        print(f"{class_name} (类别{class_id}):")
        print(f"  - F1分数: {evaluation_results['f1_per_class'][class_id]:.4f}")
        print(f"  - 精确率: {evaluation_results['precision_per_class'][class_id]:.4f}")
        print(f"  - 召回率: {evaluation_results['recall_per_class'][class_id]:.4f}")
    print()
    
    print("=== 详细分类报告 ===")
    print(evaluation_results['classification_report'])
    
    print("=== 混淆矩阵 ===")
    print(evaluation_results['confusion_matrix'])
    
    # 单条文本推理示例
    print("\n" + "="*50)
    print("单条文本推理示例:")
    
    test_text = "这个电影太精彩了！"
    result = inference.predict_single(test_text)
    print(f"文本: {result['text']}")
    print(f"预测标签: {result['predicted_label']} ({result['label_description']})")
    print(f"置信度: {result['confidence']:.4f}")
    # print(f"概率分布: {result['all_probabilities']}")