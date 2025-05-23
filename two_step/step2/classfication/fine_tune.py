import pandas as pd
import os
import json
import torch
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 自定义数据集类
class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) 
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

# 评估指标计算函数
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

# 统一训练函数
def train_classifier(data_file, model_dir, label_field, num_labels, seed: int = 23333333, n_splits: int = 5):
    # 加载数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 提取文本和标签字符串
    if label_field == "targeted_group":
        texts = [entry['content'] for entry in data if entry['quadruples'][0][label_field].split(", ")[0]  != "non-hate"]
    else:
        texts = [entry['content'] for entry in data]

    # 提取标签字符串并检查数据完整性
    labels_str = []
    for entry in data:
        if not entry['quadruples']:
            raise ValueError(f"Entry {entry['id']} has no quadruples.")
        if label_field == "targeted_group":
            label_str = entry['quadruples'][0][label_field].split(", ")[0]
            if label_str == "non-hate":
                continue
            labels_str.append(label_str)
        else:
            labels_str.append(entry['quadruples'][0][label_field])

    
    # 创建标签映射
    unique_labels = sorted(list(set(labels_str)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 检查标签数量一致性
    assert len(unique_labels) == num_labels, f"Expected {num_labels} labels, but found {len(unique_labels)}"
    
    # 转换标签为数值ID
    labels = [label2id[label] for label in labels_str]
    
    # 初始化分词器（使用中文BERT）
    # IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese
    # IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese
    # IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese
    tokenizer = AutoTokenizer.from_pretrained("./models/Erlangshen-Roberta-330M-NLI")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        logger.info((f"===== Fold {fold+1}/{n_splits} ====="))

        # 划分训练集和验证集
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # 数据编码（动态分词，避免内存爆炸）
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=1024)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=1024)
        
        # 创建数据集
        train_dataset = TextClassificationDataset(train_encodings, train_labels)
        val_dataset = TextClassificationDataset(val_encodings, val_labels)
        
        # 加载模型
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained("./models/Erlangshen-Roberta-330M-NLI", num_labels=num_labels, ignore_mismatched_sizes=True).to(device)
        
        new_model_dir = os.path.join(model_dir, str(fold))
        os.makedirs(new_model_dir, exist_ok=True)
        # 训练参数设置
        training_args = TrainingArguments(
            output_dir=new_model_dir,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            logging_strategy="steps",
            logging_steps=100,
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=16,
            num_train_epochs=30,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model="f1",
            disable_tqdm=False, 
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # 开始训练
        trainer.train()

        eval_result = trainer.evaluate()
        logger.debug(eval_result)
        accuracies.append(eval_result["eval_accuracy"])

        
        # 保存最佳模型
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
    
    logger.info((f"平均准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}"))

if __name__ == "__main__":
    # 训练第一个分类任务（二分类）
    train_classifier(
        data_file="./data/temp_train_data.json",
        model_dir="./two_step/step2/classfication/is_hate_model",
        label_field="hateful",
        num_labels=2
    )
    
    # train_classifier(
    #     train_file="/workspace/data/temp_train_data.json",
    #     val_file="/workspace/data/temp_test_data.json",
    #     model_dir="/workspace/two_step/step2/classfication/hate_type_model",
    #     label_field="targeted_group",
    #     num_labels=5
    # )
