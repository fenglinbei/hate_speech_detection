import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
def train_classifier(train_file, val_file, model_dir, label_field, num_labels):
    # 加载数据
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # 提取文本和标签字符串
    train_texts = [entry['content'] for entry in train_data if entry['quadruples'][0][label_field].split(", ")[0]  != "non-hate"]
    val_texts = [entry['content'] for entry in val_data if entry['quadruples'][0][label_field].split(", ")[0]  != "non-hate"]

    # 提取标签字符串并检查数据完整性
    train_labels_str = []
    for entry in train_data:
        if not entry['quadruples']:
            raise ValueError(f"Entry {entry['id']} has no quadruples.")
        if label_field == "targeted_group":
            label_str = entry['quadruples'][0][label_field].split(", ")[0]
            if label_str == "non-hate":
                continue
            train_labels_str.append(label_str)
        else:
            train_labels_str.append(entry['quadruples'][0][label_field])
    
    val_labels_str = []
    for entry in val_data:
        if not entry['quadruples']:
            raise ValueError(f"Entry {entry['id']} has no quadruples.")
        if label_field == "targeted_group":
            label_str = entry['quadruples'][0][label_field].split(", ")[0]
            if label_str == "non-hate":
                continue
            val_labels_str.append(label_str)
        else:
            val_labels_str.append(entry['quadruples'][0][label_field])
    
    # 创建标签映射
    all_labels_str = train_labels_str + val_labels_str
    unique_labels = sorted(list(set(all_labels_str)))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    # 检查标签数量一致性
    assert len(unique_labels) == num_labels, f"Expected {num_labels} labels, but found {len(unique_labels)}"
    
    # 转换标签为数值ID
    train_labels = [label2id[label] for label in train_labels_str]
    val_labels = [label2id[label] for label in val_labels_str]
    
    # 初始化分词器（使用中文BERT）
    # IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese
    # IDEA-CCNL/Erlangshen-TCBert-1.3B-Classification-Chinese
    # IDEA-CCNL/Erlangshen-TCBert-330M-Classification-Chinese
    tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-330M-NLI")
    
    # 文本编码
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=128
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=128
    )
    
    # 创建数据集
    train_dataset = TextClassificationDataset(train_encodings, train_labels)
    val_dataset = TextClassificationDataset(val_encodings, val_labels)
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained("IDEA-CCNL/Erlangshen-Roberta-330M-NLI", num_labels=num_labels, ignore_mismatched_sizes=True)
    
    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        num_train_epochs=50,
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
    
    # 保存最佳模型
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    # 训练第一个分类任务（二分类）
    # train_classifier(
    #     train_file="/workspace/data/temp_train_data.json",
    #     val_file="/workspace/data/temp_test_data.json",
    #     model_dir="/workspace/two_step/step2/classfication/is_hate_model",
    #     label_field="hateful",
    #     num_labels=2
    # )
    
    train_classifier(
        train_file="/workspace/data/temp_train_data.json",
        val_file="/workspace/data/temp_test_data.json",
        model_dir="/workspace/two_step/step2/classfication/hate_type_model",
        label_field="targeted_group",
        num_labels=5
    )