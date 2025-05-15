import pandas as pd
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
def train_classifier(train_file, model_dir, num_labels=2):
    # 加载数据
    df = pd.read_csv(train_file)
    texts = df["content"].tolist()
    labels = df["label_id"].tolist()
    
    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
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
    train_classifier(
        train_file="./data/is_hate_classification_first_only_labeled.csv",
        model_dir="./is_hate_model",
        num_labels=2
    )
    
    # 训练第二个分类任务（三分类）
    # train_classifier(
    #     train_file="./data/hate_type_classification_first_only_labeled.csv",
    #     model_dir="./hate_type_model",
    #     num_labels=5
    # )