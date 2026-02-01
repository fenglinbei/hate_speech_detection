import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np

label_dict = {
    "Racism": 0,
    "Region": 1,
    "LGBTQ": 2,
    "Sexism": 3,
    "others": 4,
    "non-hate": 5
}

label_dict_inv = {v: k for k, v in label_dict.items()}

# 1. 数据准备
train_data_path = "data/full/std/train.json"
train_texts = []
train_datas = json.load(open(train_data_path, "r", encoding="utf-8"))
train_labels = []
for train_data in train_datas:
    text = train_data['content']
    train_texts.append(text)
    quadruples: list[dict[str, str]] = train_data['quadruples']
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
        
    train_labels.append(label_dict[text_label])

# print(json.dumps([(i, label_dict_inv[j]) for i, j in zip(train_texts, train_labels)], ensure_ascii=False, indent=2))

tokenizer = BertTokenizer.from_pretrained("./models/bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
train_dataset = train_dataset.map(tokenize_function, batched=True)

# 2. 微调训练
model = BertForSequenceClassification.from_pretrained("./models/bert-base-chinese", num_labels=6)

training_args = TrainingArguments(
    output_dir="./models/bert-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# # 3. 测试评估
# test_texts = ["test text1", ...]
# test_labels = [0, ...]
# test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
# test_dataset = test_dataset.map(tokenize_function, batched=True)

# results = trainer.evaluate(test_dataset)
# print(results)