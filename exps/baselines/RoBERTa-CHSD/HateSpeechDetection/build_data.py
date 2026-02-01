import json
import csv
import random
from collections import Counter

label_dict = {
    "Racism": 0,
    "Region": 1,
    "LGBTQ": 2,
    "Sexism": 3,
    "others": 4,
    "non-hate": 5
}

with open("data/full/std/train.json", "r", encoding="utf-8") as f:
    data1 = json.load(f)

# 准备转换后的数据
output_data = []
for item in data1:
    content = item["content"]
    quadruples = item["quadruples"]
    
    # 提取所有hateful值
    hateful_values = []
    for q in quadruples:
        targeted_group = q.get("targeted_group", "").strip()
        hateful_values.extend([tg.strip() for tg in targeted_group.split(',') if tg.strip()])
    
    # 根据规则计算标签
    if "non-hate" in hateful_values:
        label = 5
    else:
        # 选择出现频率最高的仇恨类别（频率相同时随机选择）
        freq_counter = Counter(hateful_values)
        max_freq = max(freq_counter.values())
        most_common_classes = [cls for cls, freq in freq_counter.items() if freq == max_freq]
        
        if len(most_common_classes) > 1:
            chosen_class = random.choice(most_common_classes)  # 频率相同时随机选择
        else:
            chosen_class = most_common_classes[0]
        
        label = label_dict[chosen_class]
    
    output_data.append({"label": label, "TEXT": content})

# 输出CSV格式结果
with open('./exps/RoBERTa-CHSD/HateSpeechDetection/new_data/train.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["label", "TEXT"])
    writer.writeheader()
    for row in output_data[:int(0.9*len(output_data))]:
        writer.writerow(row)

with open('./exps/RoBERTa-CHSD/HateSpeechDetection/new_data/dev.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["label", "TEXT"])
    writer.writeheader()
    for row in output_data[int(0.9*len(output_data)):]:
        writer.writerow(row)