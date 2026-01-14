import json
from sklearn.model_selection import train_test_split

# ===========================
# 读取原始数据
# ===========================
input_path = "train_sampled_5000.json"
data = json.load(open(input_path, "r", encoding="utf-8"))

print(f"加载到文章数量: {len(data)}")

# ===========================
# 划分训练集 / 测试集
# 80% 训练，20% 测试
# ===========================
train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42
)

print(f"训练集文章数量: {len(train_data)}")
print(f"测试集文章数量: {len(test_data)}")

# ===========================
# 保存结果
# ===========================
with open("train_split.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("test_split.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("数据集划分完成！")
print("已生成 train_split.json 和 test_split.json")
