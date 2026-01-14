import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# ======================================================
# 1. 读取数据（按作业JSON格式处理）
# ======================================================
def load_data(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences = []
    labels = []
    for item in data:
        for s, lab in item["sent_and_label"]:
            sentences.append(s)
            labels.append(1 if lab == "machine" else 0)  # machine=1，human=0
    return sentences, np.array(labels)

print("Loading data...")

sentences, labels = load_data("train_split.json")

# ======================================================
# 2. 划分训练/验证集（分层抽样，保证标签分布一致）
# ======================================================
X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=labels
)

# ======================================================
# 3. SBERT语义嵌入（生成384维句子向量，捕捉语义特征）
# ======================================================
print("Loading SBERT model...")
model_path = "./all-MiniLM-L6-v2"
sbert = SentenceTransformer(model_path)
print("Encoding embeddings...")
X_train_emb = sbert.encode(X_train_txt, batch_size=64, show_progress_bar=True)
X_val_emb = sbert.encode(X_val_txt, batch_size=64, show_progress_bar=True)
input_dim = X_train_emb.shape[1]  # 固定为384（SBERT模型输出维度）

# ======================================================
# 4. MLP非线性特征增强（提升特征区分度）
# ======================================================
class FeatureEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),  # 降维：384→256
            nn.ReLU(),            # 非线性激活：增强表达能力
            nn.Linear(256, dim)   # 升维：256→384（与原嵌入维度一致）
        )
    def forward(self, x):
        return self.net(x)

# 初始化并训练MLP
enhancer = FeatureEnhancer(input_dim)
optimizer = torch.optim.Adam(enhancer.parameters(), lr=1e-3)
X_train_tensor = torch.tensor(X_train_emb, dtype=torch.float32)

print("Training MLP feature enhancer...")
for epoch in range(40):  # 训练5轮（足够拟合且避免过拟合）
    optimizer.zero_grad()  # 清空梯度
    out = enhancer(X_train_tensor)
    loss = ((out - X_train_tensor) ** 2).mean()  # MSE损失：保留核心语义+增强特征
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    print(f"Epoch {epoch} loss: {loss.item():.4f}")

# 生成增强后的特征（关闭梯度计算，避免影响后续流程）
X_train_enh = enhancer(X_train_tensor).detach().numpy()
X_val_enh = enhancer(torch.tensor(X_val_emb, dtype=torch.float32)).detach().numpy()


# ======================================================
# 5. 特征标准化（消除量纲影响，提升XGBoost稳定性）
# ======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enh)  # 训练集：拟合+转换
X_val_scaled = scaler.transform(X_val_enh)          # 验证集：仅用训练集参数转换

# ======================================================
# 6. 计算类别权重（解决样本不平衡问题）
# ======================================================
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
scale_pos = class_weights[1]  # AI生成句子（正类）的权重（平衡正负样本）

# ======================================================
# 7. 训练XGBoost分类器（核心预测模型）
# ======================================================
print("Training XGBoost...")
clf = xgb.XGBClassifier(
    objective="binary:logistic",        # 关键：二分类任务，输出概率
    use_label_encoder=False,            # 关键：确保sklearn识别为分类器（而非回归器）
    n_estimators=909,                   # 决策树数量：500（平衡性能与速度）
    max_depth=8,                        # 树深度：控制过拟合
    learning_rate=0.09,                 # 学习率：小步迭代，提升泛化能力
    subsample=0.72,                      # 样本抽样率：90%（避免过拟合）
    colsample_bytree=0.81,               # 特征抽样率：90%（避免过拟合）
    gamma=0.12,                          # 节点分裂阈值：减少冗余分裂
    reg_lambda=3.14,                       # L2正则：抑制过拟合
    reg_alpha=1.26,                      # L1正则：稀疏特征，提升泛化
    scale_pos_weight=scale_pos,         # 类别权重：平衡正负样本
    eval_metric="logloss",              # 评估指标：对数损失（适合二分类）
    n_jobs=-1,                           # 并行计算：使用所有CPU核心（加速训练）
    tree_method='hist',
    device='cuda:0',
    random_state=42  
)
clf.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],  # 验证集：监控训练过程
    verbose=True                       # 打印每轮日志
)
# ======================================================
# 保存模型
# ======================================================
joblib.dump(clf, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
torch.save(enhancer.state_dict(), "enhancer.pt") 
print("Models saved (xgb_model.pkl, scaler.pkl, enhancer.pt)")
print("Models saved (xgb_model.pkl, scaler.pkl, enhancer.pt)")

# ======================================================
# 9. 验证集评估
# ======================================================
print("\nEvaluating validation set...")
# 用XGBoost直接预测概率和标签
y_prob = clf.predict_proba(X_val_scaled)[:, 1]  # 预测为"machine"的概率（取第二列）
y_pred = (y_prob > 0.5).astype(int)             # 概率阈值0.5：>0.5→1（machine），否则0（human）

# 计算作业要求的指标（🔶1-21、🔶1-24）
acc = accuracy_score(y_val, y_pred)            # 准确率：整体预测正确的比例
prec = precision_score(y_val, y_pred)          # 精确率：预测为AI的句子中实际是AI的比例
rec = recall_score(y_val, y_pred)              # 召回率：实际是AI的句子中被正确预测的比例
micro_f1 = f1_score(y_val, y_pred, average="micro")  # Micro-F1：综合精确率和召回率（样本平衡场景）
macro_f1 = f1_score(y_val, y_pred, average="macro")  # Macro-F1：每类F1的平均值（关注类别公平性）

# 打印指标（保留4位小数）
print("\n===== Validation Metrics (4 decimal places) =====")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")

# ======================================================
# 10. 可视化分析
# ======================================================
# ROC曲线：评估模型区分正负类的能力
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="#1f77b4", label=f"AUC-ROC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="#ff7f0e", linestyle="--")  # 随机猜测线
plt.title("ROC Curve (AI-Generated Sentence Detection)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.show()

# PR曲线：更适合样本不平衡场景（如AI句子数量少）
prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_prob)
pr_auc = auc(rec_curve, prec_curve)
plt.figure(figsize=(6, 4))
plt.plot(rec_curve, prec_curve, color="#2ca02c", label=f"AUC-PR = {pr_auc:.4f}")
plt.title("Precision-Recall Curve (AI-Generated Sentence Detection)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR:  {pr_auc:.4f}")
print("\nAll steps completed!")