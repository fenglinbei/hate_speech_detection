import json
import numpy as np
import joblib
import torch
import os  # 新增：用于设置环境变量
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform

# ======================================================
# 1. 读取数据 
# ======================================================
def load_data(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences = []
    labels = []
    for item in data:
        if "sent_and_label" in item:
            for s, lab in item["sent_and_label"]:
                sentences.append(s)
                labels.append(1 if lab == "machine" else 0)
        elif "label" in item:
             sentences.append(item["sent"])
             labels.append(1 if item["label"] == "machine" else 0)
    return sentences, np.array(labels)

print("Loading data...")
try:
    sentences, labels = load_data("train_split.json")
except FileNotFoundError:
    print("Error: train_split.json not found. Please run split_dataset.py first.")
    exit()

# ======================================================
# 2. 划分训练/验证集 
# ======================================================
X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    sentences, labels, test_size=0.2, random_state=42, stratify=labels
)

# ======================================================
# 3. SBERT语义嵌入 
# ======================================================
print("Loading SBERT model...")
model_path = "./all-MiniLM-L6-v2"
sbert = SentenceTransformer(model_path)

print("Encoding embeddings...")
X_train_emb = sbert.encode(X_train_txt, batch_size=64, show_progress_bar=True)
X_val_emb = sbert.encode(X_val_txt, batch_size=64, show_progress_bar=True)

print(f"Embedding shape: {X_train_emb.shape}")

# ======================================================
# 4. 特征标准化 
# ======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_emb)
X_val_scaled = scaler.transform(X_val_emb)

# ======================================================
# 5. 计算类别权重   
# ======================================================
class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
scale_pos = class_weights[1]
print(f"Class weight (scale_pos_weight): {scale_pos:.4f}")

# ======================================================
# 6. 训练 XGBoost 分类器 (超参数搜索)
# ======================================================
print("Training XGBoost...")
clf = xgb.XGBClassifier(
    objective="binary:logistic",        
    use_label_encoder=False,            
    n_estimators=900,                  
    max_depth=9,                        
    learning_rate=0.08,                 
    subsample=0.81,                      
    colsample_bytree=0.9,               
    gamma=0.15,                         
    reg_lambda=3.14,                       
    reg_alpha=1.25,                      
    scale_pos_weight=scale_pos,         
    eval_metric="logloss",              
    n_jobs=-1,
    tree_method='hist',
    device='cuda:0',
    random_state=42                             
)

# 训练模型并监控验证集（打印每轮训练结果）
clf.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_val_scaled, y_val)],  # 验证集：监控训练过程
    verbose=True                       # 打印每轮日志
)

# ======================================================
# 7. 保存模型 
# ======================================================
joblib.dump(clf, "xgb_model_best.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Models saved: xgb_model_best.pkl, scaler.pkl")

# ======================================================
# 8. 验证集评估 
# ======================================================
print("\nEvaluating validation set...")
y_prob = clf.predict_proba(X_val_scaled)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
micro_f1 = f1_score(y_val, y_pred, average="micro")
macro_f1 = f1_score(y_val, y_pred, average="macro")

print("\n===== SBERT + XGBoost (Tuned) Results =====")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")

# ======================================================
# 9. 可视化 
# ======================================================
fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve (Tuned XGBoost)")
plt.legend()
plt.savefig("roc_curve_tuned.png")

prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_prob)
pr_auc = auc(rec_curve, prec_curve)
plt.figure(figsize=(6, 4))
plt.plot(rec_curve, prec_curve, color="#2ca02c", label=f"AUC-PR = {pr_auc:.4f}")
plt.title("Precision-Recall Curve (Tuned XGBoost)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("pr_curve_tuned.png")
plt.show()

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR:  {pr_auc:.4f}")