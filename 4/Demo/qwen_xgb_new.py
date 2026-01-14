import json
import torch
import numpy as np
import joblib
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve,
    precision_score, recall_score
)

from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import matplotlib.pyplot as plt


# =========================
# 0. GPU CHECK
# =========================

assert torch.cuda.is_available(), "CUDA not available!"
device = torch.device("cuda")
print("Using GPU:", torch.cuda.get_device_name(0))


# =========================
# 1. Load dataset
# =========================

def load_dataset(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences, labels = [], []

    for item in data:
        for s, lab in item["sent_and_label"]:
            sentences.append(s)
            labels.append(1 if lab == "machine" else 0)

    return sentences, np.array(labels)


sentences, labels = load_dataset("Demo/data/train_cleaned.json")


# =========================
# 2. Train / Val split (一级拆分：训练集和验证集)
# =========================

X_train_text, X_val_text, y_train, y_val = train_test_split(
    sentences,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# =========================
# 3. Load Qwen3-4B (GPU)
# =========================

MODEL_PATH = "/home/data/liaozijie/hate_speech_detection/models/base/Qwen3-4B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("Loading Qwen3-4B model to GPU...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.to(device)
model.eval()


# =========================
# 4. Embedding extraction
# =========================

@torch.no_grad()
def encode(sentences, batch_size=32, max_len=256):
    all_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]   # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


print("Encoding train set...")
X_train_emb = encode(X_train_text)

print("Encoding validation set...")
X_val_emb = encode(X_val_text)


# =========================
# 4.5 二级拆分：从训练集中拆分模型训练集和校准集
# =========================
X_train_model_emb, X_train_calib_emb, y_train_model, y_train_calib = train_test_split(
    X_train_emb,
    y_train,
    test_size=0.2,  # 20%训练集数据用于校准
    random_state=42,
    stratify=y_train
)


# =========================
# 5. Scaling（仅用模型训练集拟合scaler）
# =========================

scaler = StandardScaler()
X_train_model_scaled = scaler.fit_transform(X_train_model_emb)  # 模型训练集拟合
X_train_calib_scaled = scaler.transform(X_train_calib_emb)      # 校准集转换
X_val_scaled = scaler.transform(X_val_emb)                      # 验证集转换

joblib.dump(scaler, "qwen_scaler1.pkl")


# =========================
# 6. XGBoost training（用模型训练集训练）
# =========================

print("Training XGBoost...")
clf = xgb.XGBClassifier(
    objective="binary:logistic",
    n_estimators=700,           # 增加树的数量
    max_depth=8,                # 稍深一些
    learning_rate=0.07,         # 稍低一些
    subsample=0.75,
    colsample_bytree=0.85,
    gamma=0.1,                  # 正则化
    reg_lambda=2.5,
    reg_alpha=1.0,
    min_child_weight=3,         # 防止过拟合
    eval_metric=["logloss", "error"],
    n_jobs=-1,
    tree_method='hist',
    device='cpu',
    random_state=42             # 固定随机种子
)
clf.fit(
    X_train_model_scaled,  # 改用模型训练集
    y_train_model,
    eval_set=[(X_val_scaled, y_val)],  # 验证集仅用于监控
    verbose=True
)
joblib.dump(clf, "qwen_xgb1.pkl")


# =========================
# 7. Calibration（用校准集拟合，避免接触验证集）
# =========================
print("calibration")
if hasattr(clf, "_Booster"):
    clf._Booster.set_attr(early_stop=None)

calibrated = CalibratedClassifierCV(
    estimator=clf,  
    method="sigmoid"
)

calibrated.fit(X_train_calib_scaled, y_train_calib)  # 关键修复：使用校准集
joblib.dump(calibrated, "qwen_xgb_calibrated1.pkl")


# =========================
# 8. Evaluation（验证集保持"未见过"状态，评估结果更可靠）
# =========================

y_pred = calibrated.predict(X_val_scaled)
y_prob = calibrated.predict_proba(X_val_scaled)[:, 1]

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
micro_f1 = f1_score(y_val, y_pred, average="micro")
macro_f1 = f1_score(y_val, y_pred, average="macro")

print("\n===== Validation Metrics (4 decimal places) =====")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")


# =========================
# 9. ROC / PR
# =========================

fpr, tpr, _ = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="#1f77b4", label=f"AUC-ROC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="#ff7f0e", linestyle="--")
plt.title("ROC Curve (AI-Generated Sentence Detection)")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.savefig("qwen_validation_roc.png") 
plt.show()

prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_prob)
pr_auc = auc(rec_curve, prec_curve)
plt.figure(figsize=(6, 4))
plt.plot(rec_curve, prec_curve, color="#2ca02c", label=f"AUC-PR = {pr_auc:.4f}")
plt.title("Precision-Recall Curve (AI-Generated Sentence Detection)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("qwen_validation_pr.png") 
plt.show()

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR:  {pr_auc:.4f}")
print("\nAll steps completed!")
