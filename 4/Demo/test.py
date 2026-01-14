import json
import torch
import numpy as np
import joblib
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt


# =========================
# 0. 配置与设备检查
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 模型和数据路径（与训练时一致）
MODEL_PATH = "/home/data/liaozijie/hate_speech_detection/models/base/Qwen3-4B"  # 同训练时的模型路径
VALIDATION_DATA_PATH = "Demo/data/phase1_test_with_labels.json"  # 验证集数据路径（可替换为新的验证集）
SCALER_PATH = "qwen_scaler1.pkl"          # 训练好的scaler
CALIBRATED_MODEL_PATH = "qwen_xgb_calibrated1.pkl"  # 校准后的模型


# =========================
# 1. 加载验证集数据
# =========================
def load_dataset(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences, labels = [], []
    for item in data:
        for s, lab in item["sent_and_label"]:
            sentences.append(s)
            labels.append(1 if lab == "machine" else 0)
    return sentences, np.array(labels)

# 加载验证集（如果需要单独划分验证集，可复用train_test_split）
sentences, labels = load_dataset(VALIDATION_DATA_PATH)
# 如果原数据已划分好验证集，可直接加载X_val_text和y_val（或从文件读取）


# =========================
# 2. 加载模型和分词器（与训练时一致）
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading Qwen3-4B model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.to(device)
model.eval()


# =========================
# 3. 提取验证集嵌入（复用训练时的encode函数）
# =========================
@torch.no_grad()
def encode(sentences, batch_size=4, max_len=256):
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
        outputs = model(** inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

print("Encoding validation set...")
X_val_emb = encode(sentences)  # 验证集文本的嵌入


# =========================
# 4. 用训练好的scaler缩放特征
# =========================
print("Scaling features with trained scaler...")
scaler = joblib.load(SCALER_PATH)  # 加载训练时保存的scaler
X_val_scaled = scaler.transform(X_val_emb)  # 仅transform，不重新fit


# =========================
# 5. 加载模型并预测
# =========================
print("Loading calibrated model...")
calibrated = joblib.load(CALIBRATED_MODEL_PATH)  # 加载校准后的模型

# 预测
y_pred = calibrated.predict(X_val_scaled)  # 预测标签（0/1）
y_prob = calibrated.predict_proba(X_val_scaled)[:, 1]  # 预测为1的概率


# =========================
# 6. 计算并打印验证指标
# =========================
acc = accuracy_score(labels, y_pred)
prec = precision_score(labels, y_pred)
rec = recall_score(labels, y_pred)
micro_f1 = f1_score(labels, y_pred, average="micro")
macro_f1 = f1_score(labels, y_pred, average="macro")

print("\n===== Validation Metrics =====")
print(f"Accuracy:    {acc:.4f}")
print(f"Precision:   {prec:.4f}")
print(f"Recall:      {rec:.4f}")
print(f"Micro-F1:    {micro_f1:.4f}")
print(f"Macro-F1:    {macro_f1:.4f}")


# =========================
# 7. 绘制ROC和PR曲线
# =========================
# ROC曲线
fpr, tpr, _ = roc_curve(labels, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="#1f77b4", label=f"AUC-ROC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="#ff7f0e", linestyle="--")
plt.title("ROC Curve (Validation Set)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("qwen_validation_roc.png")  # 保存图片
plt.show()

# PR曲线
prec_curve, rec_curve, _ = precision_recall_curve(labels, y_prob)
pr_auc = auc(rec_curve, prec_curve)
plt.figure(figsize=(6, 4))
plt.plot(rec_curve, prec_curve, color="#2ca02c", label=f"AUC-PR = {pr_auc:.4f}")
plt.title("Precision-Recall Curve (Validation Set)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("qwen_validation_pr.png")  # 保存图片
plt.show()

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR:  {pr_auc:.4f}")
print("Validation completed!")