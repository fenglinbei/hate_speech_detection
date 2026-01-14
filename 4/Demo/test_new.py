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
MODEL_PATH = "/home/data/liaozijie/hate_speech_detection/models/base/Qwen3-4B"
VALIDATION_DATA_PATH = "Demo/data/phase1_test_with_labels.json"
SCALER_PATH = "Demo/qwen_scaler1.pkl"
CALIBRATED_MODEL_PATH = "Demo/qwen_xgb_calibrated1.pkl"

# 输出文件路径（新增）
OUTPUT_PRED_JSON = "4_test.json"


# =========================
# 1. 加载验证集数据（保留pid、domain、句子序号）
# =========================
def load_dataset_with_meta(path):
    """
    输入文件：list[ {id: pid, sent_and_label: [[sent, label], ...], domain: str}, ... ]
    输出：
      - sentences: 句子列表（按文件顺序展开）
      - labels:    0/1 标签数组（machine=1, human=0）
      - metas:     与每句对齐的元信息（pid/domain/sent_id/text/gt_label）
    """
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences, labels, metas = [], [], []

    for para in data:
        pid = para.get("id")
        domain = para.get("domain")
        sent_and_label = para.get("sent_and_label", [])

        for idx_in_para, (s, lab) in enumerate(sent_and_label, start=1):
            sentences.append(s)
            labels.append(1 if lab == "machine" else 0)
            metas.append({
                "id": idx_in_para,     # 句子在段落中的序号，从1开始
                "pid": pid,            # 段落序号/原测试文件中的id
                "text": s,             # 句子原文
                "gt_label": lab,       # 原标注 human/machine
                "domain": domain       # 原领域标签
            })

    return sentences, np.array(labels, dtype=np.int64), metas


sentences, labels, metas = load_dataset_with_meta(VALIDATION_DATA_PATH)


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
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)


print("Encoding validation set...")
X_val_emb = encode(sentences)


# =========================
# 4. 用训练好的scaler缩放特征
# =========================
print("Scaling features with trained scaler...")
scaler = joblib.load(SCALER_PATH)
X_val_scaled = scaler.transform(X_val_emb)


# =========================
# 5. 加载模型并预测
# =========================
print("Loading calibrated model...")
calibrated = joblib.load(CALIBRATED_MODEL_PATH)

y_pred = calibrated.predict(X_val_scaled)                 # 0/1
y_prob = calibrated.predict_proba(X_val_scaled)[:, 1]     # P(y=1)


# =========================
# 6. 生成并保存逐句预测结果到JSON（核心修改）
# =========================
results = []
for meta, pred01 in zip(metas, y_pred):
    pred_label = "machine" if int(pred01) == 1 else "human"
    results.append({
        "id": meta["id"],
        "pid": meta["pid"],
        "text": meta["text"],
        "pred_label": pred_label,
        "gt_label": meta["gt_label"],
        "domain": meta["domain"]
    })

with open(OUTPUT_PRED_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved prediction JSON to: {OUTPUT_PRED_JSON}")
print(f"Total sentences: {len(results)}")


# =========================
# 7. （可选保留）计算并打印验证指标
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
# 8. （可选保留）绘制ROC和PR曲线
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
plt.savefig("qwen_validation_roc.png")
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
plt.savefig("qwen_validation_pr.png")
plt.show()

print(f"\nAUC-ROC: {roc_auc:.4f}")
print(f"AUC-PR:  {pr_auc:.4f}")
print("Validation completed!")
