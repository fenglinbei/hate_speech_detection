import json
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================================================
# 1. 读取数据
# ======================================================
def load_data(path):
    """读取训练数据，兼容不同的JSON格式"""
    data = json.load(open(path, "r", encoding="utf-8"))
    sentences = []
    labels = []
    for item in data:
        # 兼容 list of dicts 格式
        s_list = item.get("sent_and_label", [])
        for s, lab in s_list:
            sentences.append(s)
            labels.append(1 if lab == "machine" else 0)  # machine=1, human=0
    return sentences, np.array(labels)

print("1. Loading data...")
try:
    sentences, labels = load_data("train_split.json")
except FileNotFoundError:
    print("❌ 错误: 找不到 train_split.json，请先运行 split_dataset.py")
    exit()

print(f"   共加载 {len(sentences)} 条样本")

# ======================================================
# 2. 中文分词 (TF-IDF 必须步骤)
# ======================================================
print("2. Tokenizing (Jieba)...")
def jieba_tokenize(text):
    return " ".join(jieba.lcut(text))

# 对所有句子进行分词
sentences_cut = [jieba_tokenize(s) for s in sentences]

# ======================================================
# 3. 划分训练/验证集
# ======================================================
print("3. Splitting dataset...")
# stratify=labels 保证验证集正负样本比例与训练集一致
X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    sentences_cut, labels, test_size=0.2, random_state=42, stratify=labels
)

# ======================================================
# 4. TF-IDF 特征提取
# ======================================================
print("4. Vectorizing (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)  # 保持与基线相同的特征数量
X_train_tfidf = vectorizer.fit_transform(X_train_txt)
X_val_tfidf = vectorizer.transform(X_val_txt)

print(f"   特征矩阵维度: {X_train_tfidf.shape}")

# ======================================================
# 5. 训练SVM模型
# ======================================================
print("5. Training SVM...")
# 使用LinearSVC而非SVC，在文本分类任务上通常表现更好且速度更快
clf = LinearSVC(random_state=42, max_iter=10000)  # 增加最大迭代次数确保收敛
clf.fit(X_train_tfidf, y_train)

# ======================================================
# 6. 评估与输出结果 (与基线保持一致的评估指标)
# ======================================================
print("\n6. Evaluating on Validation Set...")
y_pred = clf.predict(X_val_tfidf)

# --- 计算所有指标 ---
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
macro_f1 = f1_score(y_val, y_pred, average="macro")
micro_f1 = f1_score(y_val, y_pred, average="micro")

print("\n" + "="*40)
print("   TF-IDF + SVM Results")  
print("="*40)
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Micro-F1:  {micro_f1:.4f}")
print(f"Macro-F1:  {macro_f1:.4f}")
print("="*40)

# 打印详细分类报告
print("\nDetailed Report:")
print(classification_report(y_val, y_pred, target_names=['Human', 'Machine']))