# AI生成文本检测项目

本项目用于检测AI生成的文本，提供了多种模型实现以及基准对比模型的实现：

- **TF-IDF + SVM**：基于TF-IDF特征和支持向量机的文本分类
- **TF-IDF + Logistic Regression**：基于TF-IDF特征和逻辑回归的文本分类
- **SBERT + XGBoost**：基于Sentence-BERT语义嵌入和XGBoost的文本分类
- **SBERT + MLP + XGBoost**：基于SBERT嵌入、MLP特征增强和XGBoost的文本分类
- **Qwen + XGBoost**：基于Qwen3-4B大模型嵌入和XGBoost的文本分类

## 项目结构

```
DM/
├── TF-IDF_SVM.py           # TF-IDF + SVM 实现
├── TF-IDF_LR.py            # TF-IDF + Logistic Regression 实现
├── sbert_xgb.py            # SBERT + XGBoost 实现
├── sbert_mlp_xgb.py        # SBERT + MLP + XGBoost 实现
├── qwen_xgb.py             # Qwen3-4B + XGBoost 实现
├── split_dataset.py        # 数据集划分脚本
├── train_split.json        # 训练数据集
├── test_split.json         # 测试数据集
├── all-MiniLM-L6-v2/       # SBERT模型文件
├── requirements.txt        # Python依赖包
└── README.md               # 项目说明文档
```

## 环境配置

### 1. 创建Python环境

```bash
# 创建虚拟环境（推荐使用conda或venv）
conda create -n dm python=3.10
conda activate dm
```

### 2. 安装依赖包

```bash
# 一键快速安装依赖
pip install -r requirements.txt
```

## 使用方法

### 我们的方法

- 在运行sbert_xgb.py和sbert_mlp_xgb.py之前，需要先下载SBERT模型文件all-MiniLM-L6-v2并放置在项目根目录下，或者将加载模型的代码进行修改。
``` python
model_path = "./all-MiniLM-L6-v2"
sbert = SentenceTransformer(model_path)
#替换成
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```
在运行qwen_xgb.py之前，需要修改Qwen模型的加载路径为：
``` python
MODEL_PATH = "/mnt/Data/multi-agent/Qwen/Qwen3-4B" #将路径改成自己的模型路径
```
```bash
# SBERT + XGBoost（需要GPU加速）
python sbert_xgb.py

# SBERT + MLP + XGBoost（需要GPU加速）
python sbert_mlp_xgb.py

# Qwen3-4B + XGBoost（需要大量GPU显存，至少16GB）
python qwen_xgb.py
```
### 基准对比的方法

```bash
# TF-IDF + Logistic Regression（推荐首选）
python TF-IDF_LR.py

# TF-IDF + SVM
python TF-IDF_SVM.py
```
### 快速测试

```bash
# 快速测试Qwen3 4B + XGBoost模型的效果
python test.py 
```

## 数据集格式

项目支持以下JSON数据格式：

```json
[
  {
    "sent_and_label": [
      ["句子1", "human"],
      ["句子2", "machine"],
      ["句子3", "machine"]
    ]
  },
  {
    "sent_and_label": [
      ["句子4", "human"],
      ["句子5", "machine"]
    ]
  }
]
```

标签说明：
- `human`：人类撰写的文本（标签为0）
- `machine`：AI生成的文本（标签为1）

## 依赖包说明

核心依赖包包括：

- **jieba**：中文分词
- **numpy**：数值计算
- **scikit-learn**：机器学习工具包（包含SVM、LR、评估指标等）
- **xgboost**：梯度提升树模型
- **sentence-transformers**：SBERT句子嵌入模型
- **transformers**：Hugging Face Transformers库
- **torch**：PyTorch深度学习框架
- **matplotlib**：结果可视化
- **joblib**：模型保存与加载

## 注意事项

1. **SBERT/Qwen模型**：首次运行时会自动下载预训练模型
2. **GPU支持**：高级模型（SBERT/XGBoost、Qwen/XGBoost）需要GPU加速
3. **内存要求**：Qwen3-4B模型需要至少16GB GPU显存
4. **路径设置**：确保工作目录为项目根目录，不要包含中文或特殊字符

## 评估指标

所有模型都会输出以下评估指标：

- **Accuracy**：准确率
- **Precision**：精确率
- **Recall**：召回率
- **Micro-F1**：Micro-F1分数
- **Macro-F1**：Macro-F1分数
- **AUC-ROC**：ROC曲线下面积
- **AUC-PR**：PR曲线下面积


