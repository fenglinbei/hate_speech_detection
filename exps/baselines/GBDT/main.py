import torch
import json
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import re
from collections import Counter
import torch.nn.functional as F
from tqdm.auto import tqdm  # 添加tqdm导入

from exps.utils.data import *

BEST_MODEL_PATH = 'exps/GBDT/model/best_model.pth'

label_dict = {
    "Racism": 0,
    "Region": 1,
    "LGBTQ": 2,
    "Sexism": 3,
    "others": 4,
    "non-hate": 5
}

binary_label_dict = {
    "hate": 0,
    "non-hate": 1
}

label_dict_inv = {v: k for k, v in label_dict.items()}
binary_label_dict_inv = {v: k for k, v in binary_label_dict.items()}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 配置参数
class Config:
    def __init__(self):
        self.embedding_dim = 512
        self.hidden_dim = 256
        self.num_layers = 3
        self.num_classes = 6
        self.batch_size = 128
        self.learning_rate = 0.001
        self.num_epochs = 20
        self.dropout = 0.3
        self.max_seq_length = 128
        self.vocab_size = 21128
        self.class_names = ['Racism', 'Region', 'LGBTQ', 'Sexism', 'others', 'non-hate']

config = Config()

# 中文文本预处理类
class ChineseTextProcessor:
    def __init__(
            self, 
            stop_words_path: str = "exps/GBDT/stopwords_cn.txt"
            ):
        # 添加一些中文停用词（可以根据需要扩展）
        with open(stop_words_path, 'r', encoding='utf-8') as f:
            self.stop_words = set(f.read().splitlines())
        
    def clean_text(
            self, 
            text: str
            ) -> str:
        """清洗文本"""
        # 去除特殊字符和标点
        text = re.sub(r'[^\w\s]', '', text)
        # 去除数字
        text = re.sub(r'\d+', '', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(
            self, 
            text: str
            ) -> list[str]:
        """中文分词"""
        text = self.clean_text(text)
        words = jieba.cut(text)
        # 过滤停用词和单字
        words = [word for word in words if word not in self.stop_words]
        return words

# 数据集类
class HateSpeechDataset(Dataset):
    def __init__(
            self, 
            texts: list[str], 
            labels: list[int], 
            word2idx: dict,
            max_length: int
            ):
        
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 将文本转换为索引序列
        tokens = text_processor.tokenize(text)
        indices = [self.word2idx.get(word, 1) for word in tokens]  # 1代表UNK
        
        # 填充或截断
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))  # 0代表PAD
        else:
            indices = indices[:self.max_length]
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# 1. CNN模型（基于Kim的CNN文本分类）
class CNN_Text(nn.Module):
    def __init__(self, config, vocab_size, pretrained_embeddings=None):
        super(CNN_Text, self).__init__()
        self.config = config
        
        # 嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        
        # 卷积层：不同大小的卷积核
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (kernel_size, config.embedding_dim)) 
            for kernel_size in [3, 4, 5]
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(300, config.num_classes)  # 3个卷积层，每个100个特征图
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_length, embedding_dim)
        
        # 对每个卷积核进行卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch_size, 100, seq_length - kernel_size + 1, 1)
            conv_out = F.relu(conv_out.squeeze(3))  # (batch_size, 100, seq_length - kernel_size + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch_size, 100)
            conv_outputs.append(pooled)
        
        # 拼接所有卷积层的输出
        x = torch.cat(conv_outputs, 1)  # (batch_size, 300)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 2. LSTM模型
class LSTM_Text(nn.Module):
    def __init__(self, config, vocab_size, pretrained_embeddings=None):
        super(LSTM_Text, self).__init__()
        self.config = config
        
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, config.embedding_dim, padding_idx=0
            )
        
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout
        )
        
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)
    
    def forward(self, x):
        # x 是词索引 (batch_size, seq_length)
        x_indices = x
        
        x = self.embedding(x_indices)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(x)     # (batch, seq_len, hidden_dim*2)
        
        # mask: 非 PAD 的位置为 1，PAD 为 0
        mask = (x_indices != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
        
        # 将 PAD 位置的输出置零，然后做平均池化
        lstm_out = lstm_out * mask
        sum_out = lstm_out.sum(dim=1)          # (batch, hidden_dim*2)
        lengths = mask.sum(dim=1)              # (batch, 1)
        lengths = torch.clamp(lengths, min=1)  # 防止除零
        x = sum_out / lengths                  # (batch, hidden_dim*2)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 3. FastText模型
class FastText(nn.Module):
    def __init__(self, config, vocab_size, pretrained_embeddings=None):
        super(FastText, self).__init__()
        self.config = config
        
        # 嵌入层
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        
        self.fc = nn.Linear(config.embedding_dim, config.num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # 平均池化
        x = torch.mean(x, dim=1)  # (batch_size, embedding_dim)
        x = self.fc(x)
        return x

def compute_class_weights(labels):
    counter = Counter(labels)
    total = sum(counter.values())
    weights = [total / counter[i] for i in range(len(counter))]
    
    # 归一化，使最大值为 1
    max_w = max(weights)
    weights = [w / max_w for w in weights]
    return torch.tensor(weights, dtype=torch.float)

# 训练函数
def train_model(model, train_loader, val_loader, config, weights):
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_val_f1 = 0
    train_losses = []
    val_f1_scores = []
    patience = 5
    patience_counter = 0
    
    # 添加epoch进度条
    epoch_pbar = tqdm(range(config.num_epochs), desc="Training Epochs")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        # 添加batch进度条
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}', leave=False)
        
        for batch_idx, (data, target) in enumerate(batch_pbar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 更新batch进度条描述
            batch_pbar.set_postfix({'Batch Loss': f'{loss.item():.4f}'})
        
        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        
        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc='Validating', leave=False)
        
        with torch.no_grad():
            for data, target in val_pbar:
                output = model(data)
                pred = output.argmax(dim=1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        val_f1_scores.append(val_f1)
        train_losses.append(total_loss / len(train_loader))
        
        # 更新epoch进度条描述
        epoch_pbar.set_postfix({
            'Train Loss': f'{total_loss/len(train_loader):.4f}', 
            'Val F1': f'{val_f1:.4f}'
        })
        
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}')
        
        # 保存最佳模型
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return train_losses, val_f1_scores

# 特征提取函数（用于GBDT）
def extract_features(model, dataloader, config):
    model.eval()
    features = []
    labels = []

    feature_pbar = tqdm(dataloader, desc='Extracting Features')

    with torch.no_grad():
        for data, target in feature_pbar:
            # 如果你有 GPU，这里最好 .to(device)
            x_indices = data  # (batch, seq_len)

            if isinstance(model, CNN_Text):
                x = model.embedding(x_indices)
                x = x.unsqueeze(1)
                conv_outputs = []
                for conv in model.convs:
                    conv_out = conv(x)
                    conv_out = F.relu(conv_out.squeeze(3))
                    pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                    conv_outputs.append(pooled)
                feature = torch.cat(conv_outputs, 1)

            elif isinstance(model, LSTM_Text):
                # x: (batch, seq_len, embed_dim)
                x = model.embedding(x_indices)
                lstm_out, _ = model.lstm(x)  # (batch, seq_len, hidden_dim*2)

                # mask: 非 PAD 位置为 True，PAD 为 False
                mask = (x_indices != 0).unsqueeze(-1)  # (batch, seq_len, 1), dtype=bool

                # --- mean pooling ---
                # 将 bool mask 转成 float，用于加权求和
                mask_float = mask.float()  # (batch, seq_len, 1)

                # 对 PAD 位置置零，然后按长度平均
                lstm_out_masked = lstm_out * mask_float  # 广播到 (batch, seq_len, hidden_dim*2)

                sum_out = lstm_out_masked.sum(dim=1)      # (batch, hidden_dim*2)
                lengths = mask_float.sum(dim=1)           # (batch, 1)
                lengths = torch.clamp(lengths, min=1.0)   # 防止除零
                mean_out = sum_out / lengths              # (batch, hidden_dim*2)

                # --- max pooling ---
                # 使用 masked_fill 做广播掩码：PAD 位置填上极小值
                lstm_out_for_max = lstm_out.masked_fill(~mask, -1e9)  # mask 是 (batch, seq_len, 1)，会在最后一维广播到 hidden_dim*2

                max_out, _ = lstm_out_for_max.max(dim=1)  # (batch, hidden_dim*2)

                # 拼接 mean + max
                feature = torch.cat([mean_out, max_out], dim=-1)  # (batch, hidden_dim*4)

            else:  # FastText
                x = model.embedding(x_indices)
                feature = torch.mean(x, dim=1)

            features.extend(feature.cpu().numpy())
            labels.extend(target.cpu().numpy())

    return np.array(features), np.array(labels)

def read_data(file_path: str = "data/full/std/train.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    # 准备转换后的数据
    texts, lebels = [], []
    
    # 添加数据读取进度条
    data_pbar = tqdm(data1, desc="Reading Data")
    
    for item in data_pbar:
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
        
        texts.append(content)
        lebels.append(label)
    
    return texts, lebels

# 主函数
def main():
    set_seed(42)
    # 初始化文本处理器
    global text_processor
    text_processor = ChineseTextProcessor()
    
    # 1. 读取数据
    print("读取数据...")
    texts, labels = read_data("data/full/std/train.json")
    weights = compute_class_weights(labels)
    
    # 2. 构建词汇表
    print("构建词汇表...")
    word_counter = Counter()
    
    # 添加分词进度条
    tokenize_pbar = tqdm(texts, desc="Tokenizing Texts")
    
    for text in tokenize_pbar:
        words = text_processor.tokenize(text)
        word_counter.update(words)
    
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counter.most_common(config.vocab_size-2)]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 3. 准备数据加载器
    print("准备数据加载器...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = HateSpeechDataset(train_texts, train_labels, word2idx, config.max_seq_length)
    val_dataset = HateSpeechDataset(val_texts, val_labels, word2idx, config.max_seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 4. 训练不同模型
    vocab_size = len(vocab)
    
    # print("训练CNN模型...")
    # cnn_model = CNN_Text(config, vocab_size)
    # train_model(cnn_model, train_loader, val_loader, config)
    
    print("训练LSTM模型...")
    lstm_model = LSTM_Text(config, vocab_size)
    train_model(lstm_model, train_loader, val_loader, config, weights)
    
    # print("训练FastText模型...")
    # fasttext_model = FastText(config, vocab_size)
    # train_model(fasttext_model, train_loader, val_loader, config)
    
    # 5. 提取特征用于GBDT（论文中的关键思想）
    print("提取特征用于GBDT...")
    
    # 加载最佳模型
    lstm_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    
    # 提取特征
    train_features, train_labels_gbdt = extract_features(lstm_model, train_loader, config)
    val_features, val_labels_gbdt = extract_features(lstm_model, val_loader, config)
    
    # 6. 使用GBDT（需要安装scikit-learn）
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import classification_report
        
        print("训练GBDT分类器...")
        gbdt = GradientBoostingClassifier(
            n_estimators=200,      # 树的数量多一些，但 depth 浅
            learning_rate=0.05,   # 小学习率 + 多树，一般泛化更好
            max_depth=3,          # 树深度不宜太深，防止过拟合
            subsample=0.8,        # Stochastic Gradient Boosting，有正则化效果
            max_features='sqrt',  # 每次分裂只看部分特征，提升泛化能力
            random_state=42,
            verbose=1  # 打印每棵树的训练信息
        )
        
        # 添加GBDT训练进度条（通过verbose控制）
        gbdt.fit(train_features, train_labels_gbdt)
        
        # 预测
        gbdt_preds = gbdt.predict(val_features)
        
        print("GBDT分类结果:")
        print(classification_report(val_labels_gbdt, gbdt_preds, target_names=config.class_names, digits=4))
        
    except ImportError:
        print("scikit-learn未安装，跳过GBDT部分")
    
    # 7. 最终模型评估
    print("最终模型评估:")
    models = {
        # 'CNN': cnn_model,
        'LSTM': lstm_model,
        # 'FastText': fasttext_model
    }
    
    for name, model in models.items():
        model.eval()
        all_preds = []
        all_targets = []
        
        # 添加评估进度条
        eval_pbar = tqdm(val_loader, desc=f'Evaluating {name}')
        
        with torch.no_grad():
            for data, target in eval_pbar:
                output = model(data)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        f1 = f1_score(all_targets, all_preds, average='weighted')
        print(f'{name}模型 F1分数: {f1:.4f}')

if __name__ == '__main__':
    main()