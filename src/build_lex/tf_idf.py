import math
import json

from tqdm import tqdm
from collections import defaultdict
from build_lex.tokenizer import HanLP

class TFIDFVectorizer:
    def __init__(self, tokenizer: str = 'hanlp'):
        if tokenizer == 'hanlp':
            self.tokenizer = HanLP()
        else:
            raise ValueError("Unsupported tokenizer. Currently only 'hanlp' is supported.")
        
        self.word_idf = {}  # 存储每个词的IDF值
        self.vocab = set()   # 词汇表
    
    def fit(self, documents):
        """
        计算语料库中所有词的IDF值
        :param documents: 文档列表，每个文档是字符串
        """
        # 1. 预处理文档：分词和小写化
        tokenized_docs = []
        doc_freq = defaultdict(int)  # 存储包含每个词的文档数
        total_docs = len(documents)

        for doc in tqdm(documents):
            tokens = self.tokenizer.cut(doc)
            if not tokens:
                continue
            tokenized_docs.append(tokens)
            
            # 更新词汇表
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] += 1
        
        # 2. 计算每个词的IDF (使用平滑处理)
        self.vocab = set(doc_freq.keys())
        for word in self.vocab:
            # IDF公式: log(文档总数 / (包含该词的文档数 + 1)) + 1
            self.word_idf[word] = math.log(total_docs / (doc_freq[word] + 1)) + 1
    
    def transform(self, document):
        """
        将单个文档转换为TF-IDF向量
        :param document: 输入文档字符串
        :return: 字典格式的TF-IDF向量 {词: TF-IDF值}
        """
        # 1. 预处理文档
        tokens = document.lower().split()
        doc_length = len(tokens)
        
        # 2. 计算词频(TF)
        word_tf = defaultdict(int)
        for token in tokens:
            word_tf[token] += 1
        
        # 3. 计算TF-IDF
        tfidf_vector = {}
        
        for word, count in word_tf.items():
            # TF公式: 词频 / 文档总词数
            tf = count / doc_length
            
            # 如果词不在词汇表中，使用默认IDF值
            idf = self.word_idf.get(word, 0)
            
            # TF-IDF = TF × IDF
            tfidf_vector[word] = tf * idf
        
        return tfidf_vector

    def fit_transform(self, documents):
        """ 组合fit和transform操作 """
        self.fit(documents)
        return [self.transform(doc) for doc in documents]
    
    def top_k_idf(self, k: int = 10):
        """
        输出IDF值最高的前k个词语
        :param k: 要输出的词语数量
        :return: 包含(word, idf_value)的列表，按IDF降序排列
        """
        sorted_idf = sorted(self.word_idf.items(), key=lambda x: x[1], reverse=True)
        return sorted_idf[:k] if k > 0 else sorted_idf


# ================= 测试代码 =================
if __name__ == "__main__":
    # 示例文档集

    with open('data/full/std/train.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    train_corpus = [item['content'] for item in datas]

    with open('data/full/std/test.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)
    
    test_corpus = [item['content'] for item in datas]
    corpus = train_corpus + test_corpus

    # 1. 创建TF-IDF向量器
    vectorizer = TFIDFVectorizer()
    
    # 2. 训练模型 (计算IDF值)
    vectorizer.fit(corpus)
    
    k = 1000  # 设置要输出的词语数量
    top_k_idf = vectorizer.top_k_idf(k)
    
    print(f"IDF值最高的前{k}个词语:")
    for idx, (word, idf) in enumerate(top_k_idf, 1):
        print(f"{idx}. {word}: {idf:.4f}")
    
    # # 3. 转换文档
    # print("\n文档1的TF-IDF向量:")
    # doc1_vector = vectorizer.transform(corpus[0])
    # for word, score in sorted(doc1_vector.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{word}: {score:.4f}")
    
    # # 4. 转换所有文档
    # all_vectors = vectorizer.fit_transform(corpus)
    # print("\n所有文档的TF-IDF向量:")
    # for i, vec in enumerate(all_vectors):
    #     print(f"\n文档 {i+1}:")
    #     for word, score in sorted(vec.items(), key=lambda x: x[1], reverse=True):
    #         print(f"  {word}: {score:.4f}")

