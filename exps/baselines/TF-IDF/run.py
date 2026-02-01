import math
import json
from collections import Counter
import jieba  # 用于中文分词

from exps.utils.data import load_json

class TFIDFDictClassfier:
    def __init__(self, dict_path: str, corpus_path: str = None):  # 增加语料库路径
        self.categories = []
        self.word2item = {}
        self.idf_dict = {}  # 新增：存储每个词的IDF值
        self.load_dict(dict_path)
        if corpus_path:
            self.compute_idf_from_corpus(corpus_path)  # 新增：预计算IDF

    def load_dict(self, file_path):
        raw_dict = load_json(file_path)
        self.categories = raw_dict['categories']
        self.dict_datas = raw_dict["terms"]
        self.word2item = {}
        for data in self.dict_datas:
            word = data['term']
            categories = data['category'].split(', ')
            definition = data.get('definition', '')
            self.word2item[word] = {
                'word': word,
                'categories': categories,
                'definition': definition
            }
            for category in categories:
                if category not in self.categories:
                    self.categories.append(category)

    def compute_idf_from_corpus(self, corpus_path: str):
        """从训练语料库计算每个词典词的IDF值"""
        # 1. 加载语料库文档
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        documents = [item['content'] for item in corpus_data]
        
        # 2. 统计包含每个词的文档数
        doc_count = len(documents)
        word_doc_count = {word: 0 for word in self.word2item.keys()}
        
        for doc in documents:
            words_in_doc = set(jieba.cut(doc))  # 使用分词并去重
            for word in words_in_doc:
                if word in word_doc_count:
                    word_doc_count[word] += 1
        
        # 3. 计算IDF：log(总文档数 / (包含该词的文档数 + 1))，+1避免除零
        self.idf_dict = {
            word: math.log(doc_count / (count + 1)) 
            for word, count in word_doc_count.items()
        }

    def classify_with_tfidf(self, query: str) -> str:
        """使用TF-IDF加权的分类方法"""
        # 1. 对查询文本分词并计算TF
        words = list(jieba.cut(query))
        word_count = Counter(words)
        total_words = len(words)
        tf_dict = {word: count / total_words for word, count in word_count.items()}
        
        # 2. 计算每个类别的TF-IDF总分
        category_scores = {}
        for word in words:
            if word in self.word2item:  # 如果是词典中的词
                tf = tf_dict.get(word, 0)
                idf = self.idf_dict.get(word, 0)  # 获取预计算的IDF
                tfidf = tf * idf
                
                # 将该词的TF-IDF分加到其所属类别
                categories = self.word2item[word]['categories']
                for category in categories:
                    if category not in category_scores:
                        category_scores[category] = 0
                    category_scores[category] += tfidf
        
        # 3. 返回得分最高的类别
        return max(category_scores.items(), key=lambda x: x[1], default=(None, 0))[0]
    
if __name__ == "__main__":
    dict_classifier = TFIDFDictClassfier(
        dict_path="data/lexicon/annotated_lexicon.json",
        corpus_path="data/full/std/train.json"  # 提供训练语料库路径
    )
    
    test_sentences = [
        "我讨厌这个国家的某些人。",
        "所有人都应该被平等对待。",
        "那个群体真是太可恶了！",
        "傻逼黑哥哥"
    ]
    
    for sentence in test_sentences:
        category = dict_classifier.classify_with_tfidf(sentence)
        print(f"句子: {sentence}\n分类结果: {category}\n")